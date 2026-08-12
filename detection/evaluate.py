"""
evaluate.py

Report a trained detector two ways at once:

  (1) STANDARD CV METRICS -- AP@0.50 and AP@0.50:0.95, the numbers every
      detection paper prints, so results stay comparable and explainable.
      AP@0.30 is added alongside because our labels carry a known ~40s median
      timing error and the boxes are full-height, which makes IoU purely
      temporal: two equal-width intervals offset by d have IoU=(w-d)/(w+d), so
      IoU>=0.5 demands d <= w/3 -- only 10s for the 25th-percentile 30s burst.
      A ceiling analysis (see TRAINING_PLAN.md) puts the best achievable AP@0.50
      on this label set at ~0.485 and AP@0.95 at exactly 0.0.

  (2) OPERATIONAL METRICS -- what a burst-alert user actually needs, and what
      AP hides. AP collapses "missed the burst entirely" and "found it, 21s
      off" into the same zero. These separate them:
        detection rate  : fraction of true bursts with ANY sufficiently
                          overlapping prediction  -> how much is missed
        timing error    : for those, |predicted centre - true centre| in
                          seconds -> how precisely it is located
        false alarms/h  : predictions matching no cataloged burst -> how often
                          the user is bothered
      Note the false-alarm figure is an UPPER bound on real error: the Monstein
      catalog demonstrably omits real bursts (TRAINING_PLAN.md open question
      #8), so some "false alarms" are correct detections of unlisted events.

Usage:
    python detection/evaluate.py <weights> <data.yaml> [--imgsz 640] [--rect]
        [--conf 0.001] [--min-overlap 0.1]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from ultralytics import YOLO

CLASSES = ["II", "III", "V"]


def load_gt(labels_dir: str) -> dict[str, list[tuple[int, float, float]]]:
    """{image stem: [(class, x0, x1), ...]} in normalized coords. Only the time
    axis is kept -- every box is full height, so y carries no information."""
    gt = {}
    for p in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        rows = []
        for line in open(p):
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            cx, w = float(cx), float(w)
            rows.append((int(c), cx - w / 2, cx + w / 2))
        gt[os.path.basename(p)[:-4]] = rows
    return gt


def iou_1d(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def average_precision(matched: list[int], confs: list[float], n_gt: int) -> float:
    """All-point-interpolated AP (the COCO/VOC2010 convention ultralytics uses)."""
    if n_gt == 0 or not confs:
        return float("nan") if n_gt == 0 else 0.0
    order = np.argsort(-np.asarray(confs))
    tp = np.asarray(matched, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    # make precision monotonically decreasing, then integrate over recall
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rec = np.concatenate(([0.0], rec))
    prec = np.concatenate(([prec[0] if len(prec) else 1.0], prec))
    return float(np.sum(np.diff(rec) * prec[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("data")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rect", action="store_true")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--min-overlap", type=float, default=0.1,
                    help="minimum IoU for the operational 'detected' test -- not the AP "
                         "threshold. Guards against a box that merely grazes the truth.")
    ap.add_argument("--class-agnostic", action="store_true",
                    help="score DETECTION+LOCALIZATION only: collapse II/III/V into one 'burst' "
                         "class so a correctly found burst is not penalised for getting the type "
                         "wrong. This is the primary objective -- Arecibo has 1614 Type III but "
                         "only 53 II and 27 V, so classification is data-limited in a way "
                         "detection is not, and mixing the two hides how well the detector "
                         "actually finds bursts. Per-class numbers are still reported separately.")
    ap.add_argument("--window-s", type=float, default=900.0,
                    help="physical duration of one window, for converting normalized "
                         "offsets to seconds and counting false alarms per hour")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(args.data))
    gt = load_gt(os.path.join(root, "labels", "val"))
    classes = CLASSES
    if args.class_agnostic:
        classes = ["burst"]
        gt = {k: [(0, a, b) for _, a, b in v] for k, v in gt.items()}
    images = sorted(glob.glob(os.path.join(root, "images", "val", "*.png")))
    print(f"{len(images)} val images, {sum(len(v) for v in gt.values())} ground-truth boxes")

    model = YOLO(args.weights)
    preds = {}
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        for path, res in zip(batch, model.predict(batch, conf=args.conf, imgsz=args.imgsz,
                                                   rect=args.rect, verbose=False)):
            rows = []
            for b in res.boxes:
                x1, _, x2, _ = b.xyxyn[0].tolist()
                rows.append((0 if args.class_agnostic else int(b.cls), float(b.conf), x1, x2))
            preds[os.path.basename(path)[:-4]] = rows

    # ---- standard AP at several thresholds -------------------------------
    print(f"\n{'class':6s} {'n_gt':>5s} | {'AP@0.30':>8s} {'AP@0.50':>8s} {'AP@0.75':>8s}")
    for ci, cname in enumerate(classes):
        n_gt = sum(1 for v in gt.values() for c, *_ in v if c == ci)
        line = f"{cname:6s} {n_gt:5d} |"
        for thr in (0.30, 0.50, 0.75):
            matched, confs = [], []
            for stem, rows in preds.items():
                g = [(x0, x1) for c, x0, x1 in gt.get(stem, []) if c == ci]
                used = set()
                for c, cf, p0, p1 in sorted([r for r in rows if r[0] == ci], key=lambda r: -r[1]):
                    best, bi = 0.0, -1
                    for k, (x0, x1) in enumerate(g):
                        if k in used:
                            continue
                        v = iou_1d(p0, p1, x0, x1)
                        if v > best:
                            best, bi = v, k
                    hit = best >= thr
                    if hit:
                        used.add(bi)
                    matched.append(int(hit)); confs.append(cf)
            line += f" {average_precision(matched, confs, n_gt):8.3f}"
        print(line)

    # ---- operational metrics --------------------------------------------
    print(f"\noperational view (conf>={args.conf}, 'detected' = IoU >= {args.min_overlap}):")
    print(f"{'class':6s} {'n_gt':>5s} {'detected':>9s} {'|Δt| median':>12s} {'|Δt| p80':>9s}")
    total_fa = 0
    for ci, cname in enumerate(classes):
        n_gt = det = 0; offs = []
        for stem, g_rows in gt.items():
            g = [(x0, x1) for c, x0, x1 in g_rows if c == ci]
            p = sorted([r for r in preds.get(stem, []) if r[0] == ci], key=lambda r: -r[1])
            n_gt += len(g)
            for x0, x1 in g:
                best, bp = 0.0, None
                for c, cf, p0, p1 in p:
                    v = iou_1d(p0, p1, x0, x1)
                    if v > best:
                        best, bp = v, (p0, p1)
                if best >= args.min_overlap:
                    det += 1
                    offs.append(abs((bp[0] + bp[1]) / 2 - (x0 + x1) / 2) * args.window_s)
        offs = np.array(offs) if offs else np.array([np.nan])
        print(f"{cname:6s} {n_gt:5d} {det/max(n_gt,1)*100:8.0f}% "
              f"{np.nanmedian(offs):11.1f}s {np.nanpercentile(offs,80):8.1f}s")

    for stem, rows in preds.items():
        g_all = gt.get(stem, [])
        for c, cf, p0, p1 in rows:
            if not any(iou_1d(p0, p1, x0, x1) >= args.min_overlap for cc, x0, x1 in g_all if cc == c):
                total_fa += 1
    hours = len(images) * args.window_s / 3600.0
    print(f"\nfalse alarms: {total_fa} over {hours:.1f} h of val data = {total_fa/hours:.2f}/h")
    print("  (upper bound -- the catalog omits real bursts, so some are correct detections)")


if __name__ == "__main__":
    main()
