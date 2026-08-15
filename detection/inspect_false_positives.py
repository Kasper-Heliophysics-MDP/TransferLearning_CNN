"""
inspect_false_positives.py

Render the model's predictions that match NO ground-truth box, so a human can
say whether each one is actually wrong.

This answers two open questions at once:

  * How good is precision really? The v5_single detector emits 818 boxes for
    224 true ones and scores precision 0.20 at IoU>=0.5. A confidence/NMS sweep
    (NMS 0.1-0.7 x conf 0.05-0.40) moved AP50 only 0.361 -> 0.390, which rules
    out "the extras are duplicate boxes piled on the same burst" -- they are
    spread out in time, i.e. genuinely separate detections somewhere else.
  * How often does the Monstein catalog omit a real burst? (TRAINING_PLAN.md
    open question #8.) Every unlisted-but-real burst the detector finds is
    counted as a false alarm, so the reported precision is a LOWER bound and
    the false-alarm rate an UPPER bound -- by an unknown amount, until now.

The two are the same measurement: a "false positive" that is really a burst is
both a precision underestimate and a catalog omission.

Sampling is stratified by confidence, because "are high-confidence false
positives more often real bursts?" is the question that decides whether the
number can be corrected for at all.

Each panel: left = the whole 15-minute window with the prediction (red) and any
true boxes (green); right = zoomed to +/-150s around the prediction, which is
where the "is there a burst here" judgement actually gets made.

Usage:
  python detection/inspect_false_positives.py <weights> <dataset_dir> <out_dir> [--n 40]
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO


def iou_1d(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("dataset_dir"); ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--window-s", type=float, default=900.0)
    ap.add_argument("--zoom-s", type=float, default=150.0)
    ap.add_argument("--per-sheet", type=int, default=8)
    args = ap.parse_args()

    gt = {}
    for p in sorted(glob.glob(os.path.join(args.dataset_dir, "labels", "val", "*.txt"))):
        rows = []
        for line in open(p):
            if line.strip():
                _, cx, _, w, _ = line.split()
                rows.append((float(cx) - float(w) / 2, float(cx) + float(w) / 2))
        gt[os.path.basename(p)[:-4]] = rows

    images = sorted(glob.glob(os.path.join(args.dataset_dir, "images", "val", "*.png")))
    model = YOLO(args.weights)
    fps = []
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        for path, res in zip(batch, model.predict(batch, conf=args.conf, imgsz=args.imgsz, verbose=False)):
            stem = os.path.basename(path)[:-4]
            g = gt.get(stem, [])
            for b in res.boxes:
                x1, _, x2, _ = b.xyxyn[0].tolist()
                if not any(iou_1d(x1, x2, a, c) > 0 for a, c in g):   # no overlap at all
                    fps.append(dict(stem=stem, img=path, conf=float(b.conf), x0=x1, x1=x2,
                                    n_gt=len(g)))
    fp = pd.DataFrame(fps)
    print(f"{len(images)} val 图, {sum(len(v) for v in gt.values())} 个真值框")
    print(f"完全不与任何真值重叠的预测: {len(fp)}")
    if fp.empty:
        return

    # stratified by confidence so "are confident FPs more often real?" is answerable
    fp["band"] = pd.cut(fp.conf, [0, 0.1, 0.2, 0.35, 1.0], labels=["0.05-0.1", "0.1-0.2", "0.2-0.35", ">0.35"])
    print("\n置信度分布:"); print(fp.band.value_counts().sort_index().to_string())
    per = max(1, args.n // fp.band.nunique())
    sel = (fp.groupby("band", observed=True)
             .apply(lambda d: d.sample(min(per, len(d)), random_state=0), include_groups=False)
             .reset_index(level=0).reset_index(drop=True))
    sel = sel.sort_values("conf", ascending=False).reset_index(drop=True)
    sel.insert(0, "id", range(1, len(sel) + 1))
    print(f"\n抽样 {len(sel)} 个待判断")

    os.makedirs(args.out_dir, exist_ok=True)
    W = args.window_s
    for sheet in range((len(sel) + args.per_sheet - 1) // args.per_sheet):
        chunk = sel.iloc[sheet * args.per_sheet:(sheet + 1) * args.per_sheet]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(15, 2.1 * len(chunk)),
                                 gridspec_kw={"width_ratios": [2, 1]})
        axes = np.atleast_2d(axes)
        for (ax_full, ax_zoom), r in zip(axes, chunk.itertuples()):
            img = cv2.imread(r.img, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            ax_full.imshow(img, cmap="viridis", aspect="auto", extent=[0, W, h, 0])
            for a, c in gt.get(r.stem, []):
                ax_full.add_patch(Rectangle((a * W, 0), (c - a) * W, h, fill=False, ec="lime", lw=1.8))
            ax_full.add_patch(Rectangle((r.x0 * W, 0), (r.x1 - r.x0) * W, h,
                                        fill=False, ec="red", lw=2.0, ls="--"))
            ax_full.set_title(f"#{r.id}  conf={r.conf:.2f}  {r.stem[26:48]}", fontsize=7)
            c0 = (r.x0 + r.x1) / 2 * W
            lo, hi = max(0, c0 - args.zoom_s), min(W, c0 + args.zoom_s)
            ax_zoom.imshow(img, cmap="viridis", aspect="auto", extent=[0, W, h, 0])
            ax_zoom.add_patch(Rectangle((r.x0 * W, 0), (r.x1 - r.x0) * W, h,
                                        fill=False, ec="red", lw=2.0, ls="--"))
            ax_zoom.set_xlim(lo, hi)
            ax_zoom.set_title(f"#{r.id} zoom +/-{args.zoom_s:.0f}s", fontsize=7)
            for a_ in (ax_full, ax_zoom):
                a_.tick_params(labelsize=5)
        # ASCII only -- the matplotlib font here has no CJK glyphs and renders
        # Chinese titles as tofu boxes (hit this once already in range_select.py)
        fig.suptitle("green = catalogued truth    red dashed = model found it, catalog did not"
                     "    -> is there really a burst here?", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = os.path.join(args.out_dir, f"fp_sheet_{sheet + 1:02d}.png")
        fig.savefig(out, dpi=105); plt.close(fig)
        print(f"  {out}")

    tmpl = sel[["id", "conf", "stem", "x0", "x1"]].copy()
    tmpl["start_s"] = (tmpl.x0 * W).round(0); tmpl["end_s"] = (tmpl.x1 * W).round(0)
    tmpl = tmpl.drop(columns=["x0", "x1"])
    tmpl["verdict"] = ""      # y = 确实有 burst(目录漏记) / n = 误报 / ? = 看不准
    tmpl["note"] = ""
    path = os.path.join(args.out_dir, "fp_verdicts.csv")
    tmpl.to_csv(path, index=False)
    print(f"\n判断填这里(verdict 列填 y/n/?): {path}")


if __name__ == "__main__":
    main()
