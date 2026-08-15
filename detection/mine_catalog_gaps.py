"""
mine_catalog_gaps.py

Use the detector to find bursts the Monstein catalog never recorded, so a human
only has to CONFIRM them instead of hunting for them.

Why this is the fastest way to buy clean labels right now
---------------------------------------------------------
The catalog omits roughly 36% of real events (PHASE1_RESULTS.md): of 34 model
detections that overlapped no catalog entry, 32 were confirmed real by eye, and
every one of the 24 at conf>=0.1 was real. So a high-confidence detection with
no catalog entry is not a false alarm -- it is an unlabelled burst, and the
model already knows where it is.

That matters more than any model change available to us. The 2026-08-14 batch
measured the run-to-run noise floor at sigma=0.056 AP50, which is the same size
as every effect we have tried to detect; nothing on the model side is
measurable at 570 training boxes. Labels are the binding constraint, and these
are the cheapest labels on the table -- confirm/reject beats locating from
scratch.

Two biases, both real, both in the output
-----------------------------------------
1. CIRCULAR SUPPRESSION on training windows. The model was trained to stay
   quiet on its own train negatives, and does: it fires on 4.2% of unseen VAL
   negatives but only 1.9% of train ones. So what this finds on train windows is
   a LOWER bound, and the events it misses there are exactly the ones whose
   wrong supervision bit hardest. The `split` column records which side each
   candidate came from -- do not read the train/val counts as a real difference
   in burst rate. A rigorous version needs cross-fold scanning.
2. NOT EVERY CANDIDATE IS REAL. The 94% confirmation rate was measured on val
   detections at conf>=0.05 and is a rate, not a guarantee; the confirmation
   rate at 0.05-0.1 was 80%. That is why this writes review sheets and a verdict
   column rather than appending straight to boxes.csv. Nothing here enters
   training without a human y/n.

Overlapping predictions are collapsed with the same maxconf rule evaluate.py
uses, otherwise one burst arrives as three near-duplicate candidates and the
reviewer pays for it three times.

Usage:
  python detection/mine_catalog_gaps.py <weights> <window_dir> <out_dir>
      [--min-conf 0.10] [--sheets-top 120] [--imgsz 640]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_yolo_dataset import render_png, keep_dominant_config, drop_short_windows
from evaluate import iou_1d, maxconf_collapse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default=None,
                    help="detector weights; not needed with --from-csv, which skips inference")
    ap.add_argument("window_dir"); ap.add_argument("out_dir")
    ap.add_argument("--min-conf", type=float, default=0.10,
                    help="confidence floor for a candidate. Default 0.10 because that is where "
                         "the measured confirmation rate hit 24/24; 0.05-0.1 was 80%%, still "
                         "useful but no longer near-certain.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--img-size", nargs=2, type=int, default=[640, 640], metavar=("H", "W"),
                    help="render size -- must match what the model was TRAINED on, or the time "
                         "axis is scaled differently than it learned")
    ap.add_argument("--window-s", type=float, default=900.0)
    ap.add_argument("--zoom-s", type=float, default=150.0)
    ap.add_argument("--sheets-top", type=int, default=120,
                    help="render this many of the highest-confidence candidates for review "
                         "(the CSV always lists all of them)")
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--dataset-dir", default="data/ecallisto/yolo7_single",
                    help="only used to label each candidate train/val for the bias note above")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--from-csv", default=None,
                    help="re-render sheets from an existing gap_candidates.csv instead of "
                         "re-running inference. Use this to render a different subset or to fix "
                         "a too-small --sheets-top: it never rewrites the CSV, so verdicts "
                         "already filled in by hand are safe.")
    ap.add_argument("--subset", default="all", choices=["all", "new-info", "covered"],
                    help="which candidates to render (--from-csv only). 'new-info' = candidates "
                         "on windows that DO have catalog entries, i.e. the catalog logged one "
                         "burst here and missed another -- these carry information the earlier "
                         "contaminated-negative pass could not reach. 'covered' = candidates on "
                         "windows with no catalog entry at all; those windows are already listed "
                         "in contaminated_negatives.csv, so confirming them changes nothing.")
    ap.add_argument("--sheet-prefix", default=None)
    args = ap.parse_args()
    if not args.from_csv and not args.weights:
        ap.error("weights is required unless --from-csv is given")

    windows = pd.read_csv(os.path.join(args.window_dir, "windows.csv"), dtype={"date": str})
    boxes = pd.read_csv(os.path.join(args.window_dir, "boxes.csv"), dtype={"date": str})

    # Match the training distribution: a window from a minority instrument config
    # has a different pixel scale, so the model's notion of drift slope does not
    # transfer to it and its detections would not be trustworthy.
    windows = windows[~windows.excluded.astype(bool)]
    windows = keep_dominant_config(windows)
    windows = drop_short_windows(windows, 0.99)
    print(f"scanning {len(windows)} windows")

    # which split each window landed in, for the circular-suppression caveat
    split_of = {}
    for sp in ("train", "val"):
        d = os.path.join(args.dataset_dir, "images", sp)
        if os.path.isdir(d):
            for f in os.listdir(d):
                split_of[f[:-4]] = sp

    # catalog boxes in normalized time coords, so they compare directly against
    # the model's xyxyn output regardless of each window's column count
    cat: dict[str, list[tuple[float, float]]] = {}
    ncols = windows.set_index("file_name")["n_cols"].to_dict()
    for b in boxes.itertuples():
        n = ncols.get(b.file_name)
        if n:
            cat.setdefault(b.file_name, []).append((b.box_start_col / n, b.box_end_col / n))

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        # x0/x1 aren't stored (they're redundant); rebuild from start_s/end_s,
        # which were rounded to 0.1s = 0.011% of a 900s window -- far below one
        # pixel at any render size, so the boxes land where they did originally.
        df["x0"] = df.start_s / args.window_s
        df["x1"] = df.end_s / args.window_s
        if args.subset == "new-info":
            df = df[df.n_catalog_boxes > 0]
        elif args.subset == "covered":
            df = df[df.n_catalog_boxes == 0]
        df = df.sort_values("conf", ascending=False).head(args.sheets_top)
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"re-rendering {len(df)} candidate(s) from {args.from_csv} "
              f"(subset={args.subset}); the CSV itself is not modified")
        render_sheets(df, cat, args, args.sheet_prefix or f"gap_{args.subset}")
        return

    model = YOLO(args.weights)
    rows = list(windows.itertuples())
    cands = []
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        imgs, keep = [], []
        for w in chunk:
            p = os.path.join(args.window_dir, w.file_name)
            if not os.path.exists(p):
                continue
            # GRAY2BGR is not cosmetic: the trained model takes 3 channels, and
            # this reproduces exactly what it saw during training, where
            # render_png's grayscale went to disk as a PNG and came back through
            # cv2.imread, which replicates the single channel into BGR. Feeding
            # the array straight in would be a 1-channel tensor and a shape error.
            imgs.append(cv2.cvtColor(render_png(np.load(p), tuple(args.img_size)),
                                     cv2.COLOR_GRAY2BGR))
            keep.append(w)
        if not imgs:
            continue
        for w, res in zip(keep, model.predict(imgs, conf=args.min_conf, imgsz=args.imgsz,
                                              verbose=False)):
            # xyxyn is [x1,y1,x2,y2]; [::2] keeps the two time coords. Frequency
            # carries no information here -- every box is full height by design.
            preds = [(0, float(b.conf), *b.xyxyn[0].tolist()[::2]) for b in res.boxes]
            for _, cf, x0, x1 in maxconf_collapse(preds, 0.1):
                if any(iou_1d(x0, x1, a, b_) > 0 for a, b_ in cat.get(w.file_name, [])):
                    continue           # already in the catalog -- not a gap
                cands.append(dict(file_name=w.file_name, stem=w.file_name[:-4],
                                  date=w.date, conf=round(cf, 4),
                                  start_s=round(x0 * args.window_s, 1),
                                  end_s=round(x1 * args.window_s, 1),
                                  x0=x0, x1=x1,
                                  n_catalog_boxes=len(cat.get(w.file_name, [])),
                                  split=split_of.get(w.file_name[:-4], "unused")))
        if (i // args.batch) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(rows)} windows, {len(cands)} candidates so far",
                  flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    if not cands:
        print("no candidates"); return
    df = pd.DataFrame(cands).sort_values("conf", ascending=False).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    print(f"\n{len(df)} candidate bursts absent from the catalog")
    print("\nby confidence band:")
    band = pd.cut(df.conf, [args.min_conf, 0.2, 0.35, 0.5, 1.0], include_lowest=True)
    print(band.value_counts().sort_index().to_string())
    print("\nby split (train counts are SUPPRESSED -- see module docstring):")
    print(df.split.value_counts().to_string())
    print(f"\ncandidates on windows with no catalog event at all (pure negatives): "
          f"{int((df.n_catalog_boxes == 0).sum())}")

    df["verdict"] = ""      # y = real burst the catalog missed / n = false alarm / ? = unclear
    df["note"] = ""
    csv = os.path.join(args.out_dir, "gap_candidates.csv")
    df.drop(columns=["x0", "x1"]).to_csv(csv, index=False)
    print(f"\nfull list (fill the verdict column with y/n/?): {csv}")

    render_sheets(df.head(args.sheets_top), cat, args, "gap_sheet")


def render_sheets(sel, cat, args, prefix):
    W = args.window_s
    for sheet in range((len(sel) + args.per_sheet - 1) // args.per_sheet):
        chunk = sel.iloc[sheet * args.per_sheet:(sheet + 1) * args.per_sheet]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(15, 2.1 * len(chunk)),
                                 gridspec_kw={"width_ratios": [2, 1]})
        axes = np.atleast_2d(axes)
        for (ax_full, ax_zoom), r in zip(axes, chunk.itertuples()):
            img = render_png(np.load(os.path.join(args.window_dir, r.file_name)),
                             tuple(args.img_size))
            h, _ = img.shape
            for ax in (ax_full, ax_zoom):
                ax.imshow(img, cmap="viridis", aspect="auto", extent=[0, W, h, 0])
                ax.add_patch(Rectangle((r.x0 * W, 0), (r.x1 - r.x0) * W, h,
                                       fill=False, ec="red", lw=2.0, ls="--"))
                ax.tick_params(labelsize=5)
            for a_, b_ in cat.get(r.file_name, []):
                ax_full.add_patch(Rectangle((a_ * W, 0), (b_ - a_) * W, h,
                                            fill=False, ec="lime", lw=1.8))
            ax_full.set_title(f"#{r.id}  conf={r.conf:.2f}  {r.split}  "
                              f"{r.stem[4:]}", fontsize=7)
            c0 = (r.x0 + r.x1) / 2 * W
            ax_zoom.set_xlim(max(0, c0 - args.zoom_s), min(W, c0 + args.zoom_s))
            ax_zoom.set_title(f"#{r.id} zoom +/-{args.zoom_s:.0f}s", fontsize=7)
        # ASCII only: this matplotlib has no CJK glyphs and renders them as tofu
        fig.suptitle("red dashed = model found a burst the catalog does not list   "
                     "green = catalog entries in the same window   "
                     "-> is there really a burst inside the red box?", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = os.path.join(args.out_dir, f"{prefix}_{sheet + 1:02d}.png")
        fig.savefig(out, dpi=105); plt.close(fig)
        print(f"  {out}")


if __name__ == "__main__":
    main()
