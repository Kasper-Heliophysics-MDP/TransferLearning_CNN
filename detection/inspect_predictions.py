"""
inspect_predictions.py

Draws ground-truth boxes and model predictions on the same val images.

This exists because the metrics alone cannot answer the question that matters
most for this dataset: how much of the low precision is the model being wrong,
versus the model correctly finding a burst the Monstein catalog never listed?
Those two are indistinguishable in mAP -- an uncatalogued real burst that the
detector finds is scored as a false positive. Only looking at them separates
the two. (TRAINING_PLAN.md open question #8.)

green = ground truth (from the labels the training used)
red   = model prediction, with confidence
"""
from __future__ import annotations

import argparse, glob, os
import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

NAMES = ["II", "III", "V"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("dataset_dir"); ap.add_argument("out")
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--only-positive", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    lps = sorted(glob.glob(f"{args.dataset_dir}/labels/{args.split}/*.txt"))
    if args.only_positive:
        lps = [p for p in lps if os.path.getsize(p) > 0]
    sel = lps[::max(1, len(lps) // args.n)][:args.n]
    print(f"{len(lps)} candidates, showing {len(sel)}")

    n_gt = n_pred = n_pred_far = 0
    fig, axes = plt.subplots(len(sel), 1, figsize=(13, 2.1 * len(sel)))
    axes = np.atleast_1d(axes)
    for ax, lp in zip(axes, sel):
        ip = lp.replace("/labels/", "/images/").replace(".txt", ".png")
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE); H, W = img.shape
        ax.imshow(img, cmap="viridis", aspect="auto")

        gts = []
        for line in open(lp):
            if not line.strip(): continue
            c, cx, cy, w, h = line.split(); cx, w = float(cx), float(w)
            gts.append((cx - w / 2, cx + w / 2))
            ax.add_patch(Rectangle(((cx - w / 2) * W, 0), w * W, H, fill=False, ec="lime", lw=2.0))
            ax.text((cx - w / 2) * W, H * 0.10, NAMES[int(c)], color="lime", fontsize=8, weight="bold")
        n_gt += len(gts)

        r = model.predict(ip, conf=args.conf, verbose=False)[0]
        for b in r.boxes:
            x1, _, x2, _ = b.xyxyn[0].tolist()
            cf = float(b.conf); cl = int(b.cls)
            ax.add_patch(Rectangle((x1 * W, H * 0.02), (x2 - x1) * W, H * 0.96,
                                   fill=False, ec="red", lw=1.6, ls="--"))
            ax.text(x1 * W, H * 0.95, f"{NAMES[cl]} {cf:.2f}", color="red", fontsize=7, weight="bold")
            n_pred += 1
            # does this prediction overlap ANY ground-truth box at all?
            if not any(x1 < g1 and x2 > g0 for g0, g1 in gts):
                n_pred_far += 1
        ax.set_title(os.path.basename(ip)[26:-4], fontsize=7); ax.tick_params(labelsize=6)

    print(f"ground-truth boxes {n_gt}   predictions {n_pred}   "
          f"predictions not overlapping any GT {n_pred_far}")
    fig.suptitle("green = ground truth   red dashed = prediction "
                 "(are the reds wrong, or are they uncatalogued real bursts?)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=105); print("wrote", args.out)


if __name__ == "__main__":
    main()
