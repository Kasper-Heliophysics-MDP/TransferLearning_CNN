"""
inspect_own_fp.py

Render the own-station detections that match no catalogued burst, so a human can
say whether each is a false alarm or a burst the own-station catalog missed.

Same measurement that was already run on e-Callisto, where 32 of 34 such
detections (94%) turned out to be real and the catalog was shown to omit ~36% of
events. Until the same check is done here, the own-station false-alarm rate of
2.4/h is an UPPER bound and nothing more: every unlisted-but-real burst the
detector finds is currently counted against it.

The panels show the `stretch` render -- the one the detector actually ran on --
rather than a prettier version, so what the reviewer judges is what the model
saw. Note this is RAW own-station data: the vertical banding across these images
is broadband RFI, not signal, and it is why the station has a denoising pipeline
at all. A burst is a structure that drifts across frequency; a pure vertical
line spanning the full height at constant width is RFI.

Usage:
    python transfer/inspect_own_fp.py <weights> [--out-dir transfer/fp_review]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "detection"))
sys.path.insert(0, os.path.join(ROOT, "transfer"))
from evaluate import iou_1d, maxconf_collapse          # noqa: E402
from zero_shot_transfer import render_stretch, SIZE     # noqa: E402

W_S = 900.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--window-dir", default="data/burst_data/own_windows")
    ap.add_argument("--out-dir", default="transfer/fp_review")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--min-overlap", type=float, default=0.1)
    ap.add_argument("--zoom-s", type=float, default=150.0)
    ap.add_argument("--per-sheet", type=int, default=8)
    args = ap.parse_args()

    win = pd.read_csv(os.path.join(args.window_dir, "windows.csv")).set_index("file_name")
    box = pd.read_csv(os.path.join(args.window_dir, "boxes.csv"))
    gt: dict[str, list] = {}
    for b in box.itertuples():
        n = int(win.loc[b.file_name].n_cols)
        gt.setdefault(b.file_name, []).append((b.box_start_col / n, b.box_end_col / n))

    model = YOLO(args.weights)
    names = list(win.index)
    imgs = {}
    for f in names:
        arr = np.load(os.path.join(args.window_dir, f))
        fq = np.load(os.path.join(args.window_dir, "freq-" + f[:-4] + ".npy"))
        imgs[f] = render_stretch(arr, fq)

    fps = []
    for i in range(0, len(names), 16):
        chunk = names[i:i + 16]
        batch = [cv2.cvtColor(imgs[f], cv2.COLOR_GRAY2BGR) for f in chunk]
        for f, res in zip(chunk, model.predict(batch, conf=args.conf, imgsz=SIZE, verbose=False)):
            p = maxconf_collapse(
                [(0, float(b.conf), *b.xyxyn[0].tolist()[::2]) for b in res.boxes],
                args.min_overlap)
            for _, c, x0, x1 in p:
                if not any(iou_1d(x0, x1, t0, t1) >= args.min_overlap for t0, t1 in gt.get(f, [])):
                    fps.append(dict(file_name=f, conf=round(c, 4), x0=x0, x1=x1,
                                    start_s=round(x0 * W_S, 1), end_s=round(x1 * W_S, 1),
                                    n_catalog_boxes=len(gt.get(f, []))))

    df = pd.DataFrame(fps).sort_values("conf", ascending=False).reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    if df.empty:
        print("no unmatched detections")
        return
    df.insert(0, "id", range(1, len(df) + 1))
    print(f"{len(df)} unmatched detection(s) over {len(win)} windows "
          f"({len(win) * W_S / 3600:.1f} h)")
    print("by confidence:")
    print(pd.cut(df.conf, [0.05, 0.1, 0.2, 0.35, 1.0],
                 include_lowest=True).value_counts().sort_index().to_string())
    print(f"on windows that already have a catalog burst: {int((df.n_catalog_boxes > 0).sum())}")

    for sheet in range((len(df) + args.per_sheet - 1) // args.per_sheet):
        chunk = df.iloc[sheet * args.per_sheet:(sheet + 1) * args.per_sheet]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(15, 2.1 * len(chunk)),
                                 gridspec_kw={"width_ratios": [2, 1]})
        axes = np.atleast_2d(axes)
        for (ax_full, ax_zoom), r in zip(axes, chunk.itertuples()):
            img = imgs[r.file_name]
            h = img.shape[0]
            for ax in (ax_full, ax_zoom):
                ax.imshow(img, cmap="viridis", aspect="auto", extent=[0, W_S, h, 0])
                ax.add_patch(Rectangle((r.x0 * W_S, 0), (r.x1 - r.x0) * W_S, h,
                                       fill=False, ec="red", lw=2.0, ls="--"))
                ax.tick_params(labelsize=5)
            for t0, t1 in gt.get(r.file_name, []):
                ax_full.add_patch(Rectangle((t0 * W_S, 0), (t1 - t0) * W_S, h,
                                            fill=False, ec="lime", lw=1.8))
            ax_full.set_title(f"#{r.id}  conf={r.conf:.2f}  {r.file_name[4:-4]}", fontsize=7)
            c0 = (r.x0 + r.x1) / 2 * W_S
            ax_zoom.set_xlim(max(0, c0 - args.zoom_s), min(W_S, c0 + args.zoom_s))
            ax_zoom.set_title(f"#{r.id} zoom +/-{args.zoom_s:.0f}s", fontsize=7)
        # ASCII only: this matplotlib has no CJK glyphs
        fig.suptitle("red dashed = detector found a burst the OWN-STATION catalog does not list   "
                     "green = catalogued bursts   -> is there really a burst in the red box?  "
                     "(vertical full-height bands are RFI)", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = os.path.join(args.out_dir, f"own_fp_{sheet + 1:02d}.png")
        fig.savefig(out, dpi=105); plt.close(fig)
        print(f"  {out}")

    df["verdict"] = ""      # y = real burst the catalog missed / n = false alarm / ? = unclear
    df["note"] = ""
    p = os.path.join(args.out_dir, "own_fp_verdicts.csv")
    df.drop(columns=["x0", "x1"]).to_csv(p, index=False)
    print(f"\nfill the verdict column (y/n/?): {p}")


if __name__ == "__main__":
    main()
