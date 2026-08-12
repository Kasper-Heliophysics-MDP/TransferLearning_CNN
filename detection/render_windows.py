"""
render_windows.py

Eyeball check for extract_windows.py's output: draws each 15-minute window with
its boxes, so the boxes can be confirmed to land on real bursts BEFORE
committing to the ~9-hour full Arecibo run.

Also prints the check that motivated this whole path change -- the normalized
horizontal position of every box. extract_events.py's crops had mean 0.5000 /
std 0.0000 (total label leakage, a detector could score well by always
predicting a centre box); this should now be spread out.

Usage:
  python detection/render_windows.py <window_dir> <out_dir> [--n 12] [--only-with-boxes]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SOURCE_COLOR = {"manual": "lime", "catalog": "red", "zero_width_default": "orange"}


def leakage_report(boxes: pd.DataFrame, windows: pd.DataFrame) -> None:
    if boxes.empty:
        print("no boxes to report on")
        return
    n_cols = windows.set_index("file_name")["n_cols"]
    centre = ((boxes.box_start_col + boxes.box_end_col) / 2 /
              boxes.file_name.map(n_cols)).to_numpy()
    print(f"normalized box CENTRE within window (n={len(centre)}):")
    print(f"  mean {centre.mean():.4f}   std {centre.std():.4f}   "
          f"min {centre.min():.3f}   max {centre.max():.3f}")
    print(f"  fraction in 0.49-0.51: {((centre > 0.49) & (centre < 0.51)).mean() * 100:.0f}%"
          "   <- was 100% with per-event crops")
    w = (boxes.box_end_col - boxes.box_start_col).to_numpy()
    print(f"box width in columns: median {np.median(w):.0f}  min {w.min()}  max {w.max()}")
    print(f"time_source: {dict(boxes.time_source.value_counts())}")


def draw(ax, win_dir: str, wrow, boxes: pd.DataFrame) -> None:
    arr = np.load(os.path.join(win_dir, wrow.file_name)).astype(np.float32)
    n_freq, n_t = arr.shape
    dt = float(wrow.sample_interval_s)

    # per-row median subtraction: the standard eCallisto quicklook. Without it
    # the per-channel DC offsets swamp everything (verified by eye against the
    # cleaned/denoised version of the same data)
    disp = arr - np.median(arr, axis=1, keepdims=True)
    vmax = np.percentile(np.abs(disp), 99.0) or 1.0
    ax.imshow(disp, aspect="auto", origin="upper", cmap="viridis",
              vmin=-vmax * 0.3, vmax=vmax, extent=[0, n_t * dt / 60, n_freq, 0])

    for b in boxes.itertuples():
        x0, x1 = b.box_start_col * dt / 60, b.box_end_col * dt / 60
        ec = SOURCE_COLOR.get(b.time_source, "white")
        ax.add_patch(Rectangle((x0, 0), max(x1 - x0, 0.05), n_freq,
                               fill=False, ec=ec, lw=1.8, zorder=5))
        ax.text(x0, n_freq * 0.06, f" {b.type}", color=ec, fontsize=7, weight="bold", zorder=6)

    freq_path = os.path.join(win_dir, "meta", f"freq-{wrow.location}-{wrow.date}.npy")
    if os.path.exists(freq_path):
        freq = np.load(freq_path)
        ticks = np.linspace(0, n_freq - 1, 5).astype(int)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{freq[i]:.0f}" for i in ticks], fontsize=6)
    ax.set_xlabel("minutes into window", fontsize=6)
    ax.tick_params(labelsize=6)

    flag = f"  EXCLUDED({wrow.exclude_reason})" if wrow.excluded else ""
    ax.set_title(f"{wrow.date} {wrow.win_start_time}  {len(boxes)} box{flag}", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--only-with-boxes", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    windows = pd.read_csv(os.path.join(args.window_dir, "windows.csv"), dtype={"date": str})
    boxes_path = os.path.join(args.window_dir, "boxes.csv")
    boxes = (pd.read_csv(boxes_path, dtype={"date": str})
             if os.path.exists(boxes_path) else pd.DataFrame(columns=["file_name"]))

    print(f"{len(windows)} windows, {len(boxes)} boxes, "
          f"{int(windows.excluded.sum())} excluded, "
          f"{int((windows.n_boxes == 0).sum())} empty (negatives)")
    leakage_report(boxes, windows)

    sel = windows[windows.n_boxes > 0] if args.only_with_boxes else windows
    sel = sel.sample(min(args.n, len(sel)), random_state=args.seed).sort_values("file_name")

    os.makedirs(args.out_dir, exist_ok=True)
    ncol, nrow = 2, int(np.ceil(len(sel) / 2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, wrow in zip(axes, sel.itertuples()):
        draw(ax, args.window_dir, wrow, boxes[boxes.file_name == wrow.file_name])
    for ax in axes[len(sel):]:
        ax.axis("off")
    fig.suptitle("15-min windows  |  box colour = time source: "
                 "lime=manual  red=catalog  orange=zero-width default", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(args.out_dir, "windows.png")
    fig.savefig(out, dpi=105)
    print("wrote", out)


if __name__ == "__main__":
    main()
