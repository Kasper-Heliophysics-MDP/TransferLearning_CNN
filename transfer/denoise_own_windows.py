"""
denoise_own_windows.py

Blind-denoise the own-station windows and measure, before anything else, how
much of each known burst survives.

WHY BLIND
---------
The production own-station path passes the catalog burst times into clean() as
`known_burst_weight`, so the denoiser knows where not to suppress. Doing that
here would be leakage: these windows are the TEST set for a detector, and
handing the preprocessing the answer key inflates whatever comes out. So
known_burst_weight is None and the denoiser has to find the RFI on its own.

WHY THE RETENTION CHECK COMES FIRST
-----------------------------------
Blind suppression can delete the signal. This project has already been bitten
twice: extract_events_own_station.py once ran the whole file through clean()
with no ground-truth protection and erased real bursts (found by eye during
review, not by any metric), and cleaned_events/ turned out to have zero
protection over 42% of the hand-corrected events. The documented blind
retention rate ranges from 20% to 74% depending on station, so "it probably
works" is not available here.

Retention is measured per known box as the fraction of the burst's excess
signal (z-score above the per-channel background) still present after cleaning.
A before/after sheet is written for the same boxes, because both of the failures
above were caught by eye and missed by numbers.

Usage:
    python transfer/denoise_own_windows.py            # denoise + measure + render
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ecallisto_grabber", "denoising"))
sys.path.insert(0, os.path.join(ROOT, "event_review"))
from sumthreshold_denoise import clean          # noqa: E402
import data_access as da                        # noqa: E402


def zscore(a: np.ndarray) -> np.ndarray:
    """Per-channel excess in robust sigmas. NaN-aware by construction: median
    and MAD, never mean/std -- a single bad sample otherwise contaminates a
    whole channel rather than one pixel."""
    x = a.astype(np.float32)
    x = x - np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x), axis=1, keepdims=True) * 1.4826 + 1e-6
    return x / mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-dir", default="data/burst_data/own_windows")
    ap.add_argument("--out-dir", default="data/burst_data/own_windows_clean")
    ap.add_argument("--sheet-dir", default="transfer/denoise_check")
    ap.add_argument("--n-sheets", type=int, default=12, help="boxes to render before/after")
    args = ap.parse_args()

    win = pd.read_csv(os.path.join(args.window_dir, "windows.csv")).set_index("file_name")
    box = pd.read_csv(os.path.join(args.window_dir, "boxes.csv"))
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.sheet_dir, exist_ok=True)

    params = da.default_params(da.PRODUCTION_METHOD[da.SOURCE_OWN_STATION])
    rows, cache = [], {}
    for f in win.index:
        raw = np.load(os.path.join(args.window_dir, f))
        p = dict(params)
        p["time_win"] = da.time_win_for(float(win.loc[f].sample_interval_s))
        cleaned = clean(raw, known_burst_weight=None, return_intermediates=False, **p)["cleaned"]
        np.save(os.path.join(args.out_dir, f), cleaned.astype(np.float32))
        src = os.path.join(args.window_dir, "freq-" + f[:-4] + ".npy")
        dst = os.path.join(args.out_dir, "freq-" + f[:-4] + ".npy")
        if not os.path.exists(dst):
            np.save(dst, np.load(src))
        cache[f] = (raw, cleaned)

        zr, zc = zscore(raw), zscore(cleaned)
        for b in box[box.file_name == f].itertuples():
            c0, c1 = int(b.box_start_col), int(b.box_end_col)
            # only count pixels that carried real excess before cleaning;
            # retention over background pixels is meaningless
            m = zr[:, c0:c1] > 3.0
            if m.sum() < 10:
                keep = np.nan
            else:
                keep = float(np.clip(zc[:, c0:c1][m] / zr[:, c0:c1][m], 0, 1).mean())
            rows.append(dict(file_name=f, type=b.type, c0=c0, c1=c1,
                             n_hot=int(m.sum()), retention=keep))

    r = pd.DataFrame(rows)
    win[["n_cols"]].to_csv(os.path.join(args.out_dir, "_win_shape.csv"))
    for src in ("windows.csv", "boxes.csv"):
        pd.read_csv(os.path.join(args.window_dir, src)).to_csv(
            os.path.join(args.out_dir, src), index=False)

    v = r.retention.dropna()
    print(f"blind-denoised {len(win)} windows, {len(r)} known boxes\n")
    print("burst signal retained inside the known boxes:")
    print(f"  median {v.median():.2f}   quartiles {v.quantile(.25):.2f}-{v.quantile(.75):.2f}"
          f"   min {v.min():.2f}")
    for thr, lbl in [(0.2, "<0.20 (mostly erased)"), (0.5, "<0.50 (half gone)")]:
        print(f"  {lbl:24s} {int((v < thr).sum())} / {len(v)}")
    print("\nby type:")
    print(r.groupby("type").retention.agg(["count", "median", "min"]).round(2).to_string())
    r.to_csv(os.path.join(args.sheet_dir, "retention.csv"), index=False)

    # eyeball sheet: the worst-retained boxes, where erasure would show up
    worst = r.dropna(subset=["retention"]).nsmallest(args.n_sheets, "retention")
    n = len(worst)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.0 * n))
    axes = np.atleast_2d(axes)
    for (a_raw, a_cln), b in zip(axes, worst.itertuples()):
        raw, cleaned = cache[b.file_name]
        nc = raw.shape[1]
        for ax, img, lbl in ((a_raw, zscore(raw), "raw"), (a_cln, zscore(cleaned), "blind-cleaned")):
            ax.imshow(np.clip(img, -2, 8), aspect="auto", cmap="viridis",
                      extent=[0, nc, img.shape[0], 0])
            ax.add_patch(Rectangle((b.c0, 0), b.c1 - b.c0, img.shape[0],
                                   fill=False, ec="red", lw=1.8))
            ax.set_title(f"{lbl}  type {b.type}  retention {b.retention:.2f}", fontsize=7)
            ax.tick_params(labelsize=5)
    fig.suptitle("Worst-retained known bursts, blind denoising. "
                 "Red box = catalogued burst. If it is visible on the left and gone "
                 "on the right, blind denoising is erasing signal.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.sheet_dir, "worst_retention.png")
    fig.savefig(out, dpi=105); plt.close(fig)
    print(f"\nwrote {args.out_dir}/ and {out}")


if __name__ == "__main__":
    main()
