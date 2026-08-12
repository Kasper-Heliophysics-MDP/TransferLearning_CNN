"""
render_boxes.py -- eyeball check for TRAINING_PLAN.md open question #3:

  "用人工修正后的时间 + 满高度画框, 叠加在真实图上到底合不合理 -- 只推理过, 没有实际渲染看过"

Draws, on the REAL production spectrogram of each event:
  - red dashed box  = catalog time (metadata.csv event_start_time/event_end_time), full freq height
  - lime solid box  = event_review manual correction (manual_burst_range_json), full freq height
  - grey shaded band = region actually protected by known_burst_weight during
    production cleaning (catalog time +/- known_burst_margin=10 columns)

The grey band matters: production `cleaned_events/` was denoised using the
CATALOG time, so if the lime box sits outside the grey band the burst was
cleaned WITHOUT ground-truth protection.

Usage:
  python render_boxes.py <outdir> [--n 12] [--pick usable_corrected|usable_plain|typeII|typeV|discard]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = "/home/ubuntu/Desktop/TransferLearning_CNN"
RAW_DIR = os.path.join(REPO, "data/ecallisto/raw_events")
CLEAN_DIR = os.path.join(REPO, "data/ecallisto/cleaned_events")
KNOWN_BURST_MARGIN = 10  # matches clean_ecallisto_events.clean_event_file default

sys.path.insert(0, os.path.join(REPO, "ecallisto_grabber", "denoising"))
from clean_ecallisto_events import event_burst_indices  # noqa: E402


def load_table() -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(RAW_DIR, "metadata.csv"), dtype=str, keep_default_na=False)
    rev = pd.read_csv(os.path.join(RAW_DIR, "review_status.csv"), dtype=str, keep_default_na=False)
    df = meta.merge(rev, on="file_name", how="left")
    for c in ("status", "manual_burst_range_json", "override_params_json", "problem_tags", "notes"):
        df[c] = df[c].fillna("")
    return df


def crop_seconds(row) -> float:
    f = "%H:%M:%S"
    s, e = datetime.strptime(row["start_time"], f), datetime.strptime(row["end_time"], f)
    if e < s:
        e += timedelta(days=1)
    return (e - s).total_seconds()


def manual_indices(row, dt: float):
    """manual_burst_range_json stores seconds FROM CROP START (app.py's
    number inputs are offsets into the displayed crop)."""
    js = row["manual_burst_range_json"]
    if not js:
        return None
    d = json.loads(js)
    return round(float(d["start_s"]) / dt), round(float(d["end_s"]) / dt)


def draw_panel(ax, row, show_raw=False):
    fn = row["file_name"]
    dt = float(row["sample_interval_s"])
    arr = np.load(os.path.join(RAW_DIR if show_raw else CLEAN_DIR, fn))
    arr = arr.astype(np.float32)
    n_freq, n_t = arr.shape

    freq = np.load(os.path.join(RAW_DIR, "meta", f"freq-{row['location']}-{row['date']}.npy"))

    if show_raw:
        # raw is uint8-encoded: per-row median subtraction is the standard
        # eCallisto quicklook, otherwise channel offsets swamp the signal
        disp = arr - np.median(arr, axis=1, keepdims=True)
        vmax = np.percentile(np.abs(disp), 99.0) or 1.0
        ax.imshow(disp, aspect="auto", origin="upper", cmap="viridis",
                  vmin=-vmax * 0.3, vmax=vmax, extent=[0, n_t * dt, n_freq, 0])
    else:
        vmax = np.percentile(np.abs(arr), 99.5) or 1.0
        ax.imshow(arr, aspect="auto", origin="upper", cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, extent=[0, n_t * dt, n_freq, 0])

    cat = event_burst_indices(row, dt)
    man = manual_indices(row, dt)

    # grey = what production cleaning actually protected
    if cat is not None:
        lo = max(0, cat[0] - KNOWN_BURST_MARGIN)
        hi = min(n_t, cat[1] + KNOWN_BURST_MARGIN)
        ax.axvspan(lo * dt, hi * dt, color="0.5", alpha=0.18, zorder=1)

    # red dashed = catalog box, full frequency height
    if cat is not None:
        x0, x1 = cat[0] * dt, cat[1] * dt
        ax.add_patch(Rectangle((x0, 0), max(x1 - x0, dt), n_freq, fill=False,
                               ec="red", lw=1.6, ls="--", zorder=5))

    # lime solid = manually corrected box, full frequency height
    if man is not None:
        x0, x1 = man[0] * dt, man[1] * dt
        ax.add_patch(Rectangle((x0, 0), max(x1 - x0, dt), n_freq, fill=False,
                               ec="lime", lw=2.0, zorder=6))

    # frequency ticks from the real axis
    ticks = np.linspace(0, n_freq - 1, 5).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{freq[i]:.0f}" for i in ticks], fontsize=6)
    ax.tick_params(labelsize=6)

    tag = row["status"] or "unreviewed"
    corr = ""
    if cat is not None and man is not None:
        corr = f"  shift {(man[0]-cat[0])*dt:+.0f}s  width {(cat[1]-cat[0])*dt:.0f}->{(man[1]-man[0])*dt:.0f}s"
    ax.set_title(f"{fn[len('burst-Arecibo-Observatory-'):-4]}\n{tag}{corr}", fontsize=6.5)


PICKS = {
    "usable_corrected": lambda d: d[(d.status == "usable") & (d.manual_burst_range_json != "")],
    "usable_plain": lambda d: d[(d.status == "usable") & (d.manual_burst_range_json == "")],
    "typeII": lambda d: d[(d.type == "II") & (d.status != "")],
    "typeV": lambda d: d[(d.type == "V") & (d.status != "")],
    "discard": lambda d: d[d.status == "discard"],
    "negative": lambda d: d[d.type == "0"],
    "unreviewed": lambda d: d[(d.status == "") & (d.type == "III")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--pick", default="usable_corrected", choices=list(PICKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true", help="render raw instead of cleaned")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_table()
    cfg = df[(df.location == "Arecibo-Observatory") &
             (df.sample_interval_s == "0.25") & (df.n_freq_channels == "200")]
    sel = PICKS[args.pick](cfg)
    print(f"{args.pick}: {len(sel)} candidates")
    if len(sel) == 0:
        return
    sel = sel.sample(min(args.n, len(sel)), random_state=args.seed)

    ncol, nrow = 3, int(np.ceil(len(sel) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, sel.iterrows()):
        try:
            draw_panel(ax, row, show_raw=args.raw)
        except Exception as e:  # keep the sheet renderable if one event fails
            ax.text(0.5, 0.5, f"FAIL\n{e}", ha="center", va="center", fontsize=6)
            print("FAIL", row["file_name"], e)
    for ax in axes[len(sel):]:
        ax.axis("off")

    kind = "raw" if args.raw else "cleaned"
    fig.suptitle(f"Arecibo 0.25s/200ch -- {args.pick} ({kind})   "
                 f"red dashed=catalog  lime=manual  grey=protected during cleaning", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.outdir, f"{args.pick}_{kind}.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
