"""
make_box_subset.py

Build an event_review subset from CATALOG boxes in a window directory, so a
station's boxes can be reviewed and time-corrected before they are allowed into
training.

Sibling of make_gap_subset.py, which does the same job for model-found bursts
the catalog never recorded. The difference is where the prefilled time comes
from and therefore what the reviewer is correcting:

    make_gap_subset.py   prefill = the detector's box.  Reviewer confirms a
                         burst exists and draws its extent.
    make_box_subset.py   prefill = the catalog's box.   Reviewer corrects a
                         time known to be systematically early (measured
                         median +40s on Arecibo, and the same defect showed up
                         on every e-Callisto station reviewed so far).

WHY THIS EXISTS AT ALL
----------------------
Australia-ASSA was scraped to rescue Type V: Arecibo has 27 clean samples
total, which cannot feed training and evaluation at once (a 20-box val leaves
7 to train on, and the measured result was AP 0.048 / 15% detection). ASSA adds
45 V and 110 II. But its boxes are unreviewed catalog times, and the most
solid finding in PHASE1_RESULTS is that unreviewed boxes are a NEGATIVE
contribution at this scale -- 205 hand-corrected boxes beat 1500 mixed ones.
Adding ASSA to training without reviewing it first would repeat exactly the
mistake that result documents.

ONE ENTRY PER BOX, NOT PER WINDOW
---------------------------------
manual_burst_range_json holds a single {start_s, end_s}, and 7 ASSA windows
carry two II/V boxes each. One entry per window would silently drop the second.

CLIPPED BOXES: 87 OF 155 ON ASSA (56%)
--------------------------------------
Type II is long -- median 240s here, up to the full 900s window -- so more than
half of these events run past the window edge and the box is truncated at it.
Two consequences the reviewer needs to know:

  * The visible burst genuinely continues outside the image. Dragging to the
    edge is the correct annotation; there is nothing beyond it to include.
  * The truncation is not an error to fix, it is a property of a 15-minute
    canvas. If long Type II turns out to need more room, the fix is a longer
    canvas (the scrape already cached +/-2 slots, so no re-download), not a
    different annotation.

The prefill uses box_start_col/box_end_col, i.e. the already-window-clipped
extent, so it is always drawable and always matches what training would use.

Type is taken from the catalog and is real, so event_review's existing II/V
handling applies unchanged -- notably MINIMAL_VIEW_TYPES, which matters here:
clean() absorbs about 97% of a long Type II's excess in the background-fit
stage, before any tunable parameter is reached.

Usage:
    python event_review/make_box_subset.py data/ecallisto/windows_assa \\
        --types II V --out-dir data/ecallisto/assa_iiv          # dry run
    python event_review/make_box_subset.py ... --apply
    # then: streamlit run event_review/app.py
    #       sidebar -> eCallisto -> directory = data/ecallisto/assa_iiv
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detection"))
from build_yolo_dataset import keep_dominant_config, drop_short_windows   # noqa: E402

META_COLS = ["file_name", "date", "location", "start_time", "end_time", "type",
             "event_start_time", "event_end_time", "other_stations", "uncertain",
             "freq_min_mhz", "freq_max_mhz", "n_freq_channels", "sample_interval_s"]


def shift(hms: str, seconds: float) -> str:
    """HH:MM:SS(.f) + seconds. Sub-second precision is kept: event_burst_indices
    picks its parse format on whether a '.' is present, and rounding the prefill
    to whole seconds moves the box by up to 4 columns at 0.25s/sample -- a bias
    that goes straight into the labels, because the prefill is what a reviewer
    accepts untouched a large fraction of the time."""
    t = datetime.strptime(str(hms), "%H:%M:%S") + timedelta(seconds=float(seconds))
    return t.strftime("%H:%M:%S.%f")[:-5] if t.microsecond else t.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window_dir")
    ap.add_argument("--types", nargs="+", default=["II", "V"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    win = pd.read_csv(os.path.join(args.window_dir, "windows.csv"), dtype={"date": str})
    box = pd.read_csv(os.path.join(args.window_dir, "boxes.csv"), dtype={"date": str})

    # same admission rules the trainer uses, so nothing is reviewed that could
    # never reach training anyway
    win = win[~win.excluded.astype(bool)]
    win = drop_short_windows(keep_dominant_config(win), 0.99)
    box = box[box.file_name.isin(set(win.file_name)) & box.type.isin(args.types)]
    win = win.set_index("file_name")

    rows, links = [], []
    for i, b in enumerate(box.sort_values(["date", "file_name", "box_start_col"]).itertuples(), 1):
        w = win.loc[b.file_name]
        dt = float(w.sample_interval_s)
        start_s, end_s = b.box_start_col * dt, b.box_end_col * dt
        name = f"{b.type.lower()}{i:04d}-{b.file_name[:-len('.npy')]}.npy"
        links.append((name, os.path.abspath(os.path.join(args.window_dir, b.file_name))))
        rows.append({
            "file_name": name, "date": w.date, "location": w.location,
            "start_time": w.win_start_time, "end_time": w.win_end_time, "type": b.type,
            "event_start_time": shift(w.win_start_time, start_s),
            "event_end_time": shift(w.win_start_time, end_s),
            "other_stations": "", "uncertain": bool(b.uncertain),
            "freq_min_mhz": w.freq_min_mhz, "freq_max_mhz": w.freq_max_mhz,
            "n_freq_channels": w.n_freq_channels, "sample_interval_s": w.sample_interval_s,
        })

    meta = pd.DataFrame(rows)[META_COLS]
    if meta.file_name.duplicated().any():
        raise SystemExit("duplicate file_name generated")

    n_clip = int(box.clipped.sum()) if "clipped" in box else 0
    widths = (box.box_end_col - box.box_start_col) * 0.25
    print(f"{len(meta)} box(es) over {box.file_name.nunique()} window(s)")
    print(f"  types: {box.type.value_counts().to_dict()}")
    print(f"  clipped at a window edge: {n_clip} ({n_clip / max(len(box),1):.0%}) "
          f"-- the burst continues outside the image; drag to the edge")
    print(f"  width s: median {widths.median():.0f}, max {widths.max():.0f}")
    print(f"  windows holding more than one: "
          f"{int((box.file_name.value_counts() > 1).sum())}")

    freq_needed = sorted({(r["location"], r["date"]) for r in rows})
    if not args.apply:
        print(f"\nwould write {args.out_dir}/: {len(links)} symlink(s), "
              f"{len(freq_needed)} frequency axis file(s), metadata.csv")
        print("DRY RUN -- nothing written. Re-run with --apply.")
        return

    os.makedirs(os.path.join(args.out_dir, "meta"), exist_ok=True)
    for name, target in links:
        link = os.path.join(args.out_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(target, link)

    n_freq = 0
    for loc, date in freq_needed:
        src = os.path.abspath(os.path.join(args.window_dir, "meta", f"freq-{loc}-{date}.npy"))
        if not os.path.exists(src):
            print(f"  !! missing {os.path.basename(src)} -- that day falls back to channel index")
            continue
        link = os.path.join(args.out_dir, "meta", os.path.basename(src))
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(src, link)
        n_freq += 1

    meta.to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    print(f"\nwrote {args.out_dir}/: {len(links)} symlink(s), {n_freq} frequency axis file(s), "
          f"metadata.csv")
    print(f"\nreview with:  streamlit run event_review/app.py")
    print(f"  sidebar -> eCallisto -> directory = {args.out_dir}")


if __name__ == "__main__":
    main()
