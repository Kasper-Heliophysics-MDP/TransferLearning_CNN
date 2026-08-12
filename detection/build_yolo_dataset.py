"""
build_yolo_dataset.py

scrape_windows.py output (win-*.npy + windows.csv + boxes.csv) -> an
ultralytics-format YOLO dataset.

    <out>/images/{train,val}/*.png
    <out>/labels/{train,val}/*.txt
    <out>/data.yaml

Boxes use the full frequency height (cy=0.5, h=1.0) -- see TRAINING_PLAN.md
"Bounding box 构造" for why the frequency extent is deliberately not labeled,
and for the rendered check confirming bursts really do span most of the height.

Three things here are decisions, not defaults, and are worth re-reading before
trusting a number that comes out of training on this:

1. SPLIT IS BY STATION-DAY, NEVER BY WINDOW. Adjacent 15-minute windows from
   one day share RFI conditions, and a single event can straddle two of them
   (the pilot had 3 such boxes). Splitting by window would put near-duplicate
   context on both sides of the train/val line and inflate val scores.

2. VAL IS RESTRICTED TO HUMAN-REVIEWED-CLEAN WINDOWS. TRAINING_PLAN.md's data
   quality strategy is "训练集容忍噪声,验证集必须干净": a val window is kept
   only if it is a negative (no cataloged event at all -- clean by
   construction) or every one of its boxes was reviewed `usable`. Windows in a
   val day that don't meet that are DROPPED, not moved to train -- moving them
   would put the same day on both sides and defeat decision 1.

3. THE TIME AXIS GETS SQUASHED. A window is 200x3600 (1:18); at the default
   640x640 the time axis is compressed ~5.6x, so a median 35s box is ~25px
   wide. This is uniform across every window -- which is the property the whole
   15-minute-window redesign was for, since a given true drift rate then maps
   to a given pixel slope everywhere -- but whether that much time compression
   costs accuracy is open question #7 in TRAINING_PLAN.md. `--img-size 640
   1280` is the obvious next experiment.
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd

CLASSES = ["II", "III", "V"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

CONFIG_COLS = ["n_freq_channels", "freq_min_mhz", "freq_max_mhz", "sample_interval_s"]


def keep_dominant_config(windows: pd.DataFrame) -> pd.DataFrame:
    """Drop every window that isn't on the station's dominant instrument config.

    "Arecibo" is not one instrument setting. The real full scrape produced four:
    6532 windows at 200ch/15-86.6MHz/0.25s, plus 47 at 200ch/5-119.9MHz/0.25s
    and 30 at 100ch/5-85.4MHz (at two different sample intervals). Mixing them
    would undo the entire point of the fixed-window redesign -- a given true
    drift rate would land on a different pixel slope depending on which config a
    sample came from, which is exactly the contamination that made per-event
    crops unusable. Phase 1 is a single-configuration experiment by design
    (TRAINING_PLAN.md 结论4), so the minority configs are dropped, loudly.
    """
    sizes = windows.groupby(CONFIG_COLS).size().sort_values(ascending=False)
    best = sizes.index[0]
    mask = (windows[CONFIG_COLS] == pd.Series(dict(zip(CONFIG_COLS, best)))).all(axis=1)
    print(f"  config: keeping {dict(zip(CONFIG_COLS, best))} -> {int(mask.sum())} windows")
    for cfg, n in sizes.iloc[1:].items():
        print(f"          dropping {n} window(s) on {dict(zip(CONFIG_COLS, cfg))}")
    return windows[mask]


def drop_short_windows(windows: pd.DataFrame, min_frac: float) -> pd.DataFrame:
    """Drop windows materially shorter than the station's normal window.

    A truncated FITS file yields a window with fewer columns, and render_png
    resizes every window to the SAME pixel width -- so a 100-second window ends
    up at 6.4 px/s while a normal 900-second one sits at 0.71 px/s, 9x apart.
    That is exactly the pixel-scale inconsistency the whole 15-minute-window
    redesign existed to remove; it just survived in a corner of the data.

    Real scale of the problem on the full Arecibo scrape: 10 windows out of
    5983 (0.17%), 9 of them negatives. So this is a latent-correctness fix, not
    a measurable-accuracy fix -- do not expect it to move any metric.

    min_frac is a fraction of the MODAL column count (the station's nominal
    full window), not an absolute value, so it transfers to stations with a
    different sample interval. 0.99 keeps the 3586-3589-column windows (a few
    samples short of 3600, physically still 15 minutes) and drops the rest.
    """
    if min_frac <= 0:
        return windows
    modal = int(windows.n_cols.mode().iloc[0])
    keep = windows.n_cols >= modal * min_frac
    dropped = windows[~keep]
    if len(dropped):
        print(f"  short windows: dropped {len(dropped)} of {len(windows)} "
              f"(< {min_frac:.0%} of the modal {modal} columns); "
              f"{int((dropped.n_boxes > 0).sum())} of them positive")
    return windows[keep]


def render_png(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """(freq, time) uint8 window -> 8-bit image, per-row median subtracted.

    Per-row median subtraction is the standard e-Callisto quicklook: without it
    the per-channel DC offsets swamp the signal (confirmed by eye -- bursts are
    plainly visible after it and essentially invisible before). Percentile
    clipping rather than min/max because a single saturated RFI pixel otherwise
    compresses the entire real signal range into a few levels.
    """
    x = arr.astype(np.float32)
    x -= np.median(x, axis=1, keepdims=True)
    lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    img = (x * 255).astype(np.uint8)
    h, w = size_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def debias_catalog_boxes(boxes: pd.DataFrame, windows: pd.DataFrame,
                          shift_s: float, width_cap_s: float) -> pd.DataFrame:
    """Correct the measured systematic bias in raw-catalog box times.

    The first Phase 1 run exposed a train/val label-CONVENTION mismatch, not
    just noise: val boxes are 81% human-corrected, while train boxes are 100%
    raw catalog. Measured against the 63 human-corrected non-zero-width events,
    raw catalog boxes reach IoU>=0.5 only 27% of the time; shifting them +25s
    later and capping width at 90s takes that to 63% in-sample, 55% held-out
    over 200 split-half trials (5-95 pct: 41-66%). The model had faithfully
    learned the early-biased catalog convention and was then scored against the
    corrected one -- visible in the predictions as boxes sitting to the LEFT of
    the real burst.

    THIS IS NOT LOSSLESS. The 63 events it was fitted on are ones a human chose
    to correct, so they over-represent badly-offset cases, and 21% of catalog
    boxes were already right -- a global shift makes those worse. It trades a
    measurable systematic bias for better average agreement.

    Applies to TRAIN only, and only to time_source=='catalog': `manual` boxes
    are already ground truth, and `zero_width_default` boxes were fitted
    separately. Val is never touched -- it is the measuring stick.
    """
    if shift_s == 0 and width_cap_s == 0:
        return boxes
    dt = windows.set_index("file_name")["sample_interval_s"]
    ncol = windows.set_index("file_name")["n_cols"]
    m = boxes.time_source == "catalog"
    b = boxes.copy()
    step = b.loc[m, "file_name"].map(dt)
    n = b.loc[m, "file_name"].map(ncol)
    s0 = b.loc[m, "box_start_col"] + shift_s / step
    s1 = b.loc[m, "box_end_col"] + shift_s / step
    if width_cap_s > 0:
        s1 = np.minimum(s1, s0 + width_cap_s / step)
    b.loc[m, "box_start_col"] = np.clip(s0, 0, n - 1).round().astype(int)
    b.loc[m, "box_end_col"] = np.clip(s1, 1, n).round().astype(int)
    b = b[b.box_end_col > b.box_start_col]
    print(f"  debias: shifted {int(m.sum())} catalog-sourced train boxes by +{shift_s:.0f}s"
          + (f", width capped at {width_cap_s:.0f}s" if width_cap_s else "")
          + f"; {len(boxes) - len(b)} box(es) dropped for landing outside their window")
    return b


def clean_positive_windows(boxes: pd.DataFrame) -> set[str]:
    """Windows where EVERY box was human-reviewed `usable`."""
    return (set(boxes[boxes.review_status == "usable"].file_name)
            - set(boxes[boxes.review_status != "usable"].file_name))


def choose_val_days(windows: pd.DataFrame, boxes: pd.DataFrame, val_frac: float,
                     val_clean_frac: float = 1.0) -> set[str]:
    """Pick whole station-days for val, preferring days with the most
    human-reviewed-usable positive windows. Whole days only -- see decision 1
    in the module docstring.

    `val_clean_frac` caps how much of the clean-reviewed pool val is allowed to
    consume. It defaults to 1.0 (take as many clean days as val_frac wants),
    but must be lowered when the experiment needs clean windows left over for
    TRAINING -- otherwise val eats every reviewed window and a "train on
    reviewed data only" arm has nothing to train on."""
    clean_pos = clean_positive_windows(boxes)
    pos = windows[(windows.n_boxes > 0) & (~windows.excluded)]
    by_day = (pos.assign(clean=pos.file_name.isin(clean_pos))
                 .groupby("date").agg(n_pos=("file_name", "size"), n_clean=("clean", "sum")))
    by_day = by_day[by_day.n_clean > 0].sort_values(["n_clean", "n_pos"], ascending=[False, True])

    n_clean_total = int(by_day.n_clean.sum())
    target = min(max(1, int(round(len(pos) * val_frac))),
                 max(1, int(round(n_clean_total * val_clean_frac))))
    val_days, got = set(), 0
    for day, r in by_day.iterrows():
        if got >= target:
            break
        val_days.add(day)
        got += r.n_clean
    print(f"  val: {len(val_days)} station-days, {got} clean positive windows "
          f"(target ~{target}; {n_clean_total} clean windows exist in total, "
          f"{len(pos)} positive windows overall)")
    return val_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--img-size", nargs=2, type=int, default=[640, 640], metavar=("H", "W"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-clean-frac", type=float, default=1.0,
                    help="max share of the human-reviewed-clean window pool that val may "
                         "consume (see choose_val_days). Lower it to leave clean windows "
                         "available for training.")
    ap.add_argument("--train-source", default="both", choices=["both", "noisy", "clean"],
                    help="which POSITIVE windows may enter train: 'clean' = only windows whose "
                         "every box was reviewed usable; 'noisy' = only the rest; 'both' = union. "
                         "Negatives are unaffected -- they need no review, a window with no "
                         "cataloged event is clean by construction.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debias-shift-s", type=float, default=0.0,
                    help="shift TRAIN boxes whose time_source=='catalog' later by this many "
                         "seconds (0 = off). See debias_catalog_boxes().")
    ap.add_argument("--debias-width-cap-s", type=float, default=0.0,
                    help="cap the width of those same boxes, in seconds (0 = no cap)")
    ap.add_argument("--min-window-frac", type=float, default=0.0,
                    help="drop windows shorter than this fraction of the modal window length "
                         "(0 = off). See drop_short_windows(). Left off by default so datasets "
                         "built before it stay reproducible.")
    ap.add_argument("--keep-all-configs", action="store_true",
                    help="don't restrict to the dominant instrument config (Phase 2 only -- "
                         "mixing configs reintroduces the pixel-slope inconsistency)")
    args = ap.parse_args()

    windows = pd.read_csv(os.path.join(args.window_dir, "windows.csv"), dtype={"date": str})
    boxes = pd.read_csv(os.path.join(args.window_dir, "boxes.csv"), dtype={"date": str})
    boxes["review_status"] = boxes["review_status"].fillna("")

    if args.overwrite and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "labels", split), exist_ok=True)

    print(f"{len(windows)} windows, {int(windows.excluded.sum())} excluded, {len(boxes)} boxes")
    if not args.keep_all_configs:
        windows = keep_dominant_config(windows)
    windows = drop_short_windows(windows, args.min_window_frac)
    boxes = boxes[boxes.file_name.isin(set(windows.file_name))]
    val_days = choose_val_days(windows, boxes, args.val_frac, args.val_clean_frac)
    clean_pos = clean_positive_windows(boxes)

    # de-bias AFTER the split is fixed, and on the train side only -- val is the
    # measuring stick and must keep its human-corrected times untouched
    if args.debias_shift_s or args.debias_width_cap_s:
        is_val = boxes.date.isin(val_days)
        boxes = pd.concat([
            boxes[is_val],
            debias_catalog_boxes(boxes[~is_val], windows,
                                 args.debias_shift_s, args.debias_width_cap_s),
        ], ignore_index=True)

    dirty = set(boxes[boxes.review_status != "usable"].file_name)
    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}   # windows, positives, boxes
    dropped_dirty_val = 0

    skipped_by_source = 0
    for w in windows.itertuples():
        if w.excluded:
            continue
        split = "val" if w.date in val_days else "train"
        wb = boxes[boxes.file_name == w.file_name]
        if split == "train" and w.n_boxes > 0:
            is_clean = w.file_name in clean_pos
            if (args.train_source == "clean" and not is_clean) or \
               (args.train_source == "noisy" and is_clean):
                skipped_by_source += 1
                continue
        if split == "val" and w.file_name in dirty:
            # unreviewed / non-usable boxes can't go in val, and can't be moved
            # to train either (that would split one day across both sides)
            dropped_dirty_val += 1
            continue

        arr = np.load(os.path.join(args.window_dir, w.file_name))
        stem = w.file_name[:-len(".npy")]
        cv2.imwrite(os.path.join(args.out_dir, "images", split, stem + ".png"),
                    render_png(arr, tuple(args.img_size)))

        lines = []
        for b in wb.itertuples():
            if str(b.type) not in CLASS_ID:
                print(f"  [yolo] unexpected type {b.type!r} in {w.file_name}, skipping box")
                continue
            cx = (b.box_start_col + b.box_end_col) / 2 / w.n_cols
            bw = (b.box_end_col - b.box_start_col) / w.n_cols
            # full frequency height: centre 0.5, height 1.0 (deliberate, see docstring)
            lines.append(f"{CLASS_ID[str(b.type)]} {cx:.6f} 0.500000 {bw:.6f} 1.000000")
        with open(os.path.join(args.out_dir, "labels", split, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))

        counts[split][0] += 1
        counts[split][1] += 1 if lines else 0
        counts[split][2] += len(lines)

    yaml = os.path.join(args.out_dir, "data.yaml")
    with open(yaml, "w") as fh:
        fh.write(f"path: {os.path.abspath(args.out_dir)}\n"
                 "train: images/train\nval: images/val\n\nnames:\n")
        for i, c in enumerate(CLASSES):
            fh.write(f"  {i}: {c}\n")

    for split in ("train", "val"):
        n, npos, nbox = counts[split]
        print(f"  {split}: {n} images ({npos} with boxes, {n - npos} negatives), {nbox} boxes")
    if skipped_by_source:
        print(f"  --train-source={args.train_source}: excluded {skipped_by_source} positive "
              f"train window(s) of the other kind")
    if dropped_dirty_val:
        print(f"  dropped {dropped_dirty_val} window(s) from val days for having "
              f"unreviewed/non-usable boxes (kept out of train to preserve the day split)")
    print(f"wrote {yaml}")


if __name__ == "__main__":
    main()
