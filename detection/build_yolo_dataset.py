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

    WHY THE FREQUENCY BOUNDS ARE COMPARED WITH A TOLERANCE
    ------------------------------------------------------
    Exact tuple matching splits configs that are one instrument setting written
    down with different rounding. Australia-ASSA reports 14.995-86.933 MHz on
    1536 events and 15.000-86.938 on 1299 -- a 5 kHz difference. Exact matching
    treated those as separate configs and would have kept only the larger,
    silently discarding 1299 windows.

    It gets worse as soon as two stations are combined for Phase 2: Arecibo is
    15.000-86.625 and ASSA is 14.995-86.933, so exact matching picks whichever
    station has more windows and drops THE OTHER STATION ENTIRELY -- the exact
    opposite of a cross-station experiment, and it would not raise an error.

    The tolerance is relative and deliberately tight (1%): it merges settings
    differing by rounding (86.625 vs 86.933 is 0.36%) while keeping genuinely
    different instruments apart (Arecibo's minority 5-119.9MHz config is 200%
    away on freq_min; ASSA's 400ch/108-524MHz band is nowhere near). Channel
    count and sample interval must still match EXACTLY -- they set the pixel
    grid directly, and 200 vs 400 channels is not a rounding difference.

    Arecibo-only behaviour is unchanged; this was verified by rebuilding
    yolo7_single afterwards and diffing images and labels byte for byte.
    """
    def _clusters(cfgs: list[tuple], rel_tol: float = 0.01) -> list[list[tuple]]:
        """Greedy: largest config first, absorbing any config within rel_tol.
        Greedy rather than fixed buckets because buckets split at their own
        boundaries -- 14.995 and 15.000 can fall either side of a rounding
        edge, which is the very failure being fixed here."""
        out: list[list[tuple]] = []
        for cfg in cfgs:                       # pre-sorted by window count, desc
            nch, fmin, fmax, dt = cfg
            for grp in out:
                rch, rmin, rmax, rdt = grp[0]
                if nch == rch and dt == rdt \
                   and abs(fmin - rmin) <= rel_tol * max(abs(rmin), 1e-9) \
                   and abs(fmax - rmax) <= rel_tol * max(abs(rmax), 1e-9):
                    grp.append(cfg)
                    break
            else:
                out.append([cfg])
        return out

    sizes = windows.groupby(CONFIG_COLS).size().sort_values(ascending=False)
    groups = _clusters(list(sizes.index))
    counts = [sum(int(sizes[c]) for c in g) for g in groups]
    best = groups[int(np.argmax(counts))]

    mask = pd.Series(False, index=windows.index)
    for cfg in best:
        mask |= (windows[CONFIG_COLS] == pd.Series(dict(zip(CONFIG_COLS, cfg)))).all(axis=1)
    print(f"  config: keeping {dict(zip(CONFIG_COLS, best[0]))} -> {int(mask.sum())} windows")
    for cfg in best[1:]:
        print(f"          + merged {int(sizes[cfg])} window(s) on "
              f"{dict(zip(CONFIG_COLS, cfg))} (within 1%: same setting, different rounding)")
    for grp, n in zip(groups, counts):
        if grp is not best:
            print(f"          dropping {n} window(s) on {dict(zip(CONFIG_COLS, grp[0]))}")
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


def drop_contaminated_negatives(windows: pd.DataFrame, path: str,
                                protect_days: set[str] | None = None) -> pd.DataFrame:
    """Drop "negative" windows that a detector flagged as containing a burst.

    Negatives are windows with no cataloged event at all, used as empty-label
    training images. But the Monstein catalog demonstrably omits real bursts:
    of 34 model detections that matched no catalog entry, 32 (94%) were
    confirmed real by eye, and of 16 flagged NEGATIVE windows, 14 (88%) were.
    An omitted burst in a negative window is worse than label noise -- it is
    active wrong supervision, teaching the model "no burst here" while pointing
    at one.

    TWO LIMITS, both real:

    1. Circular selection. The list is produced by a model that was TRAINED on
       these very train negatives, so it has learned to stay quiet on them:
       it flags 4.2% of the unseen VAL negatives but only 1.9% of train. The
       val rate is the unbiased one, so train contamination is likely ~2x what
       this catches -- and the windows it misses are exactly the ones whose
       wrong supervision took hold hardest. A rigorous version would scan each
       fold with a model trained on the others.
    2. ~12% of what it drops is fine. Human check put the flag at 88% precise,
       and confidence does not separate cleanly (a 0.106 detection was real, a
       0.167 one was not), so no threshold fixes this.

    Dropping is still the right call -- these are wrong labels, not hard
    negatives -- but the numbers above belong in any result that uses it.

    `protect_days` exempts whole station-days (i.e. val) from the drop. Removing
    these windows from VAL is a different operation with the opposite sign: they
    are precisely the windows the detector fires on, so deleting them removes
    false positives and lifts AP for reasons that have nothing to do with the
    training change being measured. Val has to stay a fixed ruler across
    experiments, so a run that wants to measure the effect of cleaning TRAIN
    must leave val alone -- otherwise two variables move at once.
    """
    if not path or not os.path.exists(path):
        return windows
    bad = set(pd.read_csv(path)["file_name"])
    hit = windows.file_name.isin(bad)
    if protect_days:
        hit &= ~windows.date.isin(protect_days)
    if hit.any():
        print(f"  contaminated negatives: dropped {int(hit.sum())} window(s) listed in "
              f"{os.path.basename(path)}"
              + (f" (val days exempt)" if protect_days else ""))
    return windows[~hit]


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


def choose_val_days(windows: pd.DataFrame, boxes: pd.DataFrame,
                    per_class_target: dict[str, int]) -> set[str]:
    """Pick whole station-days for val, aiming at a per-class BOX budget.

    Two problems with the earlier "take the days with the most clean windows
    until val holds `val_clean_frac` of the clean pool" rule:

    1. It scaled with the pool, so val kept eating ~45% of every newly reviewed
       event forever. Hand-corrected labels are the scarce resource and the
       measured marginal value is lopsided -- val noise falls as 1/sqrt(n) (218
       -> 318 boxes buys a 17% noise reduction) while train is still on the
       steep part of its curve (243 -> 343 boxes is +41% training data, and
       label quantity demonstrably moves the result at this scale). At a 1600-
       window pool the proportional rule would have parked ~520 hand-corrected
       boxes in val that training never sees.
    2. It sorted by total clean windows, which are ~90% Type III, so the rare
       classes were left to land in val by luck. They didn't: with Type II and
       V both 100% reviewed (all 53 and 27 that Arecibo has ever recorded), val
       still held only 12 II and 7 V boxes -- too few for any per-class number
       to mean anything.

    So: fill the rare-class budgets first from the days that carry them, then
    top up Type III, then stop. Everything not needed for the budget goes to
    train. Whole days only, still -- see decision 1 in the module docstring.
    """
    clean_pos = clean_positive_windows(boxes)
    usable = windows[(windows.n_boxes > 0) & (~windows.excluded) &
                     (windows.file_name.isin(clean_pos))]
    cb = boxes[boxes.file_name.isin(set(usable.file_name))]
    # per station-day: how many clean boxes of each class it would contribute
    by_day = cb.groupby(["date", "type"]).size().unstack(fill_value=0)
    for c in CLASSES:
        if c not in by_day.columns:
            by_day[c] = 0

    val_days: set[str] = set()
    got = {c: 0 for c in CLASSES}
    # rare classes first (fewest available -> hardest to satisfy), III last
    order = sorted(CLASSES, key=lambda c: by_day[c].sum())
    for cls in order:
        need = per_class_target.get(cls, 0)
        # prefer days rich in THIS class but poor in the classes already
        # satisfied, so filling II doesn't drag in a pile of surplus III
        cand = by_day[~by_day.index.isin(val_days) & (by_day[cls] > 0)]
        cand = cand.sort_values(cls, ascending=False)
        for day, row in cand.iterrows():
            if got[cls] >= need:
                break
            val_days.add(day)
            for c in CLASSES:
                got[c] += int(row[c])

    print("  val (per-class box budget):")
    for c in CLASSES:
        avail = int(by_day[c].sum())
        print(f"    {c:4s} target {per_class_target.get(c, 0):4d}  got {got[c]:4d}  "
              f"of {avail:4d} clean available")
    print(f"    -> {len(val_days)} station-days; the other "
          f"{len(usable[~usable.date.isin(val_days)])} clean positive windows go to train")
    return val_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--img-size", nargs=2, type=int, default=[640, 640], metavar=("H", "W"))
    ap.add_argument("--val-boxes", nargs=3, type=int, default=[20, 180, 20],
                    metavar=("II", "III", "V"),
                    help="per-class val BOX budget (see choose_val_days). Absolute counts, not "
                         "a fraction, so continued reviewing compounds into TRAIN instead of "
                         "being split away. Defaults: 20/180/20 -- enough III to keep metric "
                         "noise where it already is, and a floor under the rare classes that "
                         "the old rule left at 12 and 7.")
    ap.add_argument("--train-source", default="both", choices=["both", "noisy", "clean"],
                    help="which POSITIVE windows may enter train: 'clean' = only windows whose "
                         "every box was reviewed usable; 'noisy' = only the rest; 'both' = union. "
                         "Negatives are unaffected -- they need no review, a window with no "
                         "cataloged event is clean by construction.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--drop-contaminated-negatives",
                    default="data/ecallisto/windows/contaminated_negatives.csv",
                    help="CSV of negative windows a detector flagged as actually containing a "
                         "burst; see drop_contaminated_negatives() for the two caveats. "
                         "Pass '' to keep them.")
    ap.add_argument("--contaminated-negatives-scope", default="both",
                    choices=["both", "train"],
                    help="which split --drop-contaminated-negatives applies to. 'both' (default, "
                         "kept so datasets built before this flag stay reproducible) also drops "
                         "them from val, which silently changes the measuring stick: those "
                         "windows are the ones the detector fires on, so removing them lifts AP "
                         "on its own. Use 'train' for any experiment whose result is compared "
                         "against a run built without the drop.")
    ap.add_argument("--single-class", action="store_true",
                    help="collapse II/III/V into one 'burst' class. Detection+localisation is "
                         "the primary objective and is not data-limited (1694 Arecibo positives), "
                         "whereas classification is: Type V has 27 clean samples TOTAL, so val and "
                         "train cannot both be satisfied (a 20-box val leaves 7 to train on). "
                         "Splitting 237 boxes across three heads also made training unstable -- "
                         "recall swung 0.06<->0.64 between epochs. One class uses every box for "
                         "the one question that matters.")
    ap.add_argument("--debias-shift-s", type=float, default=0.0,
                    help="shift TRAIN boxes whose time_source=='catalog' later by this many "
                         "seconds (0 = off). See debias_catalog_boxes().")
    ap.add_argument("--debias-width-cap-s", type=float, default=0.0,
                    help="cap the width of those same boxes, in seconds (0 = no cap)")
    ap.add_argument("--min-window-frac", type=float, default=0.0,
                    help="drop windows shorter than this fraction of the modal window length "
                         "(0 = off). See drop_short_windows(). Left off by default so datasets "
                         "built before it stay reproducible.")
    ap.add_argument("--gap-boxes", default="data/ecallisto/windows/gap_boxes.csv",
                    help="extra boxes for bursts the catalog never recorded (written by "
                         "apply_gap_annotations.py). They live outside boxes.csv on purpose: "
                         "refresh_boxes.py REBUILDS boxes.csv from the Monstein catalog, so a "
                         "non-catalog box placed there is erased on the next refresh. Pass '' "
                         "to exclude them.")
    ap.add_argument("--stations", nargs="*", default=None,
                    help="restrict to these locations (default: all present). REQUIRED ONCE A "
                         "SECOND STATION IS IN THE POOL: keep_dominant_config now merges configs "
                         "within 1%%, and Arecibo (15.000-86.625) and Australia-ASSA "
                         "(14.995-86.933) are 0.36%% apart -- so they land in the SAME config "
                         "group by design, and a rebuild of the Arecibo-only Phase 1 dataset "
                         "would silently absorb ASSA windows without any warning. Naming the "
                         "stations makes each dataset say out loud what it is built from.")
    ap.add_argument("--keep-all-configs", action="store_true",
                    help="don't restrict to the dominant instrument config (Phase 2 only -- "
                         "mixing configs reintroduces the pixel-slope inconsistency)")
    args = ap.parse_args()

    windows = pd.read_csv(os.path.join(args.window_dir, "windows.csv"), dtype={"date": str})
    boxes = pd.read_csv(os.path.join(args.window_dir, "boxes.csv"), dtype={"date": str})
    boxes["review_status"] = boxes["review_status"].fillna("")
    if args.gap_boxes and os.path.exists(args.gap_boxes):
        gap = pd.read_csv(args.gap_boxes, dtype={"date": str})
        gap["review_status"] = gap["review_status"].fillna("")
        boxes = pd.concat([boxes, gap], ignore_index=True)
        print(f"  gap boxes: +{len(gap)} from {os.path.basename(args.gap_boxes)} "
              f"({gap.time_source.value_counts().to_dict()})")

    if args.overwrite and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "labels", split), exist_ok=True)

    print(f"{len(windows)} windows, {int(windows.excluded.sum())} excluded, {len(boxes)} boxes")
    if args.stations:
        missing = set(args.stations) - set(windows.location.unique())
        if missing:
            raise SystemExit(f"--stations names {sorted(missing)}, not present in "
                             f"{sorted(windows.location.unique())}")
        windows = windows[windows.location.isin(args.stations)]
        print(f"  stations: {args.stations} -> {len(windows)} windows")
    elif windows.location.nunique() > 1:
        print(f"  !! WARNING: {windows.location.nunique()} stations in the pool "
              f"({sorted(windows.location.unique())}) and --stations was not given.\n"
              f"  !! Configs within 1%% are merged, so these may be combined into one dataset.\n"
              f"  !! Pass --stations explicitly unless a cross-station mix is what you want.")
    if not args.keep_all_configs:
        windows = keep_dominant_config(windows)
    windows = drop_short_windows(windows, args.min_window_frac)
    if args.contaminated_negatives_scope == "both":
        windows = drop_contaminated_negatives(windows, args.drop_contaminated_negatives)
    boxes = boxes[boxes.file_name.isin(set(windows.file_name))]
    val_days = choose_val_days(windows, boxes, dict(zip(CLASSES, args.val_boxes)))
    if args.contaminated_negatives_scope == "train":
        # after the split is fixed, so val keeps every window it would have had
        # without this flag -- same reasoning as the de-bias block below
        windows = drop_contaminated_negatives(windows, args.drop_contaminated_negatives,
                                              protect_days=val_days)
        boxes = boxes[boxes.file_name.isin(set(windows.file_name))]
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
        # positivity comes from the MERGED boxes, not windows.csv's n_boxes:
        # n_boxes counts catalog boxes only, so a window whose sole burst was a
        # catalog gap still reads n_boxes==0 and would be filtered as if it were
        # a negative while simultaneously being written with labels
        if split == "train" and len(wb) > 0:
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
            cls_id = 0 if args.single_class else CLASS_ID[str(b.type)]
            cx = (b.box_start_col + b.box_end_col) / 2 / w.n_cols
            bw = (b.box_end_col - b.box_start_col) / w.n_cols
            # full frequency height: centre 0.5, height 1.0 (deliberate, see docstring)
            lines.append(f"{cls_id} {cx:.6f} 0.500000 {bw:.6f} 1.000000")
        with open(os.path.join(args.out_dir, "labels", split, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))

        counts[split][0] += 1
        counts[split][1] += 1 if lines else 0
        counts[split][2] += len(lines)

    yaml = os.path.join(args.out_dir, "data.yaml")
    with open(yaml, "w") as fh:
        fh.write(f"path: {os.path.abspath(args.out_dir)}\n"
                 "train: images/train\nval: images/val\n\nnames:\n")
        for i, c in enumerate(["burst"] if args.single_class else CLASSES):
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
