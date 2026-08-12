"""
extract_windows.py

Second extraction path, for the object-detection training set (see
TRAINING_PLAN.md). Saves the archive's own 15-minute FITS files as whole
windows instead of cropping `event ± buffer_s` like extract_events.py does.

Why a second path rather than reusing extract_events.py's output:
extract_events.py centres every event in its crop by construction, and that
showed up in the real output as total label leakage -- 1694 positive events,
normalized box centre mean 0.5000, **std 0.0000**. A detector that always
predicts a box at the horizontal centre would score well on that dataset
without learning anything about bursts, which makes the Phase 1 mAP number
uninterpretable -- and producing an interpretable mAP is the entire point of
Phase 1. Using the archive's own 15-minute files as the unit puts the event
wherever it really was, and is also what the published e-Callisto detection
papers use as their input unit.

Three more things fall out of this for free:
  - negatives are just windows with no cataloged event (extract_events.py's
    sample_negative_windows() machinery isn't needed here)
  - every window has identical dimensions, so YOLO's letterbox resize applies
    the SAME scale factor to all of them -- a given true drift rate maps to a
    given pixel slope, which is the property the whole detection approach was
    chosen to preserve
  - context slots around each event get cached too (--context-slots), so
    changing the canvas length later doesn't force another download

Box times come from, in priority order:
  1. an event_review manual correction, if one exists for this event
  2. the catalog's own start/end time
  3. for zero-width catalog entries (36% of positives), a fitted default of
     catalog_start + ZERO_WIDTH_OFFSET_S, width ZERO_WIDTH_DURATION_S
     (see TRAINING_PLAN.md "Bounding box 构造" for how those were derived
     and their measured held-out accuracy)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from extract_events import _parse_time_range
from fetch import StationDay

SLOT_S = 15 * 60

# Fitted on the 69 zero-width-catalog events that have a human time correction:
# grid search gave median IoU 0.50 / 62% at IoU>=0.5 in-sample, and 58%
# (5-95 pct: 46-69%) under 200 split-half trials. Only fitted on Arecibo --
# re-derive before trusting these on another station.
ZERO_WIDTH_OFFSET_S = 10.0
ZERO_WIDTH_DURATION_S = 35.0

# Catalog sanity cap. `_parse_time_range` adds a day whenever end < start,
# which is right for a genuine midnight wrap but turns a catalog TYPO into an
# all-day event: real cases found on Arecibo 2022-2023, "18:27-18:26" (III) ->
# 1439 min and "19:00-10:02" (III) -> 902 min. Left alone, each paints a
# full-width box across every window of that day. Only 2 of 1759 target events
# exceed 60 min and both are these typos, while the longest plausible real
# event in the same set is a 34-minute Type II -- so 60 min removes exactly the
# artifacts and nothing real. Skipped loudly, never silently.
MAX_EVENT_DURATION_S = 60 * 60

# An event straddling a window boundary lands in BOTH windows, and the smaller
# share can be a useless sliver: a real case in the pilot put 4 columns (1.0 s)
# of a 19:15-19:19 event in one window and the other 956 in the next. A
# full-height 1-second box has no recognizable burst structure in it -- it is
# just a labeled vertical line, which is exactly what an RFI spike looks like.
# 5 s is comfortably below every genuine box seen (the narrowest human-marked
# burst in the pilot is 10 s, the 25th percentile is 20 s) and above the sliver.
# A window whose ONLY box got dropped this way is excluded rather than demoted
# to a negative -- there really is a burst in it, just not enough of one.
MIN_BOX_S = 5.0

WINDOW_COLUMNS = [
    "file_name", "date", "location", "win_start_time", "win_end_time",
    "n_freq_channels", "sample_interval_s", "freq_min_mhz", "freq_max_mhz",
    "n_cols", "n_boxes", "excluded", "exclude_reason", "offclass_types",
]

BOX_COLUMNS = [
    "file_name", "date", "location", "type",
    "event_start_time", "event_end_time",
    "box_start_time", "box_end_time", "box_start_col", "box_end_col",
    "time_source", "review_status", "uncertain", "clipped",
]


# ---------------------------------------------------------------------------
# event_review manual corrections -> absolute times
# ---------------------------------------------------------------------------

def _event_key(location: str, date: str, ev_start: str, ev_end: str, ev_type: str) -> tuple:
    """Identity of a cataloged event, independent of how it was cropped.

    Deliberately NOT the .npy file name: review_status.csv keys on
    extract_events.py's crop names, which encode the buffer and would break the
    moment the crop definition changes -- which is exactly what this module
    does. (location, date, catalog start, catalog end, type) is the underlying
    catalog identity and survives the change."""
    return (str(location), str(date), str(ev_start), str(ev_end), str(ev_type))


def load_corrections(review_csv: str, old_metadata_csv: str) -> dict[tuple, dict]:
    """Build {event_key: {"status", "start_dt", "end_dt"}} from event_review's
    review_status.csv joined to the OLD (extract_events.py) metadata.csv.

    The join is needed because review_status.csv stores only a crop file name
    plus offsets, and `manual_burst_range_json` holds **seconds from the crop's
    own start**, not absolute times -- and that crop start is
    `event_start - buffer_s`, so forgetting the offset shifts every corrected
    box by a full minute. The old metadata.csv is what carries the crop's
    absolute start_time.
    """
    if not (os.path.exists(review_csv) and os.path.exists(old_metadata_csv)):
        print(f"  [windows] no corrections loaded (missing {review_csv} or {old_metadata_csv})")
        return {}

    # keep_default_na=False: real bug hit in event_review/review_store.py --
    # empty fields become float NaN otherwise, and bool(nan) is True, so
    # `if row["manual_burst_range_json"]` passes and json.loads(nan) crashes
    rev = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    meta = pd.read_csv(old_metadata_csv, dtype=str, keep_default_na=False)
    df = meta.merge(rev, on="file_name", how="inner")

    out: dict[tuple, dict] = {}
    for _, r in df.iterrows():
        if not r["event_start_time"]:
            continue  # negative sample row, nothing to correct
        key = _event_key(r["location"], r["date"], r["event_start_time"], r["event_end_time"], r["type"])
        entry = {"status": r["status"], "start_dt": None, "end_dt": None}

        js = r["manual_burst_range_json"]
        if js:
            import json
            d = json.loads(js)
            base = datetime.strptime(r["date"], "%Y%m%d")
            crop_start = datetime.combine(base.date(), datetime.strptime(r["start_time"], "%H:%M:%S").time())
            ev_start = datetime.combine(base.date(), datetime.strptime(r["event_start_time"], "%H:%M:%S").time())
            # a crop for an event just after midnight starts on the PREVIOUS
            # day (buffer_s=60 pushes it back across 00:00); combining its
            # clock time with `date` would then be a full day late
            if crop_start > ev_start:
                crop_start -= timedelta(days=1)
            entry["start_dt"] = crop_start + timedelta(seconds=float(d["start_s"]))
            entry["end_dt"] = crop_start + timedelta(seconds=float(d["end_s"]))
        out[key] = entry
    print(f"  [windows] loaded {len(out)} reviewed events "
          f"({sum(1 for v in out.values() if v['start_dt'] is not None)} with a manual time correction)")
    return out


# ---------------------------------------------------------------------------
# per-day window extraction
# ---------------------------------------------------------------------------

def _box_times(ev_start: datetime, ev_end: datetime, correction: dict | None) -> tuple[datetime, datetime, str]:
    """Resolve one event's box time range and record which source it came from."""
    if correction is not None and correction["start_dt"] is not None:
        return correction["start_dt"], correction["end_dt"], "manual"
    if ev_end <= ev_start:
        start = ev_start + timedelta(seconds=ZERO_WIDTH_OFFSET_S)
        return start, start + timedelta(seconds=ZERO_WIDTH_DURATION_S), "zero_width_default"
    return ev_start, ev_end, "catalog"


def _parsed_events(rows: pd.DataFrame, cap_duration: bool = False) -> list[tuple[datetime, datetime, pd.Series]]:
    """Parse `time_range` into absolute datetimes.

    cap_duration: drop events longer than MAX_EVENT_DURATION_S. Only applied to
    the events that become BOXES -- off-class events are allowed to be long,
    because continuum entries (CTM median ~112 min, Type VI) genuinely last
    hours and we only use them to decide whether a window is a clean negative.
    """
    out = []
    for _, ev in rows.iterrows():
        try:
            s, e = _parse_time_range(ev["time_range"], ev["date"])
        except ValueError:
            print(f"    [windows] could not parse time_range={ev['time_range']!r}, skipping")
            continue
        if cap_duration and (e - s).total_seconds() > MAX_EVENT_DURATION_S:
            print(f"    [windows] {ev['date']} {ev['time_range']} type={ev['type']} parses to "
                  f"{(e - s).total_seconds() / 60:.0f} min -- catalog typo (end before start), skipping box")
            continue
        out.append((s, e, ev))
    return out


def boxes_for_window(
    fname: str,
    station: str,
    date: str,
    win_start: datetime,
    win_end: datetime,
    n_cols: int,
    sample_interval_s: float,
    targets: list,
    off_class: list,
    corrections: dict[tuple, dict],
) -> tuple[list[dict], bool, str, list[str]]:
    """All the box/exclusion logic for ONE window, given only its time bounds.

    Factored out so the two callers cannot drift apart: extract_windows_for_day
    (during scraping, bounds from the FITS file) and refresh_boxes.py (offline,
    bounds read back from windows.csv). Re-deriving boxes offline is what makes
    newly-reviewed events usable without another 9-hour download, so these two
    paths get run at very different times and MUST agree.

    Returns (box_rows, excluded, exclude_reason, overlapping_offclass_types).
    """
    def cols(t: datetime) -> int:
        return int(round((t - win_start).total_seconds() / sample_interval_s))

    min_box_cols = max(1, int(round(MIN_BOX_S / sample_interval_s)))
    this_boxes: list[dict] = []
    statuses: list[str] = []
    dropped_sliver = False

    for s, e, ev in targets:
        key = _event_key(station, date, s.strftime("%H:%M:%S"), e.strftime("%H:%M:%S"), str(ev["type"]))
        b_start, b_end, source = _box_times(s, e, corrections.get(key))
        if b_end <= win_start or b_start >= win_end:
            continue
        c0, c1 = max(0, cols(b_start)), min(n_cols, cols(b_end))
        if c1 - c0 < min_box_cols:
            # entirely outside this window's real columns, or only an edge
            # sliver of it landed here (see MIN_BOX_S)
            dropped_sliver = True
            continue
        status = corrections.get(key, {}).get("status", "")
        statuses.append(status)
        this_boxes.append({
            "file_name": fname, "date": date, "location": station, "type": ev["type"],
            "event_start_time": s.strftime("%H:%M:%S"), "event_end_time": e.strftime("%H:%M:%S"),
            "box_start_time": b_start.strftime("%H:%M:%S"), "box_end_time": b_end.strftime("%H:%M:%S"),
            "box_start_col": c0, "box_end_col": c1,
            "time_source": source, "review_status": status,
            "uncertain": bool(ev.get("uncertain", False)),
            "clipped": bool(cols(b_start) < 0 or cols(b_end) > n_cols),
        })

    overlapping_off = sorted({str(ev["type"]) for s, e, ev in off_class
                              if win_end > s and win_start < e})

    excluded, reason = False, ""
    if dropped_sliver and not this_boxes:
        excluded, reason = True, "only_edge_sliver_of_an_event"
    elif overlapping_off and not this_boxes:
        # An off-class event only makes a window unusable when that window would
        # otherwise be a NEGATIVE: calling it "no burst here" would be a lie.
        # A window that already has a II/III/V box is still a valid positive --
        # a continuum storm in the background doesn't make its box wrong, and
        # excluding those cost 12% of all Arecibo positives (47/435 station-days
        # carry a >60min CTM/VI entry) for no gain. The off-class types are
        # recorded either way so a stricter downstream filter stays possible.
        excluded, reason = True, "offclass_event_in_negative_window"
    elif this_boxes and all(st == "discard" for st in statuses):
        # every labelable burst in this window was human-rejected: it can't
        # be a positive (the box would point at nothing) and it isn't a
        # clean negative either (a real burst is cataloged here)
        excluded, reason = True, "all_boxes_review_discard"

    return this_boxes, excluded, reason, overlapping_off


def split_targets_offclass(target_rows, all_type_rows):
    """(targets, off_class) as boxes_for_window() expects them."""
    targets = _parsed_events(target_rows, cap_duration=True)
    all_events = _parsed_events(all_type_rows)
    target_ids = {(s, e, str(ev["type"])) for s, e, ev in targets}
    off_class = [(s, e, ev) for s, e, ev in all_events if (s, e, str(ev["type"])) not in target_ids]
    return targets, off_class


def extract_windows_for_day(
    station_day: StationDay,
    target_rows: pd.DataFrame,
    all_type_rows: pd.DataFrame,
    out_dir: str,
    corrections: dict[tuple, dict] | None = None,
    context_slots: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save one .npy per downloaded 15-minute FITS file that is within
    `context_slots` slots of a target event, plus its boxes.

    Args:
        station_day: fetch.fetch_station_day() result.
        target_rows: catalog rows for THIS station/date, already filtered to the
            burst types we train on (II/III/V) -- these become boxes.
        all_type_rows: catalog rows for this station/date with NO type filter.
            Used only to detect events we can't label: a window containing a
            Type IV can neither get a box (not one of our classes) nor count as
            a clean negative (there IS a burst in it), so it gets excluded.
            Same reasoning extract_events.py already applies in _free_intervals().
        corrections: load_corrections() output, or None.
        context_slots: how many 15-min slots on each side of an event to also
            keep. 2 (=+/-30 min) is the agreed cache depth: it covers any canvas
            length up to 75 minutes without another download, at ~19.8 GB for
            all 7 stations vs 52.2 GB for full days.

    Returns:
        (windows_df, boxes_df) -- one row per saved window, one per box.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    corrections = corrections or {}

    targets, off_class = split_targets_offclass(target_rows, all_type_rows)
    if not targets:
        return pd.DataFrame(columns=WINDOW_COLUMNS), pd.DataFrame(columns=BOX_COLUMNS)

    context = timedelta(seconds=context_slots * SLOT_S)
    win_rows, box_rows = [], []

    for f in station_day.files:
        # keep this file only if it's within the context window of a target event
        if not any(f.time_end >= s - context and f.time_obs <= e + context for s, e, _ in targets):
            continue

        fname = f"win-{station_day.station}-{station_day.date}-{f.time_obs.strftime('%H%M%S')}.npy"
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            print(f"    [windows] {fname} already exists, skipping")
            continue

        n_cols = f.data.shape[1]
        this_boxes, excluded, reason, overlapping_off = boxes_for_window(
            fname, station_day.station, station_day.date, f.time_obs, f.time_end,
            n_cols, station_day.sample_interval_s, targets, off_class, corrections,
        )

        np.save(out_path, f.data)
        win_rows.append({
            "file_name": fname, "date": station_day.date, "location": station_day.station,
            "win_start_time": f.time_obs.strftime("%H:%M:%S"),
            "win_end_time": f.time_end.strftime("%H:%M:%S"),
            "n_freq_channels": len(station_day.freq_mhz),
            "sample_interval_s": station_day.sample_interval_s,
            "freq_min_mhz": float(station_day.freq_mhz.min()),
            "freq_max_mhz": float(station_day.freq_mhz.max()),
            "n_cols": n_cols, "n_boxes": len(this_boxes),
            "excluded": excluded, "exclude_reason": reason,
            "offclass_types": ",".join(overlapping_off),
        })
        box_rows.extend(this_boxes)

    if win_rows:
        np.save(os.path.join(out_dir, "meta",
                             f"freq-{station_day.station}-{station_day.date}.npy"), station_day.freq_mhz)

    return (pd.DataFrame(win_rows, columns=WINDOW_COLUMNS),
            pd.DataFrame(box_rows, columns=BOX_COLUMNS))


def append_csv(rows: pd.DataFrame, path: str) -> None:
    if rows.empty:
        return
    rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
