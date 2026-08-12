"""
extract_events.py

Given a downloaded StationDay (see fetch.py) and the Monstein catalog rows for
that station/date (see burst_catalog.py), crop out one .npy per cataloged burst
plus a metadata row, WITHOUT any resizing:

  - width (time axis) = the event's own start/end time (from the catalog) plus a
    fixed physical-duration buffer on each side. Not forced to a fixed pixel
    count here -- see the note in fetch.py/README about why the frequency axis
    also can't just be resized blindly across stations. Buffer, not a fixed
    window, so short (Type III, often <1 min) and long (Type II, can be several
    minutes) events both keep their true extent instead of being truncated or
    mostly-padding.
  - height (frequency axis) = kept exactly as downloaded (no split into two
    tiles, no resize). freq_mhz is saved alongside so a later step can align
    multiple stations onto a common frequency band before resizing for a model.

Metadata schema matches our own station's burst_list_240330_240729.csv
(file_name, date, location, start_time, end_time, type) with extra calibration
columns appended, so the two sources can be concatenated directly.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fetch import StationDay

METADATA_COLUMNS = [
    "file_name",
    "date",
    "location",
    "start_time",
    "end_time",
    "type",
    "event_start_time",
    "event_end_time",
    "other_stations",
    "uncertain",
    "freq_min_mhz",
    "freq_max_mhz",
    "n_freq_channels",
    "sample_interval_s",
]


def _parse_time_range(time_range: str, day: str) -> tuple[datetime, datetime]:
    """'HH:MM-HH:MM' (or HH:MM:SS variants) + 'YYYYMMDD' -> (start_dt, end_dt), handles midnight wrap."""
    start_str, end_str = time_range.split("-")
    base = datetime.strptime(day, "%Y%m%d")

    def to_dt(s: str) -> datetime:
        s = s.strip()
        fmt = "%H:%M:%S" if s.count(":") == 2 else "%H:%M"
        t = datetime.strptime(s, fmt).time()
        return datetime.combine(base.date(), t)

    start_dt, end_dt = to_dt(start_str), to_dt(end_str)
    if end_dt < start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def extract_events_for_day(
    station_day: StationDay,
    catalog_rows: pd.DataFrame,
    out_dir: str,
    buffer_s: float = 60.0,
) -> pd.DataFrame:
    """
    Crop every cataloged event in `catalog_rows` out of `station_day` and save it.

    Args:
        station_day: result of fetch.fetch_station_day() for this station/date.
        catalog_rows: rows of the exploded Monstein catalog already filtered to
            this station and date (columns: date, time_range, type, station,
            uncertain -- see burst_catalog.explode_by_station/filter_catalog).
        out_dir: directory to write burst-<station>-<date>-<time>.npy into.
        buffer_s: seconds of context kept before/after the catalogued burst time.

    Returns:
        DataFrame of new metadata rows (METADATA_COLUMNS), one per successfully
        extracted event. Events that fall outside the downloaded day's time
        range are skipped with a printed warning (the archive's per-file listing
        sometimes doesn't cover a full 24h, e.g. outside daylight hours).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

    rows = []
    for _, ev in catalog_rows.iterrows():
        try:
            ev_start, ev_end = _parse_time_range(ev["time_range"], ev["date"])
        except ValueError:
            print(f"    [extract] could not parse time_range={ev['time_range']!r}, skipping")
            continue

        win_start = ev_start - timedelta(seconds=buffer_s)
        win_end = ev_end + timedelta(seconds=buffer_s)

        # has_gap() checks real per-file coverage (not a naive day_start/day_end
        # span), so an event that falls after a mid-day recording gap is still
        # found correctly instead of being wrongly reported as "outside range".
        if station_day.has_gap(win_start, win_end):
            print(f"    [extract] event {ev['time_range']} on {ev['date']} not fully covered "
                  f"by downloaded data for {station_day.station} (gap or out of range), skipping")
            continue

        col_start = station_day.time_to_column(win_start)
        col_end = station_day.time_to_column(win_end)
        if col_start is None or col_end is None or col_end <= col_start:
            continue

        crop = station_day.spectrogram[:, col_start:col_end]

        # type + end time in the name: two distinct cataloged events can share
        # the same start minute:second (real case hit while testing --
        # ALASKA-ANCHORAGE 2023-04-21 lists both "22:22-22:27 II?" and
        # "22:22-22:24 III"). Without disambiguation the second np.save()
        # silently overwrites the first file while both metadata rows still
        # claim to point at it -- a mismatched label, not a crash, so it would
        # have gone unnoticed without an eyeball check of a real batch.
        # raw type codes can contain '/', '?', ',' (e.g. "CTM/CAU", "II?", "III,V") -- not filename-safe
        safe_type = re.sub(r"[^A-Za-z0-9]+", "", str(ev["type"])) or "unk"
        fname = (
            f"burst-{station_day.station}-{ev_start.strftime('%m-%d-%Y')}-"
            f"{ev_start.strftime('%H%M%S')}-{ev_end.strftime('%H%M%S')}-type{safe_type}.npy"
        )
        if os.path.exists(os.path.join(out_dir, fname)):
            print(f"    [extract] {fname} already exists (duplicate catalog row?), skipping")
            continue
        np.save(os.path.join(out_dir, fname), crop)

        rows.append(
            {
                "file_name": fname,
                "date": ev["date"],
                "location": station_day.station,
                "start_time": win_start.strftime("%H:%M:%S"),
                "end_time": win_end.strftime("%H:%M:%S"),
                "type": ev["type"],
                "event_start_time": ev_start.strftime("%H:%M:%S"),
                "event_end_time": ev_end.strftime("%H:%M:%S"),
                "other_stations": ",".join(ev.get("all_stations", [])),
                "uncertain": bool(ev.get("uncertain", False)),
                "freq_min_mhz": float(station_day.freq_mhz.min()),
                "freq_max_mhz": float(station_day.freq_mhz.max()),
                "n_freq_channels": len(station_day.freq_mhz),
                "sample_interval_s": station_day.sample_interval_s,
            }
        )

    # save this station/day's frequency axis once (shared by every crop from this station+config)
    if rows:
        freq_path = os.path.join(
            out_dir, "meta", f"freq-{station_day.station}-{station_day.date}.npy"
        )
        np.save(freq_path, station_day.freq_mhz)

    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


def append_metadata(new_rows: pd.DataFrame, metadata_csv: str) -> None:
    """Append new_rows to metadata_csv, writing the header only if the file is new."""
    write_header = not os.path.exists(metadata_csv)
    new_rows.to_csv(metadata_csv, mode="a", header=write_header, index=False)


def _available_intervals(station_day: StationDay) -> list[tuple[datetime, datetime]]:
    """Time ranges actually backed by downloaded data, merging back-to-back files.
    Built from real per-file timestamps, NOT an assumed contiguous day span --
    see the StationDay docstring in fetch.py for why that distinction matters."""
    intervals: list[tuple[datetime, datetime]] = []
    for f in station_day.files:
        if intervals and (f.time_obs - intervals[-1][1]).total_seconds() <= station_day.sample_interval_s * 2:
            intervals[-1] = (intervals[-1][0], f.time_end)
        else:
            intervals.append((f.time_obs, f.time_end))
    return intervals


def _free_intervals(
    station_day: StationDay, all_day_events: pd.DataFrame, buffer_s: float
) -> list[tuple[datetime, datetime]]:
    """Time ranges that are (a) actually covered by downloaded data and (b) NOT
    within buffer_s of any cataloged event (any type -- deliberately not just the
    types we're extracting as positives, so an out-of-scope-but-real event can't
    get sampled as a negative)."""
    occupied = []
    for _, ev in all_day_events.iterrows():
        try:
            s, e = _parse_time_range(ev["time_range"], ev["date"])
        except ValueError:
            continue
        occupied.append((s - timedelta(seconds=buffer_s), e + timedelta(seconds=buffer_s)))
    occupied.sort()

    free = []
    for a_start, a_end in _available_intervals(station_day):
        cursor = a_start
        for o_start, o_end in occupied:
            o_start, o_end = max(o_start, a_start), min(o_end, a_end)
            if o_start >= o_end:
                continue
            if o_start > cursor:
                free.append((cursor, o_start))
            cursor = max(cursor, o_end)
        if cursor < a_end:
            free.append((cursor, a_end))
    return free


def sample_negative_windows(
    station_day: StationDay,
    all_day_events: pd.DataFrame,
    out_dir: str,
    n_negatives: int,
    buffer_s: float,
    duration_pool_s: list[float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Randomly sample `n_negatives` windows from parts of this station-day that are
    NOT within buffer_s of any cataloged event (as confirmed: "从没有任何目录事件
    覆盖的时间段里随机采样").

    Window durations are drawn from `duration_pool_s` (pass in the observed
    positive-event durations for the run) rather than fixed, so negatives aren't
    trivially separable from positives by width alone -- a classifier that could
    tell burst vs. non-burst just by how wide the crop is would be learning an
    artifact of how the dataset was built, not the actual signal.

    Returns a DataFrame in the same METADATA_COLUMNS as extract_events_for_day,
    with type="0" (no_burst) and empty event_start_time/event_end_time (there is
    no cataloged event to report a sub-window for).
    """
    os.makedirs(out_dir, exist_ok=True)
    free = _free_intervals(station_day, all_day_events, buffer_s)
    if not free or not duration_pool_s:
        return pd.DataFrame(columns=METADATA_COLUMNS)

    rows = []
    attempts, max_attempts = 0, n_negatives * 20
    while len(rows) < n_negatives and attempts < max_attempts:
        attempts += 1
        duration = float(rng.choice(duration_pool_s))
        candidates = [(s, e) for s, e in free if (e - s).total_seconds() >= duration]
        if not candidates:
            continue
        weights = np.array([(e - s).total_seconds() for s, e in candidates])
        s, e = candidates[rng.choice(len(candidates), p=weights / weights.sum())]
        max_offset = (e - s).total_seconds() - duration
        win_start = s + timedelta(seconds=float(rng.uniform(0, max_offset)))
        win_end = win_start + timedelta(seconds=duration)

        col_start = station_day.time_to_column(win_start)
        col_end = station_day.time_to_column(win_end)
        if col_start is None or col_end is None or col_end <= col_start:
            continue

        fname = (
            f"nonburst-{station_day.station}-{win_start.strftime('%m-%d-%Y')}-"
            f"{win_start.strftime('%H%M%S')}.npy"
        )
        np.save(os.path.join(out_dir, fname), station_day.spectrogram[:, col_start:col_end])
        rows.append(
            {
                "file_name": fname,
                "date": station_day.date,
                "location": station_day.station,
                "start_time": win_start.strftime("%H:%M:%S"),
                "end_time": win_end.strftime("%H:%M:%S"),
                "type": "0",
                "event_start_time": "",
                "event_end_time": "",
                "other_stations": "",
                "uncertain": False,
                "freq_min_mhz": float(station_day.freq_mhz.min()),
                "freq_max_mhz": float(station_day.freq_mhz.max()),
                "n_freq_channels": len(station_day.freq_mhz),
                "sample_interval_s": station_day.sample_interval_s,
            }
        )

    return pd.DataFrame(rows, columns=METADATA_COLUMNS)
