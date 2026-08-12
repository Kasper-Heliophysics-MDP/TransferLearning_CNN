"""
clean_ecallisto_events.py

Wires the real Monstein catalog burst times that scrape.py already saves into
metadata.csv (event_start_time/event_end_time) into sumthreshold_denoise's
known_burst_weight, so eCallisto events get the same ground-truth burst
protection as own-station data instead of relying on blind auto-detection
(confirmed unreliable across stations -- see 开发日志.md).

Unlike sumthreshold_cleaning_wrapper (built for own-station's CSV/DataFrame,
(time, frequency) input), extract_events.py already saves .npy crops as
(frequency, time) -- the orientation sumthreshold_denoise.clean() natively
expects -- so this calls clean() directly rather than going through that
wrapper's transpose/DataFrame handling.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sumthreshold_denoise import clean, _REFERENCE_TIME_WIN, _REFERENCE_SAMPLE_INTERVAL_S


def event_burst_indices(row: pd.Series, sample_interval_s: float) -> tuple[int, int] | None:
    """
    (start_idx, end_idx) of the real cataloged burst WITHIN the saved crop.

    The crop spans `start_time` to `end_time` (the catalog's event time +/-
    extract_events.py's buffer_s), NOT event_start_time/event_end_time (the
    catalog's own, un-buffered event window) -- so the event's position
    inside the array has to be computed from the offset between the two, not
    assumed to start at column 0.

    Returns None for negative (non-burst) rows, where event_start_time is
    blank by construction (see extract_events.sample_negative_windows).
    """
    if pd.isna(row.get("event_start_time")) or str(row.get("event_start_time", "")) == "":
        return None

    def to_dt(s: str) -> datetime:
        # own-station timestamps carry millisecond precision (derived from
        # real per-sample CSV timestamps); eCallisto's never do (human-curated
        # catalog values) -- this path was only ever exercised against
        # eCallisto data until now, so the no-fractional-seconds assumption
        # went unnoticed
        s = str(s)
        fmt = "%H:%M:%S.%f" if "." in s else "%H:%M:%S"
        return datetime.strptime(s, fmt)

    win_start = to_dt(str(row["start_time"]))
    ev_start = to_dt(str(row["event_start_time"]))
    ev_end = to_dt(str(row["event_end_time"]))
    # extract_events_for_day already handles midnight wrap when building
    # start_time/event_start_time, but the two are parsed independently here
    # (no shared date) -- realign if the naive HH:MM:SS comparison suggests
    # the event happened "before" its own window's start
    if ev_start < win_start:
        ev_start += timedelta(days=1)
        ev_end += timedelta(days=1)

    start_idx = round((ev_start - win_start).total_seconds() / sample_interval_s)
    end_idx = round((ev_end - win_start).total_seconds() / sample_interval_s)
    return start_idx, end_idx


def clean_event_file(
    npy_path: str,
    row: pd.Series,
    method: str = "comprehensive",
    known_burst_margin: int = 10,
) -> dict:
    """
    Load one extract_events.py-produced .npy and run it through clean() with
    the real catalog burst time as known_burst_weight (if this is a positive
    event row -- negatives get no protection, there's nothing to protect).

    Returns the full clean() result dict (background, residual, cleaned,
    burst_weight, etc.) plus the raw loaded data under 'raw', so callers can
    plot before/after without re-loading the file.
    """
    data = np.load(npy_path).astype(np.float32)
    sample_interval_s = float(row["sample_interval_s"])

    known_burst_weight = None
    indices = event_burst_indices(row, sample_interval_s)
    if indices is not None:
        start_idx, end_idx = indices
        lo = max(0, start_idx - known_burst_margin)
        hi = min(data.shape[1], end_idx + known_burst_margin)
        known = np.zeros(data.shape[1], dtype=np.float32)
        known[lo:hi] = 1.0
        known_burst_weight = np.broadcast_to(known, data.shape)

    # same sample-rate-aware time_win scaling as sumthreshold_cleaning_wrapper
    time_win = round(_REFERENCE_TIME_WIN * _REFERENCE_SAMPLE_INTERVAL_S / sample_interval_s)
    time_win = max(time_win // 2 * 2 + 1, 3)

    method_params = {
        "comprehensive": {},
        "fast": dict(bg_iter=2, window_sizes=(1, 4, 16)),
        "conservative": dict(base_threshold_sigma=9.0, suppression_ramp=3.0),
    }.get(method, {})

    result = clean(data, time_win=time_win, base_threshold_sigma=12.0, burst_sigma=2.0,
                    known_burst_weight=known_burst_weight, **method_params)
    result["raw"] = data
    result["burst_indices"] = indices
    return result


def clean_all_events(out_dir: str, method: str = "comprehensive") -> dict[str, dict]:
    """Run clean_event_file on every row in out_dir/metadata.csv. Returns
    {file_name: result_dict}."""
    meta = pd.read_csv(os.path.join(out_dir, "metadata.csv"), dtype=str)
    results = {}
    for _, row in meta.iterrows():
        npy_path = os.path.join(out_dir, row["file_name"])
        if not os.path.exists(npy_path):
            print(f"  [clean_all_events] missing {npy_path}, skipping")
            continue
        results[row["file_name"]] = clean_event_file(npy_path, row, method=method)
    return results
