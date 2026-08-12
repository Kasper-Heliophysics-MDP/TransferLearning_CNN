"""
Own-station equivalent of ecallisto_grabber/extract_events.py: denoise a
whole file once, then crop out one native-resolution .npy per cataloged
burst in that file -- NO fixed-size fine-slicing, NO resize.

Why this exists instead of reusing BurstFixedWindowSlicer.slice_burst_with_fixed_windows:
that method's final step (resize_window) forces every window to (128,128)
via cv2.resize on BOTH axes. Own station covers ~16-24MHz; eCallisto stations
cover very different, often non-overlapping bands (e.g. AUSTRIA-UNIGRAZ
45-81MHz per ecallisto_grabber/README.md). Resizing each source independently
before those bands are aligned bakes in a source-dependent pixels-per-MHz
scale, so the same real drift rate ends up looking like a different slope
depending on which station a sample came from -- exactly the problem
ecallisto_grabber/extract_events.py already avoids for eCallisto ("No
resizing... that alignment step isn't built yet"). This applies the same
"denoise + crop only, resize later once there's a shared physical grid"
policy to own-station, so both sources are ready to be unified the same way
once that alignment step is designed.

Metadata schema matches ecallisto_grabber/extract_events.py's
METADATA_COLUMNS so the two can eventually be concatenated. freq_mhz is kept
in the CSV's native column order (descending, ~24->16MHz) and saved
alongside each file's crops rather than normalized to any convention --
eCallisto's own freq_mhz isn't normalized across stations either (fetch.py
just reads whatever order each station's FITS table provides), so imposing
one here would be guessing at a convention the "unify" step hasn't decided
yet.

Efficiency note: denoises each file ONCE and crops every burst in it from
that single result, instead of the previous per-burst pipeline which
re-denoised the same file once per burst row (12x redundant work for the
Marquette file, which has 12 catalog rows).
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from slicing_utils_new import time_to_column_indices
# slicing_utils_new's import above appends ecallisto_grabber/denoising to
# sys.path as a side effect -- see that module's header -- so this resolves
# without repeating the same sys.path setup here.
from sumthreshold_denoise import clean, _REFERENCE_TIME_WIN, _REFERENCE_SAMPLE_INTERVAL_S

SAMPLE_INTERVAL_S = 0.1  # own station's fixed CSV sampling interval (matches BurstFixedWindowSlicer)
DEFAULT_KNOWN_BURST_MARGIN = 10  # matches sumthreshold_cleaning_wrapper's default

# mirrors sumthreshold_denoise._METHOD_PARAMS (same values clean_ecallisto_events.py
# duplicates rather than importing the underscore-prefixed original)
_METHOD_PARAMS = {
    "comprehensive": {},
    "fast": dict(bg_iter=2, window_sizes=(1, 4, 16)),
    "conservative": dict(base_threshold_sigma=9.0, suppression_ramp=3.0),
}

METADATA_COLUMNS = [
    "file_name", "date", "location", "start_time", "end_time", "type",
    "event_start_time", "event_end_time", "other_stations", "uncertain",
    "freq_min_mhz", "freq_max_mhz", "n_freq_channels", "sample_interval_s",
]


def leading_timestamp(name: str) -> str | None:
    m = re.match(r"^(\d+)-", os.path.basename(name))
    return m.group(1) if m else None


def resolve_csv_path(file_name: str, csv_dir: str) -> str | None:
    """Same typo-tolerant match as process_one_burst.py (catalog says
    "PeachMountain.csv", disk has "PeachMountian.csv") -- match on the
    leading numeric timestamp instead of the full name."""
    direct = os.path.join(csv_dir, file_name)
    if os.path.exists(direct):
        return direct
    target = leading_timestamp(file_name)
    if target is None:
        return None
    for candidate in os.listdir(csv_dir):
        if leading_timestamp(candidate) == target:
            return os.path.join(csv_dir, candidate)
    return None


def extract_events_for_csv(
    csv_path: str,
    catalog_rows: pd.DataFrame,
    out_dir: str,
    buffer_s: float = 60.0,
    method: str = "fast",
) -> pd.DataFrame:
    """Denoise csv_path once, crop every burst in catalog_rows out of it.

    BUG FIXED (found via event_review -- 5/6 first real reviews flagged "main
    burst erased with white blank"): this function used to call
    BurstFixedWindowSlicer.load_and_preprocess_csv(csv_path, apply_denoising=True,
    cleaning_method=method) WITHOUT burst_start_time/burst_end_time -- the
    catalog's burst indices were computed and used only afterward, for
    cropping, never passed in as known_burst_weight. So every burst in every
    one of the 74 originally-generated rough_events files was denoised with
    ZERO ground-truth protection, relying entirely on burst_weight_2d's
    auto-detection -- which this project has repeatedly confirmed is
    unreliable across stations/events (blind-test protection ratio measured
    20%-74%). Now builds one known_burst_weight covering EVERY cataloged
    burst in this file (not just the one currently being cropped) before the
    single per-file clean() call, so the "denoise once per file" efficiency
    win doesn't come at the cost of ground-truth protection.
    """
    os.makedirs(out_dir, exist_ok=True)

    raw = pd.read_csv(csv_path, on_bad_lines="skip")
    times = raw["Time"]
    freq_hz = raw.columns[2:].to_numpy(dtype=float)
    freq_mhz = freq_hz / 1e6
    data = raw.iloc[:, 2:].to_numpy(dtype=np.float32).T  # (time, freq) -> (freq, time)
    del raw
    gc.collect()

    n_time = data.shape[1]

    known = np.zeros(n_time, dtype=np.float32)
    any_located = False
    for _, ev in catalog_rows.iterrows():
        try:
            s_idx, e_idx = time_to_column_indices(times, ev["start_time"], ev["end_time"])
        except Exception as e:
            print(f"    [extract] could not locate burst time {ev['start_time']}-{ev['end_time']} for protection: {e}")
            continue
        any_located = True
        lo = max(0, s_idx - DEFAULT_KNOWN_BURST_MARGIN)
        hi = min(n_time, e_idx + DEFAULT_KNOWN_BURST_MARGIN)
        known[lo:hi] = 1.0
    known_burst_weight = np.broadcast_to(known, data.shape) if any_located else None

    time_win = round(_REFERENCE_TIME_WIN * _REFERENCE_SAMPLE_INTERVAL_S / SAMPLE_INTERVAL_S)
    time_win = max(time_win // 2 * 2 + 1, 3)
    params = dict(time_win=time_win, base_threshold_sigma=12.0, burst_sigma=2.0)
    params.update(_METHOD_PARAMS.get(method, {}))

    result = clean(data, known_burst_weight=known_burst_weight, return_intermediates=False, **params)
    cleaned = result["cleaned"]  # (freq, time), matches extract_events.py's orientation

    buffer_samples = int(round(buffer_s / SAMPLE_INTERVAL_S))

    rows = []
    for _, ev in catalog_rows.iterrows():
        try:
            event_start_idx, event_end_idx = time_to_column_indices(times, ev["start_time"], ev["end_time"])
        except Exception as e:
            print(f"    [extract] could not locate burst time {ev['start_time']}-{ev['end_time']}: {e}")
            continue

        col_start = max(0, event_start_idx - buffer_samples)
        col_end = min(n_time, event_end_idx + buffer_samples)
        if col_end <= col_start:
            continue
        crop = cleaned[:, col_start:col_end]

        safe_type = re.sub(r"[^A-Za-z0-9]+", "", str(ev["type"])) or "unk"
        event_start_time = times.iloc[event_start_idx]
        event_end_time = times.iloc[min(event_end_idx, len(times) - 1)]
        win_start_time = times.iloc[col_start]
        win_end_time = times.iloc[col_end - 1]  # col_end is a slice upper bound (exclusive); last included sample is col_end-1
        fname = (
            f"burst-{ev['location']}-{ev['date']}-"
            f"{str(event_start_time).replace(':', '').replace('.', '')}-type{safe_type}.npy"
        )
        if os.path.exists(os.path.join(out_dir, fname)):
            print(f"    [extract] {fname} already exists (duplicate catalog row?), skipping")
            continue
        np.save(os.path.join(out_dir, fname), crop)

        rows.append({
            "file_name": fname,
            "date": ev["date"],
            "location": ev["location"],
            "start_time": str(win_start_time),
            "end_time": str(win_end_time),
            "type": ev["type"],
            "event_start_time": str(event_start_time),
            "event_end_time": str(event_end_time),
            "other_stations": "",
            "uncertain": False,
            "freq_min_mhz": float(freq_mhz.min()),
            "freq_max_mhz": float(freq_mhz.max()),
            "n_freq_channels": len(freq_mhz),
            "sample_interval_s": SAMPLE_INTERVAL_S,
        })

    if rows:
        os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        np.save(os.path.join(out_dir, "meta", f"freq-{base}.npy"), freq_mhz)

    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


def append_metadata(new_rows: pd.DataFrame, metadata_csv: str) -> None:
    write_header = not os.path.exists(metadata_csv)
    new_rows.to_csv(metadata_csv, mode="a", header=write_header, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--file-index", type=int, required=True,
                     help="index into the sorted list of unique file_name values in the catalog")
    ap.add_argument("--buffer-s", type=float, default=60.0)
    ap.add_argument("--method", default="fast", choices=["fast", "comprehensive", "conservative"])
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    unique_files = sorted(df["file_name"].unique())
    if not (0 <= args.file_index < len(unique_files)):
        print(f"RESULT file_index={args.file_index} status=ERROR reason=index_out_of_range total={len(unique_files)}")
        sys.exit(1)
    target_name = unique_files[args.file_index]
    catalog_rows = df[df["file_name"] == target_name]

    csv_path = resolve_csv_path(target_name, args.csv_dir)
    if csv_path is None:
        print(f"RESULT file_index={args.file_index} status=ERROR reason=file_not_found file_name={target_name!r}")
        sys.exit(1)

    t0 = time.time()
    try:
        new_rows = extract_events_for_csv(
            csv_path, catalog_rows, args.out_dir, buffer_s=args.buffer_s, method=args.method
        )
    except Exception as e:
        print(f"RESULT file_index={args.file_index} status=ERROR reason=exception file={os.path.basename(csv_path)} error={e!r}")
        sys.exit(1)

    append_metadata(new_rows, os.path.join(args.out_dir, "metadata.csv"))
    dt = time.time() - t0
    print(f"RESULT file_index={args.file_index} status=OK file={os.path.basename(csv_path)} "
          f"bursts_in_file={len(catalog_rows)} events_extracted={len(new_rows)} seconds={dt:.1f}")


if __name__ == "__main__":
    main()
