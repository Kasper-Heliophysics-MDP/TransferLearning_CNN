"""
CLI: process exactly one burst_list.csv row through BurstFixedWindowSlicer
and exit.

Why one-row-per-process instead of looping over the whole catalog in one
Python process: real own-station files run up to ~1.3GB/500k rows, and even
with the memory fixes in slicing_utils_new.py + sumthreshold_denoise.py, a
single file's peak sits in the several-GB range. Looping in-process relies on
Python's GC to fully release each file's arrays before the next one starts,
which HANDOFF.md flags as already having caused OOM on this 15GB no-swap
machine during earlier own-station validation. Running one burst per OS
process (invoked in a shell loop) makes the OS reclaim everything between
files regardless of GC timing -- see batch_process_own_station.sh.

Usage:
    python process_one_burst.py --catalog <burst_list.csv> --csv-dir <dir>
        --out-dir <dir> --index <row index into the catalog> [--method fast]
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

from slicing_utils_new import BurstFixedWindowSlicer


def leading_timestamp(name: str) -> str | None:
    m = re.match(r"^(\d+)-", os.path.basename(name))
    return m.group(1) if m else None


def resolve_csv_path(file_name: str, csv_dir: str) -> str | None:
    """Match a catalog file_name against what's actually on disk.

    The catalog and the on-disk files disagree on spelling for at least one
    entry (catalog: "PeachMountain.csv", disk: "PeachMountian.csv" -- a
    genuine transposed-letter typo, not just whitespace/case, confirmed by
    diffing catalog file_name values against os.listdir()). A
    normalize-and-compare fallback doesn't fix a real typo, so match on the
    leading numeric timestamp instead (verified unique across all 48
    on-disk files) and ignore the location name entirely.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--method", default="fast", choices=["fast", "comprehensive", "conservative"])
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    if not (0 <= args.index < len(df)):
        print(f"RESULT index={args.index} status=ERROR reason=index_out_of_range total={len(df)}")
        sys.exit(1)
    row = df.iloc[args.index]

    csv_path = resolve_csv_path(row["file_name"], args.csv_dir)
    if csv_path is None:
        print(f"RESULT index={args.index} status=ERROR reason=file_not_found file_name={row['file_name']!r}")
        sys.exit(1)

    burst_type = int(row["type"])
    out_dir = os.path.join(args.out_dir, f"type_{burst_type}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    try:
        slicer = BurstFixedWindowSlicer(window_duration=4 * 60, overlap_ratio=0.5, target_size=(128, 128))
        result = slicer.slice_burst_with_fixed_windows(
            csv_file_path=csv_path,
            burst_start_time=row["start_time"],
            burst_end_time=row["end_time"],
            save_dir=out_dir,
            apply_denoising=True,
            burst_type=burst_type,
            cleaning_method=args.method,
        )
    except Exception as e:
        print(f"RESULT index={args.index} status=ERROR reason=exception file={os.path.basename(csv_path)} error={e!r}")
        sys.exit(1)

    dt = time.time() - t0
    print(f"RESULT index={args.index} status=OK file={os.path.basename(csv_path)} "
          f"type={burst_type} windows={len(result['windows'])} seconds={dt:.1f}")


if __name__ == "__main__":
    main()
