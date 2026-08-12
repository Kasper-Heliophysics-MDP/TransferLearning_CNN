"""
scrape_windows.py

Driver for extract_windows.py -- the detection-training data path. Same
catalog -> fetch -> save loop as scrape.py, but saves whole 15-minute FITS
windows (plus boxes) instead of per-event crops. See extract_windows.py's
docstring for why that distinction matters, and TRAINING_PLAN.md for the plan
this feeds.

Outputs into --out-dir:
    win-<station>-<YYYYMMDD>-<HHMMSS>.npy   one per kept 15-minute window
    meta/freq-<station>-<YYYYMMDD>.npy      frequency axis per station-day
    windows.csv                             one row per window
    boxes.csv                               one row per burst box

Resumable: skips (station, date) pairs already present in windows.csv.

Usage:
    # smoke test: one Arecibo station-day
    python scrape_windows.py --start 2021-10-10 --end 2021-10-10 \
        --stations Arecibo-Observatory --types II III V \
        --out-dir /tmp/win_test --limit 1

    # the real Phase 1 run (~9 hours, ~6.6 GB)
    python scrape_windows.py --start 2021-01-01 --end 2024-03-31 \
        --stations Arecibo-Observatory --types II III V \
        --out-dir ../data/ecallisto/windows
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from burst_catalog import explode_by_station, fetch_catalog_range, filter_catalog
from extract_windows import append_csv, extract_windows_for_day, load_corrections
from fetch import fetch_station_day

DEFAULT_REVIEW_CSV = "../data/ecallisto/raw_events/review_status.csv"
DEFAULT_OLD_METADATA_CSV = "../data/ecallisto/raw_events/metadata.csv"


def _already_done(windows_csv: str) -> set[tuple[str, str]]:
    if not os.path.exists(windows_csv):
        return set()
    df = pd.read_csv(windows_csv, usecols=["location", "date"], dtype=str)
    return set(zip(df["location"], df["date"]))


def run(
    start: date,
    end: date,
    stations: list[str] | None,
    types: list[str] | None,
    out_dir: str,
    drop_uncertain: bool,
    context_slots: int,
    limit: int | None,
    review_csv: str,
    old_metadata_csv: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    windows_csv = os.path.join(out_dir, "windows.csv")
    boxes_csv = os.path.join(out_dir, "boxes.csv")

    print(f"Fetching Monstein catalog {start} .. {end} ...")
    catalog = fetch_catalog_range(start, end)
    # ALL types, unfiltered -- needed to spot windows holding a burst we can't
    # label (Type IV etc.), which are neither positives nor clean negatives
    exploded = explode_by_station(catalog)
    filtered = filter_catalog(exploded, stations=stations, types=types, drop_uncertain=drop_uncertain)
    print(f"{len(filtered)} (station, event) rows after filtering "
          f"(stations={stations or 'ALL'}, types={types or 'ALL'})")

    corrections = load_corrections(review_csv, old_metadata_csv)

    done = _already_done(windows_csv)
    groups = list(filtered.groupby(["station", "date"]))
    print(f"{len(groups)} distinct (station, date) pairs to fetch; {len(done)} already done")

    n_run, day_times, n_win, n_box = 0, [], 0, 0
    for (station, date_str), rows in groups:
        if (station, date_str) in done:
            continue
        if limit is not None and n_run >= limit:
            print(f"\nReached --limit {limit}, stopping (resume later to continue).")
            break

        y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
        print(f"\n[{n_run + 1}] {station} {date_str} -- {len(rows)} cataloged event(s)")

        t0 = time.monotonic()
        station_day = fetch_station_day(station, y, m, d)
        if station_day is None:
            n_run += 1
            continue

        all_day = exploded[(exploded["station"] == station) & (exploded["date"] == date_str)]
        win_meta, box_meta = extract_windows_for_day(
            station_day, rows, all_day, out_dir,
            corrections=corrections, context_slots=context_slots,
        )
        append_csv(win_meta, windows_csv)
        append_csv(box_meta, boxes_csv)

        elapsed = time.monotonic() - t0
        day_times.append(elapsed)
        n_win += len(win_meta)
        n_box += len(box_meta)
        n_excl = int(win_meta["excluded"].sum()) if not win_meta.empty else 0
        print(f"  -> {len(win_meta)} window(s) ({n_excl} excluded) + {len(box_meta)} box(es) "
              f"in {elapsed:.1f}s ({len(station_day.files)} files downloaded)")
        n_run += 1

    if day_times:
        print(f"\nTiming: n={len(day_times)}, mean={np.mean(day_times):.1f}s, "
              f"median={np.median(day_times):.1f}s per station-day")
        print(f"Saved {n_win} windows / {n_box} boxes this run")
        remaining = len(groups) - len(done) - len(day_times)
        if remaining > 0:
            print(f"Extrapolated for the {remaining} remaining station-days: "
                  f"~{np.mean(day_times) * remaining / 3600:.1f} hours")
    print(f"\nDone. {windows_csv} / {boxes_csv}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--stations", nargs="*", default=None)
    p.add_argument("--types", nargs="*", default=None, help="burst type codes to box, e.g. II III V")
    p.add_argument("--out-dir", default="../data/ecallisto/windows")
    p.add_argument("--context-slots", type=int, default=2,
                   help="15-min slots kept on each side of an event (2 = +/-30 min)")
    p.add_argument("--keep-uncertain", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="max NEW station-days this run")
    p.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    p.add_argument("--old-metadata-csv", default=DEFAULT_OLD_METADATA_CSV)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start=datetime.strptime(args.start, "%Y-%m-%d").date(),
        end=datetime.strptime(args.end, "%Y-%m-%d").date(),
        stations=args.stations,
        types=args.types,
        out_dir=args.out_dir,
        drop_uncertain=not args.keep_uncertain,
        context_slots=args.context_slots,
        limit=args.limit,
        review_csv=args.review_csv,
        old_metadata_csv=args.old_metadata_csv,
    )
