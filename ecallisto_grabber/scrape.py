"""
scrape.py

End-to-end, no-clicking eCallisto scraper: catalog (with type) -> raw data -> cropped,
labeled .npy + metadata.csv, plus randomly-sampled negatives.

Blueprint: Kasper-Heliophysics-MDP/eCallisto-Burst-Grabber/data_collector.py, which
does the same job but drives locate_bursts.py, a GUI where a human clicks on
candidate windows to confirm each burst. This version has no manual-labeling step:
the Monstein catalog (burst_catalog.py) already carries verified time + type per
station, so every (station, date) pair with a cataloged event can be downloaded
(fetch.py) and cropped (extract_events.py) unattended.

Negatives: for each station-day, `--negatives-per-day` windows are sampled from
whatever part of that day is NOT within --buffer-s of ANY cataloged event (any
type, not just the ones being extracted as positives -- so an out-of-scope real
event can't get mislabeled as a negative). Durations are drawn from the positive
events' own duration distribution so width alone can't separate the classes.

Usage:
    python scrape.py --start 2024-03-01 --end 2024-03-07 --types II III V --limit 5

Resumable: skips (station, date) pairs already present in metadata.csv, so a run
can be Ctrl-C'd and restarted.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
import os

import numpy as np
import pandas as pd

from burst_catalog import explode_by_station, fetch_catalog_range, filter_catalog
from extract_events import (
    _parse_time_range,
    append_metadata,
    extract_events_for_day,
    sample_negative_windows,
)
from fetch import fetch_station_day


def _already_done(metadata_csv: str) -> set[tuple[str, str]]:
    if not os.path.exists(metadata_csv):
        return set()
    df = pd.read_csv(metadata_csv, usecols=["location", "date"], dtype=str)
    return set(zip(df["location"], df["date"]))


def _event_durations_s(rows: pd.DataFrame) -> list[float]:
    out = []
    for _, ev in rows.iterrows():
        try:
            s, e = _parse_time_range(ev["time_range"], ev["date"])
            out.append((e - s).total_seconds())
        except ValueError:
            continue
    return out


def run(
    start: date,
    end: date,
    stations: list[str] | None,
    types: list[str] | None,
    out_dir: str,
    buffer_s: float,
    drop_uncertain: bool,
    negatives_per_day: int,
    neg_seed: int,
    limit: int | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metadata_csv = os.path.join(out_dir, "metadata.csv")
    rng = np.random.default_rng(neg_seed)

    print(f"Fetching Monstein catalog {start} .. {end} ...")
    catalog = fetch_catalog_range(start, end)
    exploded = explode_by_station(catalog)  # ALL types -- needed to keep negatives clean
    filtered = filter_catalog(exploded, stations=stations, types=types, drop_uncertain=drop_uncertain)
    print(f"{len(filtered)} (station, event) rows after filtering "
          f"(stations={stations or 'ALL'}, types={types or 'ALL'})")

    duration_pool = _event_durations_s(filtered) or [60.0]
    print(f"duration pool for negatives: n={len(duration_pool)}, "
          f"median={np.median(duration_pool):.0f}s")

    done = _already_done(metadata_csv)
    groups = list(filtered.groupby(["station", "date"]))
    print(f"{len(groups)} distinct (station, date) pairs to fetch; {len(done)} already done")

    n_run = 0
    day_times = []
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

        pos_meta = extract_events_for_day(station_day, rows, out_dir, buffer_s=buffer_s)

        all_day_events = exploded[(exploded["station"] == station) & (exploded["date"] == date_str)]
        neg_meta = sample_negative_windows(
            station_day, all_day_events, out_dir,
            n_negatives=negatives_per_day, buffer_s=buffer_s,
            duration_pool_s=duration_pool, rng=rng,
        )

        new_meta = pd.concat([pos_meta, neg_meta], ignore_index=True)
        if not new_meta.empty:
            append_metadata(new_meta, metadata_csv)

        elapsed = time.monotonic() - t0
        day_times.append(elapsed)
        print(f"  -> {len(pos_meta)} positive + {len(neg_meta)} negative crop(s) "
              f"in {elapsed:.1f}s ({len(station_day.files)} files)")
        n_run += 1

    if day_times:
        print(f"\nTiming: n={len(day_times)}, mean={np.mean(day_times):.1f}s, "
              f"median={np.median(day_times):.1f}s, "
              f"min={min(day_times):.1f}s, max={max(day_times):.1f}s per station-day")
        if limit is not None and len(groups) > limit:
            est_total_s = np.mean(day_times) * (len(groups) - len(done))
            print(f"Extrapolated total for all {len(groups) - len(done)} remaining station-days: "
                  f"~{est_total_s/3600:.1f} hours (mean-time based; real run will vary by file "
                  f"count per station-day)")

    print(f"\nDone. Metadata: {metadata_csv}")


def count_station_days(
    start: date, end: date, stations: list[str] | None, types: list[str] | None, drop_uncertain: bool
) -> int:
    """Cheap (catalog-only, no FITS downloads) count of how many (station, date)
    pairs a given query would need to fetch. Useful before committing to a full run."""
    catalog = fetch_catalog_range(start, end)
    exploded = explode_by_station(catalog)
    filtered = filter_catalog(exploded, stations=stations, types=types, drop_uncertain=drop_uncertain)
    return filtered.groupby(["station", "date"]).ngroups


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--stations", nargs="*", default=None, help="station names to keep (default: all)")
    p.add_argument("--types", nargs="*", default=None, help="burst type codes to keep, e.g. II III V")
    p.add_argument("--out-dir", default="../data/ecallisto/raw_events")
    p.add_argument("--buffer-s", type=float, default=60.0, help="context seconds kept before/after each event")
    p.add_argument("--negatives-per-day", type=int, default=2,
                    help="random non-burst windows to sample per station-day")
    p.add_argument("--neg-seed", type=int, default=42)
    p.add_argument("--keep-uncertain", action="store_true",
                    help="also keep events where this station's detection was marked '(...)' as uncertain")
    p.add_argument("--limit", type=int, default=None, help="max number of NEW station-days to fetch this run")
    p.add_argument("--count-only", action="store_true",
                    help="just print how many (station, date) pairs this query matches, no downloads")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _start = datetime.strptime(args.start, "%Y-%m-%d").date()
    _end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.count_only:
        n = count_station_days(_start, _end, args.stations, args.types, not args.keep_uncertain)
        print(f"\n{n} distinct (station, date) pairs would be fetched.")
    else:
        run(
            start=_start,
            end=_end,
            stations=args.stations,
            types=args.types,
            out_dir=args.out_dir,
            buffer_s=args.buffer_s,
            drop_uncertain=not args.keep_uncertain,
            negatives_per_day=args.negatives_per_day,
            neg_seed=args.neg_seed,
            limit=args.limit,
        )
