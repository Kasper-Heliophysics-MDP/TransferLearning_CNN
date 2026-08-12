"""
refresh_boxes.py

Re-derive boxes.csv (and the box-dependent columns of windows.csv) for an
existing scrape_windows.py output directory, WITHOUT re-downloading anything.

Why this has to exist: boxes.csv is written during scraping, so it freezes the
review state at that moment. The first full Arecibo scrape baked in a snapshot
of 272 reviewed events / 132 manual time corrections; by the time the reviewer
had worked through 512 events / 256 corrections, none of the newer work was in
the dataset. Without this script the only way to pick those up would be another
~9-hour re-scrape of data already sitting on disk, which would make continued
reviewing pointless.

Nothing here touches the window .npy files -- only the catalog (a few small
monthly text files) is re-fetched, and every box is recomputed through the same
extract_windows.boxes_for_window() the scraper itself uses, so the two paths
cannot disagree.

Usage:
    python detection/refresh_boxes.py <window_dir> \
        [--types II III V] [--review-csv ...] [--old-metadata-csv ...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecallisto_grabber"))
from burst_catalog import explode_by_station, fetch_catalog_range, filter_catalog  # noqa: E402
from extract_windows import (  # noqa: E402
    BOX_COLUMNS, WINDOW_COLUMNS, boxes_for_window, load_corrections, split_targets_offclass,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("window_dir")
    ap.add_argument("--types", nargs="*", default=["II", "III", "V"])
    ap.add_argument("--keep-uncertain", action="store_true")
    ap.add_argument("--review-csv", default="data/ecallisto/raw_events/review_status.csv")
    ap.add_argument("--old-metadata-csv", default="data/ecallisto/raw_events/metadata.csv")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    win_csv = os.path.join(args.window_dir, "windows.csv")
    box_csv = os.path.join(args.window_dir, "boxes.csv")
    windows = pd.read_csv(win_csv, dtype={"date": str})
    print(f"{len(windows)} windows in {win_csv}")

    if not args.no_backup:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        for src in (win_csv, box_csv):
            if os.path.exists(src):
                shutil.copy(src, f"{src}.bak-{stamp}")
        print(f"  backed up windows.csv / boxes.csv with suffix .bak-{stamp}")

    dates = sorted(windows.date.unique())
    start = datetime.strptime(dates[0], "%Y%m%d").date()
    end = datetime.strptime(dates[-1], "%Y%m%d").date()
    stations = sorted(windows.location.unique())
    print(f"catalog {start} .. {end} for {stations}")

    catalog = fetch_catalog_range(start, end)
    exploded = explode_by_station(catalog)
    filtered = filter_catalog(exploded, stations=stations, types=args.types,
                              drop_uncertain=not args.keep_uncertain)
    corrections = load_corrections(args.review_csv, args.old_metadata_csv)

    box_rows, win_rows = [], []
    for (station, date), grp in windows.groupby(["location", "date"]):
        targets, off_class = split_targets_offclass(
            filtered[(filtered.station == station) & (filtered.date == date)],
            exploded[(exploded.station == station) & (exploded.date == date)],
        )
        base = datetime.strptime(date, "%Y%m%d")
        for w in grp.itertuples():
            win_start = datetime.combine(
                base.date(), datetime.strptime(w.win_start_time, "%H:%M:%S").time())
            # a window whose clock end is before its start ran across midnight
            win_end = win_start + timedelta(seconds=w.n_cols * w.sample_interval_s)
            boxes, excluded, reason, off = boxes_for_window(
                w.file_name, station, date, win_start, win_end,
                int(w.n_cols), float(w.sample_interval_s), targets, off_class, corrections,
            )
            box_rows.extend(boxes)
            d = w._asdict(); d.pop("Index", None)
            d.update(n_boxes=len(boxes), excluded=excluded,
                     exclude_reason=reason, offclass_types=",".join(off))
            win_rows.append(d)

    new_windows = pd.DataFrame(win_rows, columns=WINDOW_COLUMNS)
    new_boxes = pd.DataFrame(box_rows, columns=BOX_COLUMNS)

    old_boxes = pd.read_csv(box_csv, dtype={"date": str}) if os.path.exists(box_csv) else pd.DataFrame()
    print(f"\nboxes   {len(old_boxes):5d} -> {len(new_boxes):5d}")
    if len(old_boxes):
        o = old_boxes.review_status.fillna("").ne("").sum()
        print(f"  with a review status  {o:5d} -> {new_boxes.review_status.fillna('').ne('').sum():5d}")
        print(f"  time_source=manual    {(old_boxes.time_source == 'manual').sum():5d} -> "
              f"{(new_boxes.time_source == 'manual').sum():5d}")
    print(f"excluded windows {int(windows.excluded.sum()):5d} -> {int(new_windows.excluded.sum()):5d}")
    print(new_windows[new_windows.excluded].exclude_reason.value_counts().to_string())

    new_windows.to_csv(win_csv, index=False)
    new_boxes.to_csv(box_csv, index=False)
    print(f"\nrewrote {win_csv} and {box_csv}")


if __name__ == "__main__":
    main()
