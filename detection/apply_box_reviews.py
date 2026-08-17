"""
apply_box_reviews.py

Fold reviews done in a make_box_subset.py directory back into that window
directory's boxes.csv: hand-corrected times become the box extent, and the
review verdict is recorded so build_yolo_dataset can tell clean boxes from the
rest.

Distinct from the gap path (apply_gap_annotations.py -> gap_boxes.csv). These
are CATALOG boxes, so they already live in boxes.csv and are updated in place.
That also means refresh_boxes.py can regenerate them -- it re-reads the catalog
and re-applies corrections from a review CSV -- whereas a gap box has no
catalog row to regenerate from and must be kept outside.

MAPPING BACK IS DONE BY POSITION, NOT BY THE FILENAME COUNTER
------------------------------------------------------------
Subset entries are named with a running counter (ii0007-...), which does not
identify which box in a window it was when a window holds two. The metadata's
event_start_time is the box's original start, exact to the column (verified at
build time), so (window, original start column) is used as the key instead.

Safety, matching the other write-back scripts: dry run by default, timestamped
backup before any write, and only rows that were actually reviewed are touched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timedelta

import pandas as pd

BACKUP_DIR = "event_review/backups"


def parse_hms(s: str) -> float:
    s = str(s)
    t = datetime.strptime(s, "%H:%M:%S.%f" if "." in s else "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def hms(base: str, seconds: float) -> str:
    t = datetime.strptime(str(base), "%H:%M:%S") + timedelta(seconds=float(seconds))
    return t.strftime("%H:%M:%S.%f")[:-5] if t.microsecond else t.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir")
    ap.add_argument("window_dir")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.subset_dir, "metadata.csv"), dtype=str)
    rs = pd.read_csv(os.path.join(args.subset_dir, "review_status.csv"))
    box_csv = os.path.join(args.window_dir, "boxes.csv")
    box = pd.read_csv(box_csv, dtype={"date": str})
    win = pd.read_csv(os.path.join(args.window_dir, "windows.csv"),
                      dtype={"date": str}).set_index("file_name")

    status = rs.set_index("file_name")
    # (window, original start column) -> row index in boxes.csv
    key = {(r.file_name, int(r.box_start_col)): i for i, r in box.iterrows()}

    n_status, n_time, missing, unreviewed = 0, 0, 0, 0
    for m in meta.itertuples():
        if m.file_name not in status.index:
            unreviewed += 1
            continue
        st = status.loc[m.file_name]
        if not str(st.get("status", "")).strip():
            unreviewed += 1
            continue

        window_file = m.file_name.split("-", 1)[1]
        w = win.loc[window_file]
        dt = float(w.sample_interval_s)
        orig_col = round((parse_hms(m.event_start_time) - parse_hms(m.start_time)) / dt)
        idx = key.get((window_file, orig_col))
        if idx is None:
            print(f"  !! no boxes.csv row for {window_file} at column {orig_col}")
            missing += 1
            continue

        box.at[idx, "review_status"] = st["status"]
        n_status += 1

        raw = st.get("manual_burst_range_json")
        if pd.notna(raw) and str(raw).strip():
            r = json.loads(raw)
            c0 = max(0, round(float(r["start_s"]) / dt))
            c1 = min(int(w.n_cols), round(float(r["end_s"]) / dt))
            if c1 <= c0:
                print(f"  !! {m.file_name}: degenerate range after clamping, time not applied")
                continue
            box.at[idx, "box_start_col"] = c0
            box.at[idx, "box_end_col"] = c1
            box.at[idx, "box_start_time"] = hms(w.win_start_time, c0 * dt)
            box.at[idx, "box_end_time"] = hms(w.win_start_time, c1 * dt)
            box.at[idx, "time_source"] = "manual"
            n_time += 1

    print(f"{len(meta)} subset entries: {n_status} verdicts applied, "
          f"{n_time} with a hand-corrected time")
    print(f"  unreviewed {unreviewed}, unmatched {missing}")
    print(f"  boxes.csv review_status now: {box.review_status.value_counts(dropna=False).to_dict()}")
    print(f"  time_source now: {box.time_source.value_counts().to_dict()}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    shutil.copy2(box_csv, os.path.join(BACKUP_DIR, f"boxes_assa_{stamp}.csv"))
    box.to_csv(box_csv, index=False)
    print(f"\nwrote {box_csv} (backup in {BACKUP_DIR})")


if __name__ == "__main__":
    main()
