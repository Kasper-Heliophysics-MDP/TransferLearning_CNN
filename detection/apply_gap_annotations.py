"""
apply_gap_annotations.py

Turn the hand-annotated catalog-gap subset (data/ecallisto/gap_annotate/, built
by event_review/make_gap_subset.py and reviewed in the app) into training boxes.

WHY THESE GO IN A SEPARATE FILE AND NOT boxes.csv
-------------------------------------------------
refresh_boxes.py does not append to boxes.csv -- it REBUILDS it from the
Monstein catalog. A gap box is by definition absent from that catalog, so
anything written into boxes.csv is silently erased the next time anyone
refreshes, and refreshing is already on the roadmap (there are ~90 reviews still
waiting to be absorbed). These therefore live in gap_boxes.csv, which no
catalog rebuild touches, and build_yolo_dataset.py merges the two.

WHAT COMES OUT, AND HOW HONEST EACH ROW IS
------------------------------------------
`time_source` records provenance per row, because the two halves are not equally
trustworthy and the difference is invisible afterwards otherwise:

  gap_manual  the reviewer dragged the box themselves
  gap_model   the reviewer kept the detector's box unchanged (bit-exact)

The second kind is the detector's own output re-entering training along its
weakest axis (box extent: AP@0.75 0.16). The reviewer inspected these and
judged the boxes correct, and the evidence supports that reading -- the
keep-rate falls monotonically with confidence, 66% in the 0.10-0.15 band down
to 0% above 0.35, i.e. the boxes that were kept are the ones the detector drew
well. Both kinds are included. The label exists so that if a later result looks
strange, the question "was it the model-derived boxes?" can actually be asked
instead of guessed at.

SIDE EFFECT: WINDOWS COME BACK FROM THE DEAD
--------------------------------------------
63 of the 90 windows in contaminated_negatives.csv are windows these gaps sit
in. They were being dropped from training entirely, as negatives known to hide a
burst. Once the burst is labelled they are no longer contaminated negatives --
they are ordinary positive windows -- so they are removed from that list and
rejoin training with real boxes. That turns "throw 90 windows away" into "throw
27 away, recover 63 as labelled positives".

Usage:
    python detection/apply_gap_annotations.py                      # dry run
    python detection/apply_gap_annotations.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timedelta

import pandas as pd

BOX_COLUMNS = ["file_name", "date", "location", "type", "event_start_time", "event_end_time",
               "box_start_time", "box_end_time", "box_start_col", "box_end_col",
               "time_source", "review_status", "uncertain", "clipped"]
BACKUP_DIR = "event_review/backups"


def hms(base: str, seconds: float) -> str:
    t = datetime.strptime(str(base), "%H:%M:%S") + timedelta(seconds=float(seconds))
    return t.strftime("%H:%M:%S.%f")[:-5] if t.microsecond else t.strftime("%H:%M:%S")


def parse_hms(s: str) -> float:
    s = str(s)
    fmt = "%H:%M:%S.%f" if "." in s else "%H:%M:%S"
    t = datetime.strptime(s, fmt)
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-dir", default="data/ecallisto/gap_annotate")
    ap.add_argument("--window-dir", default="data/ecallisto/windows")
    ap.add_argument("--out", default=None, help="default: <window-dir>/gap_boxes.csv")
    ap.add_argument("--default-type", default="III")
    ap.add_argument("--type-override", nargs="*", default=[],
                    help="entries whose class is not the default, as <file_name_prefix>=<TYPE>, "
                         "e.g. gap0067=II")
    ap.add_argument("--neg-csv", default="data/ecallisto/windows/contaminated_negatives.csv")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()
    out_csv = args.out or os.path.join(args.window_dir, "gap_boxes.csv")

    meta = pd.read_csv(os.path.join(args.subset_dir, "metadata.csv"), dtype=str)
    rs = pd.read_csv(os.path.join(args.subset_dir, "review_status.csv"))
    win = pd.read_csv(os.path.join(args.window_dir, "windows.csv"),
                      dtype={"date": str}).set_index("file_name")

    overrides = {}
    for spec in args.type_override:
        k, _, v = spec.partition("=")
        overrides[k] = v
    unmatched = [k for k in overrides
                 if not meta.file_name.str.startswith(k).any()]
    if unmatched:
        raise SystemExit(f"--type-override prefixes match nothing: {unmatched}")

    status = rs.set_index("file_name")
    rows, n_manual, n_model, skipped = [], 0, 0, 0
    for m in meta.itertuples():
        if m.file_name not in status.index:
            skipped += 1
            continue
        st = status.loc[m.file_name]
        if str(st.get("status", "")) != "usable":
            skipped += 1
            continue

        raw_range = st.get("manual_burst_range_json")
        if pd.notna(raw_range) and str(raw_range).strip():
            r = json.loads(raw_range)
            start_s, end_s = float(r["start_s"]), float(r["end_s"])
            src = "gap_manual"
            n_manual += 1
        else:
            # kept the prefill exactly; recover it from the metadata times,
            # which is where the prefill came from in the first place
            start_s = parse_hms(m.event_start_time) - parse_hms(m.start_time)
            end_s = parse_hms(m.event_end_time) - parse_hms(m.start_time)
            if start_s < 0:                      # window ran across midnight
                start_s += 86400
            if end_s < 0:
                end_s += 86400
            src = "gap_model"
            n_model += 1

        # strip the gapNNNN- prefix to get back to the real window array
        window_file = m.file_name.split("-", 1)[1]
        w = win.loc[window_file]
        dt = float(w.sample_interval_s)
        n_cols = int(w.n_cols)
        c0, c1 = round(start_s / dt), round(end_s / dt)
        clipped = c0 < 0 or c1 > n_cols
        c0, c1 = max(0, c0), min(n_cols, c1)
        if c1 <= c0:
            print(f"  !! {m.file_name}: degenerate box after clamping ({c0},{c1}), skipped")
            skipped += 1
            continue

        btype = args.default_type
        for k, v in overrides.items():
            if m.file_name.startswith(k):
                btype = v

        rows.append({
            "file_name": window_file, "date": w.date, "location": w.location, "type": btype,
            "event_start_time": hms(w.win_start_time, start_s),
            "event_end_time": hms(w.win_start_time, end_s),
            "box_start_time": hms(w.win_start_time, c0 * dt),
            "box_end_time": hms(w.win_start_time, c1 * dt),
            "box_start_col": c0, "box_end_col": c1,
            "time_source": src, "review_status": "usable",
            "uncertain": False, "clipped": clipped,
        })

    box = pd.DataFrame(rows, columns=BOX_COLUMNS)
    print(f"{len(meta)} subset entries -> {len(box)} box(es)  "
          f"(skipped {skipped})")
    print(f"  time_source: gap_manual {n_manual}  gap_model {n_model}")
    print(f"  types: {box.type.value_counts().to_dict()}")
    print(f"  distinct windows: {box.file_name.nunique()}   clipped: {int(box.clipped.sum())}")
    print(f"  box width s: median {((box.box_end_col-box.box_start_col)*0.25).median():.0f}")

    neg = pd.read_csv(args.neg_csv) if os.path.exists(args.neg_csv) else pd.DataFrame()
    freed = sorted(set(neg.get("file_name", [])) & set(box.file_name))
    print(f"\ncontaminated negatives: {len(neg)} -> {len(neg) - len(freed)} "
          f"({len(freed)} now carry real boxes and rejoin training as positives)")

    if not args.apply:
        print(f"\nwould write {out_csv}")
        print("DRY RUN -- nothing written. Re-run with --apply.")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for path in (out_csv, args.neg_csv):
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(BACKUP_DIR,
                                            f"{os.path.basename(path)}.{stamp}.bak"))
    box.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}: {len(box)} row(s)")
    if freed:
        neg[~neg.file_name.isin(freed)].to_csv(args.neg_csv, index=False)
        print(f"wrote {args.neg_csv}: {len(neg)} -> {len(neg) - len(freed)} row(s)")
    print("\nNOTE: boxes.csv is untouched -- refresh_boxes.py rebuilds it from the catalog "
          "and would erase these.\n      build_yolo_dataset.py merges gap_boxes.csv in.")


if __name__ == "__main__":
    main()
