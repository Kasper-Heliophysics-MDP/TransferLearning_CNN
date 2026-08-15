"""
make_gap_subset.py

Turn confirmed catalog-gap candidates (detection/gap_review/to_annotate.csv)
into an event_review subset directory, so the burst times can be drawn by hand
instead of inherited from the model.

WHY THE TIMES MUST BE REDRAWN AT ALL
------------------------------------
mine_catalog_gaps.py already produced a start_s/end_s for every candidate. They
are NOT usable as training boxes: box extent is the detector's weakest axis
(AP@0.75 0.16), and maxconf keeps the most CONFIDENT box in a cluster, not the
best-placed one. Writing those coordinates back would train the model on its own
output along the one dimension already known to be bad. They go in here as the
PREFILL only -- the number the reviewer drags away from.

THE UNIT IS ONE CANDIDATE, NOT ONE WINDOW
-----------------------------------------
8 of the 117 windows carry 2-3 separate confirmed gaps, and
`manual_burst_range_json` stores a single {start_s, end_s}. One row per window
would silently drop the second burst in those windows. So each candidate gets
its own entry, `gap<id>-<window stem>.npy`, symlinked to the same underlying
window array. The reviewer sees such a window once per burst, each time with a
different prefilled range -- which is correct: they are separate annotations.

TYPE IS "gap", AND THAT IS DELIBERATE
-------------------------------------
These bursts are absent from the catalog, so their class is genuinely unknown --
writing "III" because it is the most common would be inventing data. "gap" is
inert in the two places event_review branches on type: gentle_params_for()
returns None (production defaults) and it is added to MINIMAL_VIEW_TYPES so the
panel shows per-row median subtraction and nothing else. That last part is the
important one: it is exactly the render the reviewer already judged these on in
the sheets (build_yolo_dataset.render_png does the same thing), and unlike
clean() it CANNOT erase a burst -- which matters here because clean()'s
protection window would be built from the model's guessed time, the precise
failure that made cleaned_events unusable for corrected events.

Usage:
    python event_review/make_gap_subset.py            # dry run
    python event_review/make_gap_subset.py --apply
    # then: streamlit run event_review/app.py
    #       sidebar -> eCallisto -> directory = data/ecallisto/gap_annotate
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

META_COLS = ["file_name", "date", "location", "start_time", "end_time", "type",
             "event_start_time", "event_end_time", "other_stations", "uncertain",
             "freq_min_mhz", "freq_max_mhz", "n_freq_channels", "sample_interval_s"]


def shift(hms: str, seconds: float) -> str:
    """HH:MM:SS(.f) + seconds, wrapping past midnight the same way the rest of
    the pipeline does (a 15-minute window starting at 23:52 legitimately ends on
    the next day, and event_burst_indices realigns on the same assumption).

    Sub-second precision is KEPT. Candidate times carry 0.1s, and formatting
    them as whole seconds moved the prefilled box by up to 4 columns at
    0.25s/sample -- measured: 112 of 126 entries landed 2-4 columns early.
    That is far below a human's own repeatability (sigma ~7.8s) and would never
    show up in a metric, which is exactly why it is worth removing: the prefill
    is the number a reviewer accepts without touching 38% of the time, so any
    systematic offset in it gets baked straight into the labels. Both formats
    are already handled by event_burst_indices, which picks its parse format on
    whether a '.' is present."""
    t = datetime.strptime(str(hms), "%H:%M:%S") + timedelta(seconds=float(seconds))
    return t.strftime("%H:%M:%S.%f")[:-5] if t.microsecond else t.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="detection/gap_review/to_annotate.csv")
    ap.add_argument("--window-dir", default="data/ecallisto/windows")
    ap.add_argument("--out-dir", default="data/ecallisto/gap_annotate")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    todo = pd.read_csv(args.verdicts)
    win = pd.read_csv(args.window_dir + "/windows.csv", dtype={"date": str}) \
            .set_index("file_name")

    missing = sorted(set(todo.file_name) - set(win.index))
    if missing:
        raise SystemExit(f"{len(missing)} window(s) in {args.verdicts} are not in windows.csv, "
                         f"e.g. {missing[:3]}")

    rows, links = [], []
    for c in todo.itertuples():
        w = win.loc[c.file_name]
        stem = c.file_name[:-len(".npy")]
        name = f"gap{int(c.id):04d}-{stem}.npy"
        links.append((name, os.path.abspath(os.path.join(args.window_dir, c.file_name))))
        rows.append({
            "file_name": name,
            "date": w.date,
            "location": w.location,
            # the crop IS the whole window here, so the event's offset inside it
            # is (event_start_time - start_time) -- exactly what
            # event_burst_indices reconstructs
            "start_time": w.win_start_time,
            "end_time": w.win_end_time,
            "type": "gap",
            "event_start_time": shift(w.win_start_time, c.start_s),
            "event_end_time": shift(w.win_start_time, c.end_s),
            "other_stations": "",
            "uncertain": False,
            "freq_min_mhz": w.freq_min_mhz,
            "freq_max_mhz": w.freq_max_mhz,
            "n_freq_channels": w.n_freq_channels,
            "sample_interval_s": w.sample_interval_s,
        })

    meta = pd.DataFrame(rows)[META_COLS]
    dupes = meta.file_name.duplicated().sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicate file_name(s) -- ids are not unique")

    freq_needed = sorted({(r["location"], r["date"]) for r in rows})
    print(f"{len(meta)} candidate(s) over {todo.file_name.nunique()} window(s)")
    print(f"  {len(freq_needed)} frequency-axis file(s) to link")
    print(f"  windows appearing more than once: "
          f"{int((todo.file_name.value_counts() > 1).sum())} "
          f"(each gets one entry per burst -- see module docstring)")
    if not args.apply:
        print(f"\nwould write {args.out_dir}/ : {len(links)} symlink(s) + metadata.csv")
        print("DRY RUN -- nothing written. Re-run with --apply.")
        return

    os.makedirs(os.path.join(args.out_dir, "meta"), exist_ok=True)
    for name, target in links:
        link = os.path.join(args.out_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(target, link)

    n_freq = 0
    for loc, date in freq_needed:
        src = os.path.abspath(os.path.join(args.window_dir, "meta", f"freq-{loc}-{date}.npy"))
        if not os.path.exists(src):
            print(f"  !! missing frequency axis {os.path.basename(src)} -- the panel will "
                  f"fall back to channel indices for that day")
            continue
        link = os.path.join(args.out_dir, "meta", os.path.basename(src))
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(src, link)
        n_freq += 1

    meta.to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    print(f"\nwrote {args.out_dir}/")
    print(f"  {len(links)} symlink(s), {n_freq} frequency axis file(s), metadata.csv")
    print(f"\nreview with:  streamlit run event_review/app.py")
    print(f"  sidebar -> eCallisto -> directory = {args.out_dir}")
    print("\nreview_status.csv is created by the app on first save.")


if __name__ == "__main__":
    main()
