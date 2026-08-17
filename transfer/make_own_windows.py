"""
make_own_windows.py

Cut real fixed 15-minute windows out of the own-station continuous CSVs, so the
e-Callisto-trained detector can be tested on own-station data under the same
input contract it was trained on.

WHY NOT REUSE data/burst_data/rough_events/
-------------------------------------------
Those are event-centred crops ("denoise + coarse crop around the catalog
time"). Feeding them to a detector reproduces the exact defect that invalidated
the first Phase 1 attempt: with the burst always at the centre, a detection rate
measured on them says nothing about a detector that has to find bursts at
arbitrary positions. The own station has full continuous recordings, so real
windows on a fixed grid cost only a little code and remove the problem entirely.

Windows are aligned to absolute 15-minute boundaries of the session clock, not
to the events, which is what makes the burst position vary.

WHAT THIS DOES NOT SOLVE
------------------------
The frequency mismatch. The own station covers 15.996-24.004 MHz in 411
channels (0.0195 MHz/channel); the training data covers 15-86.6 MHz in 200
channels (0.358 MHz/channel), 18x coarser. At the pixel scale the model was
trained on (0.112 MHz/px), the own station's entire 8 MHz band is 72 pixels of
a 640-pixel image. Deciding what to do about that is the experiment
(render_own.py), not something this script hides.

Usage:
    python transfer/make_own_windows.py                 # dry run
    python transfer/make_own_windows.py --apply
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "event_review"))
import data_access as da   # noqa: E402

WINDOW_S = 900.0
TYPE_MAP = {"2": "II", "3": "III", "5": "V"}


def to_dt(s: str) -> datetime:
    s = str(s)
    return datetime.strptime(s, "%H:%M:%S.%f" if "." in s else "%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/burst_data/rough_events")
    ap.add_argument("--out-dir", default="data/burst_data/own_windows")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.events, "metadata.csv"), dtype=str)
    rs_path = os.path.join(args.events, "review_status.csv")
    if os.path.exists(rs_path):
        rs = pd.read_csv(rs_path).set_index("file_name")
        keep = [f for f in meta.file_name
                if str(rs["status"].get(f, "")) == "usable"]
        meta = meta[meta.file_name.isin(keep)]
    print(f"{len(meta)} reviewed-usable own-station events")

    # group events by the 15-minute slot they fall in, so one window can carry
    # several boxes (same contract as the e-Callisto side)
    slots: dict[tuple, list] = {}
    unresolved = 0
    for r in meta.itertuples():
        try:
            csv_path = da.resolve_own_station_csv(r._asdict())
        except Exception as e:                       # noqa: BLE001
            print(f"  !! {r.file_name}: {e}")
            unresolved += 1
            continue
        ev0, ev1 = to_dt(r.event_start_time), to_dt(r.event_end_time)
        base = ev0.replace(hour=0, minute=0, second=0, microsecond=0)
        slot_i = int((ev0 - base).total_seconds() // WINDOW_S)
        slots.setdefault((csv_path, r.date, r.location, slot_i), []).append(
            (r.file_name, str(r.type), ev0, ev1))

    print(f"  -> {len(slots)} distinct 15-min slots ({unresolved} unresolved)")
    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    win_rows, box_rows, failed = [], [], 0
    for (csv_path, date, location, slot_i), evs in sorted(slots.items(), key=lambda k: str(k[0])):
        base = to_dt("00:00:00")
        w0 = base + timedelta(seconds=slot_i * WINDOW_S)
        w1 = w0 + timedelta(seconds=WINDOW_S)
        try:
            arr, freq = da._load_own_station_raw_slice(
                csv_path, w0.strftime("%H:%M:%S.%f"), w1.strftime("%H:%M:%S.%f"))
        except Exception as e:                       # noqa: BLE001
            print(f"  !! slice failed {os.path.basename(csv_path)} slot {slot_i}: {e}")
            failed += 1
            continue
        if arr.size == 0 or arr.shape[1] < 100:
            print(f"  !! slot {slot_i} of {os.path.basename(csv_path)}: "
                  f"only {arr.shape[1]} columns, skipped")
            failed += 1
            continue

        stem = f"own-{location.replace(' ', '')}-{date}-{slot_i:02d}"
        np.save(os.path.join(args.out_dir, stem + ".npy"), arr.astype(np.float32))
        np.save(os.path.join(args.out_dir, f"freq-{stem}.npy"), freq)
        dt_s = WINDOW_S / arr.shape[1]
        win_rows.append(dict(file_name=stem + ".npy", date=date, location=location,
                             win_start_time=w0.strftime("%H:%M:%S"),
                             n_cols=arr.shape[1], n_freq_channels=arr.shape[0],
                             sample_interval_s=round(dt_s, 6),
                             freq_min_mhz=float(freq.min()), freq_max_mhz=float(freq.max()),
                             n_boxes=len(evs)))
        for fn, typ, ev0, ev1 in evs:
            c0 = max(0, int((ev0 - w0).total_seconds() / dt_s))
            c1 = min(arr.shape[1], int((ev1 - w0).total_seconds() / dt_s))
            if c1 <= c0:
                c1 = min(arr.shape[1], c0 + int(35 / dt_s))    # zero-width fallback
            box_rows.append(dict(file_name=stem + ".npy", source_event=fn,
                                 type=TYPE_MAP.get(typ, typ),
                                 box_start_col=c0, box_end_col=c1,
                                 clipped=(ev0 < w0 or ev1 > w1)))

    W = pd.DataFrame(win_rows); B = pd.DataFrame(box_rows)
    W.to_csv(os.path.join(args.out_dir, "windows.csv"), index=False)
    B.to_csv(os.path.join(args.out_dir, "boxes.csv"), index=False)
    print(f"\nwrote {args.out_dir}/: {len(W)} window(s), {len(B)} box(es), {failed} failed")
    if len(B):
        print(f"  types: {B.type.value_counts().to_dict()}")
        centre = ((B.box_start_col + B.box_end_col) / 2 / W.set_index('file_name')
                  .n_cols.reindex(B.file_name).values)
        print(f"  box centre position in window: mean {centre.mean():.3f} "
              f"std {centre.std():.3f}  <- std must NOT be ~0 (that was the "
              f"leak that invalidated the first attempt)")


if __name__ == "__main__":
    main()
