"""
survey_stations.py

Sampled per-station survey: for every station contributing II/V events to the
catalog, download a handful of its files (not a whole day) spread across
different cataloged dates, to read its real frequency range plus an RFI
severity score (rfi_quality.py) -- not just a basic data sanity check, since
std/saturation alone don't catch horizontal/vertical line interference (a
spectrogram full of RFI stripes can have perfectly normal std and zero
saturation). Used to decide which stations are worth scraping in bulk.

Usage:
    python survey_stations.py --start 2021-01-01 --end 2024-03-31 --types II V \
        --samples-per-station 3 --out stations_survey.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, date

import numpy as np
import pandas as pd

from burst_catalog import explode_by_station, fetch_catalog_range, filter_catalog
from fetch import _list_day_urls, download_fits
from rfi_quality import rfi_severity


def survey(start: date, end: date, types: list[str] | None, samples_per_station: int = 3) -> pd.DataFrame:
    print(f"Fetching Monstein catalog {start} .. {end} ...")
    catalog = fetch_catalog_range(start, end)
    exploded = explode_by_station(catalog)
    filtered = filter_catalog(exploded, types=types, drop_uncertain=False)

    event_counts = filtered.groupby("station").size().sort_values(ascending=False)
    stations = list(event_counts.index)
    print(f"{len(stations)} distinct stations contribute a {types or 'ALL'} event at least once, "
          f"sampling up to {samples_per_station} file(s) each")

    rows = []
    for i, station in enumerate(stations):
        # spread sample dates out (not just the first N rows, which can cluster on one date)
        station_rows = filtered[filtered["station"] == station]
        sample_dates = list(dict.fromkeys(station_rows["date"]))[:samples_per_station]

        freq_min, freq_max, n_freq, interval = None, None, None, None
        h_fracs, v_fracs, stds, sats, zeros = [], [], [], [], []
        n_ok = 0
        for date_str in sample_dates:
            y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
            try:
                urls = _list_day_urls(y, m, d)
            except Exception as e:
                print(f"  [{i+1}/{len(stations)}] {station} {date_str}: could not list day: {e}")
                continue
            station_urls = [u for u in urls if station in u]
            if not station_urls:
                continue
            f = download_fits(station_urls[len(station_urls) // 2])  # a file mid-day, not the very first
            if f is None:
                continue

            n_ok += 1
            freq_min, freq_max, n_freq, interval = (
                float(f.freq_mhz.min()), float(f.freq_mhz.max()), len(f.freq_mhz), f.sample_interval_s
            )
            d8 = f.data.astype(np.float64)
            stds.append(float(d8.std()))
            sats.append(float((d8 >= 254).mean()))
            zeros.append(float((d8 == 0).mean()))
            r = rfi_severity(f.data)
            h_fracs.append(r["horizontal_rfi_frac"])
            v_fracs.append(r["vertical_rfi_frac"])

        if n_ok == 0:
            print(f"  [{i+1}/{len(stations)}] {station}: no usable sample file found")
            continue

        rows.append(
            {
                "station": station,
                "n_events": int(event_counts[station]),
                "n_samples": n_ok,
                "freq_min_mhz": freq_min,
                "freq_max_mhz": freq_max,
                "n_freq_channels": n_freq,
                "sample_interval_s": interval,
                "data_std": float(np.mean(stds)),
                "frac_saturated": float(np.mean(sats)),
                "frac_zero": float(np.mean(zeros)),
                "horizontal_rfi_frac": float(np.mean(h_fracs)),
                "vertical_rfi_frac": float(np.mean(v_fracs)),
            }
        )
        print(f"  [{i+1}/{len(stations)}] {station}: {freq_min:.1f}-{freq_max:.1f} MHz, "
              f"{event_counts[station]} events, RFI h={np.mean(h_fracs):.3f} v={np.mean(v_fracs):.3f} "
              f"(n={n_ok} samples)")

    return pd.DataFrame(rows).sort_values("n_events", ascending=False).reset_index(drop=True)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--types", nargs="*", default=None)
    p.add_argument("--samples-per-station", type=int, default=3)
    p.add_argument("--out", default="stations_survey.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = survey(
        datetime.strptime(args.start, "%Y-%m-%d").date(),
        datetime.strptime(args.end, "%Y-%m-%d").date(),
        args.types,
        samples_per_station=args.samples_per_station,
    )
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} stations to {args.out}")
