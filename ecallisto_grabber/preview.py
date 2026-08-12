"""
preview.py

Visual counterpart to survey_stations.py's RFI numbers: grab one real
cataloged II/V event per station and plot them in a grid (raw, unresized,
robust-percentile display scaling only -- no cleaning), so the RFI severity
scores can be eyeballed against what the spectrograms actually look like
before committing to a bulk scrape.

Usage:
    python preview.py --start 2021-01-01 --end 2024-03-31 --types II V \
        --stations Australia-ASSA BIR NORWAY-EGERSUND ... --out preview.png
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from burst_catalog import explode_by_station, fetch_catalog_range, filter_catalog
from extract_events import _parse_time_range
from fetch import fetch_station_day


def build_preview(
    stations: list[str],
    start: date,
    end: date,
    types: list[str] | None,
    out_path: str,
    buffer_s: float = 90.0,
    labels: dict[str, str] | None = None,
) -> None:
    labels = labels or {}
    print(f"Fetching Monstein catalog {start} .. {end} ...")
    catalog = fetch_catalog_range(start, end)
    exploded = explode_by_station(catalog)
    filtered = filter_catalog(exploded, types=types, drop_uncertain=True)

    ncols = 4
    nrows = (len(stations) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, station in enumerate(stations):
        ax = axes[i]
        rows = filtered[filtered["station"] == station]
        if rows.empty:
            ax.set_title(f"{station}\n(no cataloged event)", fontsize=9)
            ax.axis("off")
            continue
        row = rows.iloc[0]
        y, m, d = int(row["date"][:4]), int(row["date"][4:6]), int(row["date"][6:8])
        print(f"  [{i + 1}/{len(stations)}] {station} {row['date']} {row['time_range']} ({row['type']}) ...")

        sd = fetch_station_day(station, y, m, d)
        if sd is None:
            ax.set_title(f"{station}\n(fetch failed)", fontsize=9)
            ax.axis("off")
            continue

        ev_start, ev_end = _parse_time_range(row["time_range"], row["date"])
        win_start = ev_start - timedelta(seconds=buffer_s)
        win_end = ev_end + timedelta(seconds=buffer_s)
        if sd.has_gap(win_start, win_end):
            ax.set_title(f"{station}\n(gap around event)", fontsize=9)
            ax.axis("off")
            continue
        cs, ce = sd.time_to_column(win_start), sd.time_to_column(win_end)
        if cs is None or ce is None or ce <= cs:
            continue

        crop = sd.spectrogram[:, cs:ce].astype(float)
        vmin, vmax = np.percentile(crop, [2, 98])
        ax.imshow(
            crop, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
            extent=[0, crop.shape[1] * sd.sample_interval_s, sd.freq_mhz.min(), sd.freq_mhz.max()],
        )
        tag = f" ({labels[station]})" if station in labels else ""
        ax.set_title(f"{station}{tag}\ntype {row['type']}, {row['time_range']} UTC", fontsize=9)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("MHz", fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(len(stations), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--types", nargs="*", default=None)
    p.add_argument("--stations", nargs="+", required=True)
    p.add_argument("--buffer-s", type=float, default=90.0)
    p.add_argument("--out", default="preview.png")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_preview(
        stations=args.stations,
        start=datetime.strptime(args.start, "%Y-%m-%d").date(),
        end=datetime.strptime(args.end, "%Y-%m-%d").date(),
        types=args.types,
        out_path=args.out,
        buffer_s=args.buffer_s,
    )
