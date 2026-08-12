"""
burst_catalog.py

Fetch and parse the official e-Callisto "Monstein" burst list
(https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein/).

This is a monthly, human-curated, network-wide catalog: for every burst seen
anywhere in the e-Callisto network it records date, time range, TYPE
(roman-numeral burst class, e.g. "III"), and which stations observed it.

Blueprint: Kasper-Heliophysics-MDP/Prepro-F25/one_day.py::extract_bursts().
That version already knows the URL/format but only keeps the time range for
a single station and silently discards the Type column. This module keeps
everything (type + full station list) and works across a date range instead
of a single day.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Iterable

import pandas as pd
import requests

BASE_LABELS_URL = "https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein"


def _month_url(year: int, month: int) -> str:
    return f"{BASE_LABELS_URL}/{year:04d}/e-CALLISTO_{year:04d}_{month:02d}.txt"


def fetch_month_catalog(year: int, month: int, timeout: int = 30) -> pd.DataFrame:
    """
    Download and parse one month of the Monstein burst list.

    Returns a DataFrame with columns:
        date        str, "YYYYMMDD"
        time_range  str, "HH:MM-HH:MM" (occasionally seconds are included upstream)
        type        str, e.g. "III", "II", "VI", "RBR", ... (raw code, not decoded)
        stations    list[str], station names that observed this event
                    (parenthesized "(STATION)" entries from the source file, which
                    upstream marks as a less-certain detection, are kept as-is
                    including the parentheses so callers can decide whether to trust them)

    Returns an empty DataFrame (same columns) if the month has no file yet
    (e.g. current month not published) or fails to parse.
    """
    url = _month_url(year, month)
    empty = pd.DataFrame(columns=["date", "time_range", "type", "stations"])

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [burst_catalog] could not fetch {url}: {e}")
        return empty

    rows = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        line_date, time_range, btype, stations_str = parts[0], parts[1], parts[2], parts[3]
        if not line_date.isdigit() or len(line_date) != 8:
            continue
        stations = [s.strip() for s in stations_str.split(",") if s.strip()]
        rows.append(
            {"date": line_date, "time_range": time_range, "type": btype.strip(), "stations": stations}
        )

    if not rows:
        return empty
    return pd.DataFrame(rows)


def fetch_catalog_range(start: date, end: date, timeout: int = 30) -> pd.DataFrame:
    """
    Fetch and concatenate the Monstein catalog for every month in [start, end] (inclusive).

    Args:
        start, end: datetime.date objects; only year/month are used.

    Returns:
        Concatenated DataFrame (see fetch_month_catalog for columns), sorted by date.
    """
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    frames = []
    for y, m in months:
        print(f"  [burst_catalog] fetching {y:04d}-{m:02d} ...")
        df = fetch_month_catalog(y, m, timeout=timeout)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date", "time_range", "type", "stations"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("date").reset_index(drop=True)


def explode_by_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (event, station) instead of one row per event with a station list.

    Adds:
        station:      single station name for this row (parentheses stripped)
        uncertain:    True if the source wrapped this station in parentheses,
                      e.g. "(HUMAIN)" -- the network's own convention for a
                      lower-confidence detection.
        all_stations: the full original station list for this event (same list
                      repeated across every row it was exploded into) -- kept so
                      downstream metadata can record "who else saw this burst".
    """
    df = df.copy()
    df["all_stations"] = df["stations"]
    out = df.explode("stations").rename(columns={"stations": "station"})
    out = out.dropna(subset=["station"])
    uncertain = out["station"].str.startswith("(") & out["station"].str.endswith(")")
    out["station"] = out["station"].str.strip("()")
    out["uncertain"] = uncertain.values
    return out.reset_index(drop=True)


def filter_catalog(
    df: pd.DataFrame,
    stations: Iterable[str] | None = None,
    types: Iterable[str] | None = None,
    drop_uncertain: bool = False,
) -> pd.DataFrame:
    """
    Filter an exploded (one row per station) catalog.

    Args:
        stations: keep only these station names (exact match after paren-stripping).
        types: keep only these burst type codes (exact match, e.g. ["II", "III", "V"]).
        drop_uncertain: if True, drop rows where the source marked the detection as uncertain.
    """
    out = df
    if stations is not None:
        stations = set(stations)
        out = out[out["station"].isin(stations)]
    if types is not None:
        types = set(types)
        out = out[out["type"].isin(types)]
    if drop_uncertain and "uncertain" in out.columns:
        out = out[~out["uncertain"]]
    return out.reset_index(drop=True)


if __name__ == "__main__":
    from datetime import date as _date

    df = fetch_catalog_range(_date(2024, 3, 1), _date(2024, 3, 1))
    print(df.head(10))
    exploded = explode_by_station(df)
    print("\nType counts (station-observations):")
    print(exploded["type"].value_counts().head(10))
