"""
fetch.py

Download a station's raw spectrogram data from the public e-Callisto archive.

Blueprint: Kasper-Heliophysics-MDP/Prepro-F25/one_day.py. Kept the same overall
approach (list a day's directory, filter by station, sort circularly from a UTC
offset, concatenate) but extended to also carry the per-file FITS calibration
(frequency axis in MHz + seconds-per-sample) through, instead of returning a bare
pixel array. Two real files checked while building this (AUSTRIA-UNIGRAZ) showed
0.25s/sample and a NON-uniform frequency axis read from the FITS binary table
(HDU 1) -- NOT a fixed MHz-per-row step -- and our own station's CSVs are
0.1s/sample. Both the sample interval and the frequency axis therefore have to be
read per file/station rather than assumed.
"""

from __future__ import annotations

import gzip
import io
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import urljoin

import numpy as np
import requests
from astropy.io import fits
from bs4 import BeautifulSoup

BASE_URL = "https://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"


def _get_with_retry(url: str, timeout: int, retries: int = 3, backoff_s: float = 2.0) -> requests.Response:
    """A real scrape run hits thousands of requests against a shared academic
    server; transient resets/timeouts are expected (seen in practice: a
    'Connection reset by peer' and a refused connection in the same 10-station-day
    test batch). Retry a few times with backoff instead of losing that file."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _parse_hms(s: str, base_date) -> datetime:
    """Parse a FITS 'HH:MM:SS[.sss]' timestamp against `base_date`. e-Callisto
    files use hour=24 for exact midnight-at-end-of-day (e.g. TIME-END='24:00:00'
    on a file spanning 23:45-24:00) which datetime.strptime rejects outright --
    found in the wild on the very first real multi-station-day test run, and
    silently dropped every last-file-of-the-day until this existed."""
    s = s.split(".")[0]
    h, m, sec = (int(x) for x in s.split(":"))
    extra_days, h = divmod(h, 24)
    return datetime.combine(base_date, datetime.min.time()) + timedelta(days=extra_days, hours=h, minutes=m, seconds=sec)


@dataclass
class FitsFile:
    """One 15-minute e-Callisto FITS file, decoded."""

    url: str
    data: np.ndarray  # (n_freq, n_time) uint8, as stored (high freq first)
    freq_mhz: np.ndarray  # (n_freq,) non-uniform, read from the FITS binary table
    sample_interval_s: float  # CDELT1
    time_obs: datetime  # absolute UTC start time of this file
    time_end: datetime


@dataclass
class StationDay:
    """
    A full day (or from start_time onward) of concatenated station data.

    IMPORTANT: files are NOT guaranteed to be back-to-back in time. Real
    stations have gaps (equipment/network downtime) -- confirmed on a real
    scrape run: ALASKA-ANCHORAGE 2023-04-21 has 59 files spanning 00:00-24:00,
    but a naive "column_start_time + column_index * sample_interval_s" formula
    put the end of the data at 14:44, silently misindexing (or dropping) every
    event after that gap even though later files existed. `col_offsets` below
    tracks each file's real starting column so time_to_column can be gap-aware.
    """

    station: str
    date: str  # YYYYMMDD
    spectrogram: np.ndarray  # (n_freq, n_time_total) uint8, one block per file, concatenated in order
    freq_mhz: np.ndarray  # (n_freq,) taken from the first file; see note in fetch_station_day
    sample_interval_s: float
    files: List[FitsFile] = field(default_factory=list)
    col_offsets: List[int] = field(default_factory=list)  # col_offsets[i] = spectrogram column where files[i] starts

    @property
    def column_start_time(self) -> datetime:
        return self.files[0].time_obs

    def time_to_column(self, t: datetime) -> Optional[int]:
        """Absolute UTC datetime -> column index, or None if `t` isn't covered
        by any downloaded file (before the first file, after the last, or in a
        gap between two files)."""
        for f, col_start in zip(self.files, self.col_offsets):
            if f.time_obs <= t <= f.time_end:
                offset = round((t - f.time_obs).total_seconds() / self.sample_interval_s)
                return col_start + min(offset, f.data.shape[1] - 1)
        return None

    def has_gap(self, t_start: datetime, t_end: datetime) -> bool:
        """True if [t_start, t_end] is not fully covered by a contiguous run of
        files -- i.e. any part of the window falls before the first covering
        file, after the last, or in a gap between two consecutive files."""
        relevant = [
            (f, c) for f, c in zip(self.files, self.col_offsets) if f.time_end > t_start and f.time_obs < t_end
        ]
        if not relevant:
            return True
        if relevant[0][0].time_obs > t_start or relevant[-1][0].time_end < t_end:
            return True
        for (a, _), (b, _) in zip(relevant, relevant[1:]):
            if (b.time_obs - a.time_end).total_seconds() > self.sample_interval_s * 2:
                return True
        return False


def _list_day_urls(year: int, month: int, day: int) -> List[str]:
    path = f"{year:04d}/{month:02d}/{day:02d}/"
    url = urljoin(BASE_URL, path)
    resp = _get_with_retry(url, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href in ("../", "./") or href.startswith("?"):
            continue
        out.append(urljoin(url, href))
    return out


def download_fits(url: str, timeout: int = 30) -> Optional[FitsFile]:
    """Download and decode one .fit.gz file. Returns None on any parse failure
    (a handful of files on the archive are truncated/corrupt; skip rather than crash)."""
    try:
        r = _get_with_retry(url, timeout=timeout)
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
            with fits.open(io.BytesIO(gz.read())) as hdul:
                data = np.array(hdul[0].data)
                header = hdul[0].header
                freq_mhz = np.array(hdul[1].data[0][1], dtype=float) if len(hdul) > 1 else None
                if freq_mhz is None or len(freq_mhz) != data.shape[0]:
                    # fall back: can't calibrate this file's frequency axis reliably
                    return None
                sample_interval_s = float(header.get("CDELT1", np.nan))
                date_obs = datetime.strptime(header.get("DATE-OBS", header.get("DATE")), "%Y/%m/%d").date()
                t0 = _parse_hms(header.get("TIME-OBS", "00:00:00"), date_obs)
                t1 = _parse_hms(header.get("TIME-END", "00:00:00"), date_obs)
                if t1 < t0:
                    t1 += timedelta(days=1)
                return FitsFile(
                    url=url,
                    data=data,
                    freq_mhz=freq_mhz,
                    sample_interval_s=sample_interval_s,
                    time_obs=t0,
                    time_end=t1,
                )
    except Exception as e:
        print(f"    [fetch] skipping {url}: {e}")
        return None


def fetch_station_day(
    station: str, year: int, month: int, day: int, start_time: str = "000000"
) -> Optional[StationDay]:
    """
    Download and concatenate all of a station's files for one UTC day, starting
    from `start_time` ("HHMMSS") and wrapping circularly (matches Prepro-F25's
    one_day.py behaviour: local-day boundaries don't line up with UTC-day file
    listings, so we rotate the file list instead of just sorting it).

    Frequency calibration note: a station is USUALLY on one fixed config
    (channel count) all day, but not always -- confirmed on a real preview run:
    Australia-ASSA 2021-04-19 has both 200-channel and 400-channel files mixed
    into the same day. Picking "whichever config the first file happens to be"
    and discarding every file that doesn't match throws away most of the day
    (that's what happened here: only 1 of ~90 files survived, missing the
    actual cataloged event entirely). Instead, group files by channel count and
    keep the group with the most files -- the day's dominant config -- so a
    handful of stray files in another mode don't gut the whole day.
    """
    urls = _list_day_urls(year, month, day)
    station_urls = [u for u in urls if station in u]
    if not station_urls:
        print(f"  [fetch] no files found for {station} on {year:04d}-{month:02d}-{day:02d}")
        return None

    time_re = re.compile(r"_(\d{6})_")

    def file_time(u: str) -> int:
        m = time_re.search(u)
        if not m:
            return 0
        h, mi, s = int(m.group(1)[:2]), int(m.group(1)[2:4]), int(m.group(1)[4:6])
        return h * 3600 + mi * 60 + s

    station_urls.sort(key=file_time)
    offset_sec = int(start_time[:2]) * 3600 + int(start_time[2:4]) * 60 + int(start_time[4:6])
    idx = next((i for i, u in enumerate(station_urls) if file_time(u) >= offset_sec), 0)
    ordered_urls = station_urls[idx:] + station_urls[:idx]

    files: List[FitsFile] = []
    for u in ordered_urls:
        f = download_fits(u)
        if f is not None:
            files.append(f)

    if not files:
        print(f"  [fetch] all files failed to decode for {station} on {year:04d}-{month:02d}-{day:02d}")
        return None

    # files were downloaded in ordered_urls order (time-sorted, rotated from start_time);
    # sort strictly by time_obs so col_offsets and has_gap()'s adjacency check are meaningful
    files.sort(key=lambda f: (f.time_obs, f.url))

    # De-duplicate simultaneous files: some stations run multiple independent
    # receivers and publish a full file series per timestamp for each --
    # confirmed real case: ALASKA-HAARP 2024-03-07 has a "_62" and "_63"
    # suffixed series, same channel count and frequency range, but NOT
    # duplicate data (row-to-row correlation ~0.06 -- genuinely different
    # measurements, not a redundant copy of the same one). Grouping by
    # channel count alone (below) can't catch this, since both series have
    # identical channel counts -- naive concatenation would silently double
    # the time axis, with the "next chunk" actually being a simultaneous
    # second receiver's reading instead of the next 15 minutes. Every column
    # index computed downstream of this (has_gap, time_to_column, every
    # cataloged event's crop) would be wrong for a day like this. Keep one
    # file per overlapping time slot, picking deterministically (files are
    # already sorted by (time_obs, url), so the earlier url wins) so a rerun
    # is reproducible.
    deduped: List[FitsFile] = []
    for f in files:
        if deduped:
            prev = deduped[-1]
            overlap_s = (min(f.time_end, prev.time_end) - max(f.time_obs, prev.time_obs)).total_seconds()
            this_dur_s = (f.time_end - f.time_obs).total_seconds()
            if overlap_s > 0 and this_dur_s > 0 and overlap_s > 0.5 * this_dur_s:
                continue
        deduped.append(f)
    if len(deduped) < len(files):
        print(f"    [fetch] {station} {year:04d}-{month:02d}-{day:02d} has "
              f"{len(files) - len(deduped)} file(s) from a simultaneous second receiver "
              f"(overlapping timestamps); keeping one series")
    files = deduped

    # group by channel count, keep the day's dominant config (see docstring)
    by_n_freq: dict[int, list[FitsFile]] = {}
    for f in files:
        by_n_freq.setdefault(f.data.shape[0], []).append(f)
    if len(by_n_freq) > 1:
        counts = {k: len(v) for k, v in by_n_freq.items()}
        print(f"    [fetch] {station} {year:04d}-{month:02d}-{day:02d} has mixed configs: "
              f"{counts} files by channel count; keeping the majority")
    dominant_n_freq = max(by_n_freq, key=lambda k: len(by_n_freq[k]))
    files = by_n_freq[dominant_n_freq]

    freq_mhz = files[0].freq_mhz
    sample_interval_s = files[0].sample_interval_s

    chunks: List[np.ndarray] = []
    kept_files: List[FitsFile] = []
    col_offsets: List[int] = []
    col = 0
    for f in files:
        chunks.append(f.data)
        kept_files.append(f)
        col_offsets.append(col)
        col += f.data.shape[1]
    spectrogram = np.concatenate(chunks, axis=1)

    return StationDay(
        station=station,
        date=f"{year:04d}{month:02d}{day:02d}",
        spectrogram=spectrogram,
        freq_mhz=freq_mhz,
        sample_interval_s=sample_interval_s,
        files=kept_files,
        col_offsets=col_offsets,
    )


if __name__ == "__main__":
    sd = fetch_station_day("AUSTRIA-UNIGRAZ", 2024, 3, 1, start_time="053000")
    if sd:
        print(f"station day shape: {sd.spectrogram.shape}")
        print(f"freq range: {sd.freq_mhz.min():.2f}-{sd.freq_mhz.max():.2f} MHz, "
              f"{len(sd.freq_mhz)} channels")
        print(f"sample interval: {sd.sample_interval_s}s")
        print(f"column_start_time: {sd.column_start_time}")
