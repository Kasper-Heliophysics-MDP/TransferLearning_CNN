# ecallisto_grabber

Unattended eCallisto scraper: pulls raw spectrograms **and** verified burst
time + type metadata from the public e-Callisto archive, with no manual
labeling step.

`denoising/sumthreshold_denoise.py` (added later, in this same folder) is a
SumThreshold-based RFI cleaner that works on both this scraper's output and
our own station's data — see `开发日志.md` for the full story (why it
replaced `denoise_new.py`, architecture, what's verified vs. still open).

Blueprint: `Kasper-Heliophysics-MDP/eCallisto-Burst-Grabber` +
`Prepro-F25/one_day.py` (Callen Fields). Reuses the same data source and
overall download approach, but:

- `one_day.py`'s `extract_bursts()` threw away the burst **type** column and
  only looked at one station/day at a time. `burst_catalog.py` here keeps
  type + the full station list and can pull a whole date range at once.
- `eCallisto-Burst-Grabber`'s labeling step (`locate_bursts.py`) is an
  interactive GUI a human has to click through. Not needed here: the Monstein
  catalog (`soleil.i4ds.ch/.../BurstLists/2010-yyyy_Monstein/`) already has
  human-verified time + type for every event, network-wide, back to 2010.
  This module just automates "catalog says X happened here, at this time, of
  this type -> go get it".

## Files

- `burst_catalog.py` — fetch/parse the Monstein burst list for a date range,
  filter by station/type.
- `fetch.py` — download+concatenate one station's raw FITS files for a day.
  Reads each file's **actual** frequency axis and sample interval from its
  FITS header/binary-table instead of assuming a fixed value — a real check
  during development found AUSTRIA-UNIGRAZ at 0.25s/sample with a
  non-uniform 45–81 MHz frequency axis, while our own station's CSVs are
  0.1s/sample and ~16–24 MHz. These differ per station and have to be read,
  not hardcoded.
- `extract_events.py` — crop each cataloged event out of a downloaded day
  (event's own start/end time + a buffer, default 60s each side) and save it,
  with metadata in the **same column layout as our own
  `burst_list_240330_240729.csv`** (`file_name, date, location, start_time,
  end_time, type`) plus calibration columns (`freq_min_mhz`, `freq_max_mhz`,
  `n_freq_channels`, `sample_interval_s`) so the two sources can be
  concatenated directly.
- `scrape.py` — CLI orchestration. Resumable (skips station/date pairs
  already in `metadata.csv`).
- `extract_windows.py` + `scrape_windows.py` — **the detection-training data
  path** (added 2026-08-11). Saves FITS-boundary-aligned 15-minute windows
  instead of per-event crops. See "Why a second extraction path" below.

## Why a second extraction path (`extract_windows.py`)

`extract_events.py` crops `event ± buffer_s`, which makes the event land dead
centre of every crop — measured on the real output: **1694 positive events,
normalized box centre mean 0.5000, std 0.0000**. For a classifier that is
harmless; for the object detector `TRAINING_PLAN.md` now calls for, it is
label leakage — a model that always predicts a box at the horizontal centre
scores well without learning anything about bursts, making the Phase 1 mAP
uninterpretable.

`extract_windows.py` saves the archive's own 15-minute FITS files as the unit
instead (which is also what the published e-Callisto detection papers use).
Event position within the window is then whatever it really was, negatives are
windows with no cataloged event (no separate sampling logic needed), and every
window has identical dimensions so nothing gets rescaled by a different factor.

The two paths coexist: `extract_events.py`'s output is still what
`event_review/` reviews and what `denoising/` was validated against.

## What this deliberately does NOT do (yet)

- **No resizing.** Every saved `.npy` is native resolution: full frequency
  axis, and time axis sized either to the event's true duration + buffer
  (`extract_events.py`) or to the FITS file's own 15 minutes
  (`extract_windows.py`) — never to a fixed pixel count. Resizing to a fixed
  height should happen *after* aligning all stations (and our own station)
  onto a common physical frequency band, otherwise the same true drift rate
  ends up looking like a different slope depending on which station a sample
  came from. That alignment step isn't built yet (now scheduled as Phase 3,
  see `TRAINING_PLAN.md`).
- **No FITS caching.** Both paths download a whole station-day and keep only
  the crops/windows they were asked for. `extract_windows.py` mitigates this
  by keeping ±2 slots of context around every event (~19.8 GB for all 7
  stations), so future changes to the canvas length don't force a re-download
  — but a change beyond ±30 minutes still would.

## Usage

```bash
# --- per-event crops (scrape.py): what event_review/ and denoising/ use ---

# smoke test: one station, one day, capped at 1 new station-day
python scrape.py --start 2024-03-01 --end 2024-03-01 \
    --stations AUSTRIA-UNIGRAZ --out-dir /tmp/test --limit 1

# real run: specific types, let it run across many stations/days
python scrape.py --start 2021-01-01 --end 2024-03-31 \
    --types II III V --out-dir ../data/ecallisto/raw_events

# Ctrl-C any time; rerunning the same command skips (station, date) pairs
# already present in <out-dir>/metadata.csv

# --- 15-minute windows (scrape_windows.py): the detection training set ---

# smoke test
python scrape_windows.py --start 2022-01-01 --end 2022-01-31 \
    --stations Arecibo-Observatory --types II III V \
    --out-dir /tmp/win_test --limit 1

# the Phase 1 run (~9 hours, ~6.6 GB)
python scrape_windows.py --start 2021-01-01 --end 2024-03-31 \
    --stations Arecibo-Observatory --types II III V \
    --out-dir ../data/ecallisto/windows

# then eyeball the boxes before trusting the output
python ../detection/render_windows.py ../data/ecallisto/windows /tmp/render --n 12
```

Note `fetch_catalog_range()` resolves to **whole months** (documented in its
docstring: "only year/month are used"), so `--start`/`--end` inside one month
still queues that entire month's station-days -- `--limit` is what actually
caps a smoke test, not a one-day date range.

`scrape_windows.py` writes `windows.csv` (one row per 15-minute window) and
`boxes.csv` (one row per burst box, with `time_source` recording whether the
box came from a human correction, the raw catalog, or the zero-width default).
It resumes off `windows.csv`.

## Verified against real data

`fetch.py` and `scrape.py` were both run against the live archive while
building this (not just written against the README's description of the
format) — confirmed: a full AUSTRIA-UNIGRAZ day downloads and concatenates to
the expected shape, and a `--limit 1` run correctly extracted 2 real
Type III events on 2024-03-01 with metadata that matches the catalog
(`06:03-06:04` and `08:08-08:09` UTC) and correct `.npy` shapes given the
60s buffer and 0.25s sample interval (e.g. 3 minutes / 0.25s = 720 columns).
