"""
data_access.py

Source-aware loading of raw + cleaned spectrograms, for both data sources
that share extract_events.py's metadata.csv schema:

- eCallisto (ecallisto_grabber/scrape.py output): .npy IS the true raw crop.
  Cleaned = clean_ecallisto_events.clean_event_file() run on it directly.
- Own-station (data/burst_data/rough_events/): the .npy on disk is already
  the PRODUCTION-cleaned output -- extract_events_own_station.py denoises
  before cropping (apply_denoising=True, cleaning_method="fast"), confirmed
  by reading that script directly. There is no raw own-station crop saved
  anywhere. True raw has to be reconstructed from the original per-session
  CSV in data/burst_data/csv/original/, reading only the needed time slice
  (those files run up to ~1.3GB -- never load one whole).
"""

from __future__ import annotations

import glob
import os
import re
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecallisto_grabber", "denoising"))
from clean_ecallisto_events import clean_event_file, event_burst_indices  # noqa: E402
from sumthreshold_denoise import clean, _REFERENCE_TIME_WIN, _REFERENCE_SAMPLE_INTERVAL_S  # noqa: E402

SOURCE_ECALLISTO = "ecallisto"
SOURCE_OWN_STATION = "own_station"

# what production actually runs for each source (slicing_utils_new.py /
# extract_events_own_station.py use "fast"; clean_ecallisto_events' own
# default is "comprehensive") -- the parameter panel seeds from these so
# opening it without touching anything doesn't change what's displayed
PRODUCTION_METHOD = {SOURCE_ECALLISTO: "comprehensive", SOURCE_OWN_STATION: "fast"}
DEFAULT_KNOWN_BURST_MARGIN = 10

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OWN_STATION_CSV_DIR = os.path.normpath(os.path.join(_REPO_ROOT, "data", "burst_data", "csv", "original"))
OWN_STATION_FREQ_META_DIR = os.path.normpath(
    os.path.join(_REPO_ROOT, "data", "burst_data", "rough_events", "meta")
)

_VRFI_DEFAULTS = dict(vrfi_coverage_threshold=0.1, vrfi_sigma=3.0)  # matches clean()'s own defaults

_METHOD_PRESETS = {
    "comprehensive": dict(base_threshold_sigma=12.0, suppression_ramp=2.0, burst_sigma=2.0,
                           window_sizes=(1, 2, 4, 8, 16, 32), **_VRFI_DEFAULTS),
    "fast": dict(base_threshold_sigma=12.0, suppression_ramp=2.0, burst_sigma=2.0,
                 window_sizes=(1, 4, 16), **_VRFI_DEFAULTS),
    "conservative": dict(base_threshold_sigma=9.0, suppression_ramp=3.0, burst_sigma=2.0,
                          window_sizes=(1, 2, 4, 8, 16, 32), **_VRFI_DEFAULTS),
}

# First-pass suggested remedies per event_review.review_store problem tag,
# for the "try an automatic retry before falling back to full manual
# case-by-case tuning" workflow. Seeded from what's actually known about
# each mechanism (see sumthreshold_denoise.py docstrings), NOT validated
# against real review outcomes yet -- there was zero review data when these
# were written. Expect to revise once auto-retry has actually run on real
# flagged events.
#
# TAG_HORIZONTAL_RFI has no confident preset: persistent_row_correction only
# removes each row's constant offset, not its own texture/fluctuation (see
# that function's docstring) -- this is a documented structural limitation,
# not just an undertuned parameter, so nudging base_threshold_sigma down is
# a guess, not a known fix. Included anyway so the workflow doesn't skip the
# category, but expect a lower auto-retry success rate here specifically.
SUGGESTED_PARAMS_BY_TAG = {
    "time_wrong": {},  # handled via manual_burst_range, not a clean() param change
    "burst_faint": dict(known_burst_margin=DEFAULT_KNOWN_BURST_MARGIN * 3, suppression_ramp=3.0),
    "horizontal_rfi": dict(base_threshold_sigma=9.0),
    "vertical_rfi": dict(vrfi_coverage_threshold=0.2, vrfi_sigma=2.5),
}


# Gentler cleaning preset offered (as a one-click starting point, NOT applied
# automatically) for the burst types that the default settings treat too
# harshly. Every value moves in the "suppress less / protect more" direction,
# and each direction was read off clean()'s own code rather than guessed:
#   base_threshold_sigma   -> sumthreshold_axis_ratio's threshold; HIGHER flags less
#   suppression_ramp       -> frac = clip(ratio/ramp); LARGER suppresses less
#   burst_sigma            -> burst_weight_2d sensitivity; LOWER protects more
#   vrfi_coverage_threshold / vrfi_sigma -> HIGHER makes the vertical-RFI veto
#                             harder to trigger, so real broadband bursts survive
#   known_burst_margin     -> LARGER protects further past the catalog window,
#                             which matters because catalog times are known to
#                             run early (see TRAINING_PLAN.md)
#
# Type III keeps the production defaults: it is the class the defaults were
# tuned against and the one with enough reviewed data to say they work.
#
# HONEST LIMITS -- read before trusting this:
#  1. These values are mechanism-derived, NOT validated against an outcome
#     metric. An energy-ratio check was attempted and discarded as unreliable
#     (it returned NaN on every long Type II and claimed the gentle preset was
#     WORSE on Type V) -- the same failure mode this project already documented
#     for automated "is the burst still there" metrics. The human eye in this
#     very panel is the intended judge; this preset only saves slider dragging.
#  2. For LONG Type II events these sliders can barely help at all. Inside the
#     protected window every one of them has mathematically zero effect
#     (measured: max|diff| exactly 0.0000 across all six), and the burst is
#     mostly lost one stage earlier -- robust_smooth_background() absorbed 97%
#     of a real Type II's excess, and it never receives known_burst_weight.
#     The preset can only help the parts of the burst OUTSIDE the catalog
#     window, and short events where the protected window is small.
GENTLE_PARAMS_BY_TYPE = {
    "II": dict(known_burst_margin=120, base_threshold_sigma=16.0, suppression_ramp=3.5,
               burst_sigma=1.0, vrfi_coverage_threshold=0.26, vrfi_sigma=4.5),
    "V": dict(known_burst_margin=120, base_threshold_sigma=16.0, suppression_ramp=3.5,
              burst_sigma=1.0, vrfi_coverage_threshold=0.26, vrfi_sigma=4.5),
}


# Burst types whose right-hand panel defaults to minimal_view() instead of the
# full clean(). Measured on 8 reviewed-usable Type II/V events (raw vs current
# cleaned, side by side, judged by eye -- the only check this project trusts
# for "is the burst still there"):
#   2/8  cleaned all but erased a faint Type V  (both had a zero-width catalog
#        entry, so the protected window was ~4% of the crop and auto-detection
#        alone had to carry it -- it didn't)
#   1/8  cleaned introduced a hard discontinuity that isn't in the raw
#   1/8  cleaned was genuinely CRISPER than raw (striped structure sharper)
#   4/8  about the same
# So "II/V never benefit from denoising" is too strong -- 1 in 8 does. Hence a
# default, not a removal: SHOW_FULL_CLEAN_KEY lets the reviewer switch back
# per-event for that minority.
MINIMAL_VIEW_TYPES = {"II", "V"}


def uses_minimal_view(burst_type: str) -> bool:
    return str(burst_type).strip() in MINIMAL_VIEW_TYPES


def minimal_view(raw: np.ndarray) -> np.ndarray:
    """Per-row median subtraction and nothing else -- the standard e-Callisto
    quicklook.

    Removes the per-channel DC offsets that dominate the untouched raw panel
    (render_pair draws raw with no normalisation at all), while doing ZERO
    suppression, so unlike clean() it cannot erase a burst. This is what long
    Type II events need: there, clean() loses the burst in the background-fit
    stage before any tunable parameter is reached (measured: robust_smooth_
    background absorbed 97% of a real Type II's excess), and every slider in
    the panel has mathematically zero effect inside the protected window.
    """
    return raw - np.median(raw, axis=1, keepdims=True)


def gentle_params_for(burst_type: str) -> dict | None:
    """Gentle preset for this burst type, or None if the type should keep the
    production defaults (Type III and anything else)."""
    return GENTLE_PARAMS_BY_TYPE.get(str(burst_type).strip())


def default_params(method: str) -> dict:
    """A copy -- caller may mutate freely before passing to reprocess()."""
    return dict(_METHOD_PRESETS.get(method, _METHOD_PRESETS["comprehensive"]))


def time_win_for(sample_interval_s: float) -> int:
    tw = round(_REFERENCE_TIME_WIN * _REFERENCE_SAMPLE_INTERVAL_S / sample_interval_s)
    return max(tw // 2 * 2 + 1, 3)


def load_metadata(out_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(out_dir, "metadata.csv"), dtype=str)


def protect_indices_for(event_indices: tuple[int, int] | None, margin: int, n_cols: int) -> tuple[int, int] | None:
    if event_indices is None:
        return None
    lo, hi = event_indices
    return max(0, lo - margin), min(n_cols, hi + margin)


# ---------------------------------------------------------------------------
# own-station true-raw reconstruction
# ---------------------------------------------------------------------------

def _stem_station_name(stem: str) -> str:
    """'240330182002-PeachMountian' -> 'peachmountian' (strip the leading
    12-digit date+time prefix, spaces, and case, for fuzzy comparison)."""
    m = re.match(r"^\d{12}-(.+)$", stem)
    return (m.group(1) if m else stem).replace(" ", "").lower()


@st.cache_data(show_spinner=False)
def _own_station_csv_stems() -> list[str]:
    """Stems of every original CSV that has a matching meta/freq-*.npy --
    i.e. every source file rough_events actually pulled at least one real
    event from, not the whole csv/original/ listing (which also holds the
    non-spectral burst_list_*.csv)."""
    stems = []
    for path in glob.glob(os.path.join(OWN_STATION_FREQ_META_DIR, "freq-*.npy")):
        stem = os.path.basename(path)[len("freq-"):-len(".npy")]
        stems.append(stem)
    return stems


def _hms_to_seconds(time_strs: pd.Series) -> np.ndarray:
    """Vectorized 'HH:MM:SS[.mmm]' -> seconds-since-midnight float array."""
    parts = time_strs.str.split(":", expand=True)
    return (parts[0].astype(float) * 3600 + parts[1].astype(float) * 60 + parts[2].astype(float)).to_numpy()


def _elapsed_since_session_start(time_strs: pd.Series, session_start: str) -> np.ndarray:
    """Seconds elapsed since `session_start` (a 'HH:MM:SS' clock time),
    handling one midnight wrap.

    own-station recordings can run overnight (confirmed real case:
    240621161136-Skyline High School.csv spans 16:11 -> 01:18 the next day).
    The Time column is bare HH:MM:SS with no date, so past midnight the
    STRINGS stop being monotonic ("00:00:01" sorts before "23:59:59"
    alphabetically despite coming later) -- searchsorted on raw strings
    silently returns garbage there (confirmed: searching for a 23:53 target
    inside that file returned len(array), i.e. "not found", instead of the
    real index partway through). Converting to elapsed-seconds-since-the-
    file's-own-start and adding a day to anything that comes out negative
    (i.e. clock-earlier than the reference, which can only mean next-day)
    restores monotonicity. Same underlying wrap this project already
    handles elsewhere (ecallisto_grabber's TIME-END=24:00:00 parsing,
    clean_ecallisto_events.event_burst_indices's ev_start<win_start check) --
    this function just hadn't been given the same treatment yet.
    """
    ref = _hms_to_seconds(pd.Series([session_start]))[0]
    elapsed = _hms_to_seconds(time_strs) - ref
    elapsed[elapsed < 0] += 86400.0
    return elapsed


def resolve_own_station_csv(row: dict) -> str:
    """(date, location) -> the exact original CSV path this event's raw data
    actually came from.

    Fuzzy match handles a real, confirmed spelling drift: metadata's
    location "PeachMountian" (a typo, inherited from the first-ever file for
    that station) vs. the correctly-spelled "Peach Mountain.csv" used by
    every later recording of the same station -- exact/normalized matching
    alone resolves only 64/74 real rows, silently failing the other 10.
    Verified: SequenceMatcher ratio is >=0.92 for genuine same-station
    spelling variants seen in this dataset vs. <=0.33 for any other station,
    so 0.8 is a safe threshold with real margin on both sides.

    Time-range verification handles the other real wrinkle: some
    (date, station) pairs have multiple recording sessions in one day (e.g.
    8 separate Peach Mountain files on 240420) -- date+station alone doesn't
    disambiguate, so this reads (only) the Time column of each remaining
    candidate to find which session's range actually covers this event.
    """
    date = str(row["date"])
    same_day = [s for s in _own_station_csv_stems() if s[:6] == date]
    if not same_day:
        raise FileNotFoundError(f"no original CSV found for date={date}")

    target = str(row["location"]).replace(" ", "").lower()
    ranked = sorted(same_day, key=lambda s: -SequenceMatcher(None, target, _stem_station_name(s)).ratio())
    best_ratio = SequenceMatcher(None, target, _stem_station_name(ranked[0])).ratio()
    if best_ratio < 0.8:
        raise FileNotFoundError(
            f"no CSV stem matches location={row['location']!r} on {date} "
            f"(closest candidate {ranked[0]!r}, ratio={best_ratio:.2f})"
        )
    candidates = [s for s in ranked if SequenceMatcher(None, target, _stem_station_name(s)).ratio() >= 0.8]

    if len(candidates) == 1:
        return os.path.join(OWN_STATION_CSV_DIR, candidates[0] + ".csv")

    target_start, target_end = str(row["start_time"]), str(row["end_time"])
    for stem in candidates:
        path = os.path.join(OWN_STATION_CSV_DIR, stem + ".csv")
        times = pd.read_csv(path, usecols=["Time"])["Time"].astype(str)
        # elapsed-since-file-start comparison, not raw string <=/>= -- a
        # session crossing midnight (see _elapsed_since_session_start) has a
        # LEXICALLY SMALLER end time than its own start, which would make a
        # real covering file look like it doesn't cover the target
        session_start = times.iloc[0]
        target_lo, target_hi = _elapsed_since_session_start(pd.Series([target_start, target_end]), session_start)
        file_end_elapsed = _elapsed_since_session_start(pd.Series([times.iloc[-1]]), session_start)[0]
        if target_lo <= target_hi and target_hi <= file_end_elapsed:
            return path
    raise FileNotFoundError(
        f"{len(candidates)} candidate sessions for {row['location']} on {date}, "
        f"none of their Time ranges covers [{target_start}, {target_end}]"
    )


@st.cache_data(show_spinner="Reading original CSV slice...")
def _load_own_station_raw_slice(
    csv_path: str, start_time: str, end_time: str
) -> tuple[np.ndarray, np.ndarray]:
    """Read only the [start_time, end_time) row range + frequency axis from a
    (potentially gigabyte-scale) original CSV. Never loads the whole file:
    the Time column is read alone first to locate row offsets, then only
    that row range is read with all columns.

    Returns (spectral (freq, time) float32, freq_mhz float32).
    """
    times = pd.read_csv(csv_path, usecols=["Time"])["Time"].astype(str)
    # elapsed-seconds-since-file-start, not raw string searchsorted -- see
    # _elapsed_since_session_start's docstring for why (real overnight
    # session crossing midnight makes the raw Time strings non-monotonic)
    elapsed = _elapsed_since_session_start(times, times.iloc[0])
    target_start, target_end = _elapsed_since_session_start(pd.Series([start_time, end_time]), times.iloc[0])
    start_row = int(np.searchsorted(elapsed, target_start, side="left"))
    end_row = int(np.searchsorted(elapsed, target_end, side="right"))

    header = pd.read_csv(csv_path, nrows=0).columns
    freq_mhz = header[2:].astype(float).to_numpy(dtype=np.float32) / 1e6

    # skiprows=range(1, start_row+1) skips that many DATA rows (row 0 stays
    # the header, since it's not in the skip set) -- standard pattern for
    # reading a specific row range without loading everything before it
    chunk = pd.read_csv(csv_path, skiprows=range(1, start_row + 1), nrows=end_row - start_row)
    spectral = chunk.iloc[:, 2:].to_numpy(dtype=np.float32).T  # (time, freq) -> (freq, time)
    return spectral, freq_mhz


# ---------------------------------------------------------------------------
# eCallisto
# ---------------------------------------------------------------------------

def _ecallisto_freq(out_dir: str, row: dict) -> np.ndarray | None:
    path = os.path.join(out_dir, "meta", f"freq-{row['location']}-{row['date']}.npy")
    return np.load(path) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# unified per-event loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading event...")
def load_event(out_dir: str, row_dict: dict, source: str) -> dict:
    """raw + production-cleaned + everything needed to render one event, for
    either source. `row_dict` is `row.to_dict()` (not a bare pd.Series --
    plain dicts hash cleanly for st.cache_data)."""
    row = row_dict
    sample_interval_s = float(row["sample_interval_s"])
    event_indices = event_burst_indices(pd.Series(row), sample_interval_s)

    if source == SOURCE_ECALLISTO:
        npy_path = os.path.join(out_dir, row["file_name"])
        result = clean_event_file(
            npy_path, pd.Series(row),
            method=PRODUCTION_METHOD[source], known_burst_margin=DEFAULT_KNOWN_BURST_MARGIN,
        )
        raw, cleaned = result["raw"], result["cleaned"]
        freq_mhz = _ecallisto_freq(out_dir, row)
        csv_path = None
    else:
        cleaned = np.load(os.path.join(out_dir, row["file_name"])).astype(np.float32)
        csv_path = resolve_own_station_csv(row)
        raw, freq_mhz = _load_own_station_raw_slice(csv_path, row["start_time"], row["end_time"])

    protect_indices = protect_indices_for(event_indices, DEFAULT_KNOWN_BURST_MARGIN, raw.shape[1])
    return dict(
        raw=raw, cleaned=cleaned, freq_mhz=freq_mhz, sample_interval_s=sample_interval_s,
        event_indices=event_indices, protect_indices=protect_indices, csv_path=csv_path,
    )


@st.cache_data(show_spinner="Reprocessing...")
def reprocess(
    data: np.ndarray,
    sample_interval_s: float,
    event_indices: tuple[int, int] | None,
    known_burst_margin: int,
    base_threshold_sigma: float,
    suppression_ramp: float,
    burst_sigma: float,
    window_sizes: tuple[int, ...],
    vrfi_coverage_threshold: float = 0.1,
    vrfi_sigma: float = 3.0,
) -> dict:
    """Re-run clean() on an already-loaded RAW array with explicit
    parameters. Always operates on `data` as given -- callers are
    responsible for passing the true-raw array (own-station callers must use
    _load_own_station_raw_slice's output, never the production rough_events
    .npy, or a first suppression pass can't be undone by a second one over
    already-zeroed data).

    `event_indices` here is whatever the caller decides is the true burst
    range -- pass a manually-corrected range instead of the catalog-derived
    one to protect it instead. This function doesn't know or care which one
    it got; app.py resolves that."""
    known_burst_weight = None
    if event_indices is not None:
        lo, hi = protect_indices_for(event_indices, known_burst_margin, data.shape[1])
        known = np.zeros(data.shape[1], dtype=np.float32)
        known[lo:hi] = 1.0
        known_burst_weight = np.broadcast_to(known, data.shape)

    time_win = time_win_for(sample_interval_s)
    return clean(
        data, time_win=time_win, base_threshold_sigma=base_threshold_sigma,
        suppression_ramp=suppression_ramp, burst_sigma=burst_sigma,
        window_sizes=window_sizes, known_burst_weight=known_burst_weight,
        vrfi_coverage_threshold=vrfi_coverage_threshold, vrfi_sigma=vrfi_sigma,
    )
