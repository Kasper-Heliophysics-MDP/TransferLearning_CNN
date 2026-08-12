"""
rfi_quality.py

Quantify RFI severity in a raw spectrogram: what fraction of frequency
channels show persistent narrowband interference (horizontal lines -- a
channel that's anomalously bright most of the time), and what fraction of
time columns show broadband instantaneous interference (vertical lines -- a
moment where many channels spike at once).

Same MAD/occupancy-threshold idea as radburst_tl/data_preprocessing_new/
denoise_new.py::AdvancedRFICleaner (steps 1-3), reimplemented standalone here
rather than cross-imported so this module doesn't pick up scipy/skimage as
dependencies just for a quality scan. Not a cleaner -- just a severity score
for deciding which stations are worth scraping in bulk.
"""

from __future__ import annotations

import numpy as np


def rfi_severity(
    spectrogram_freq_by_time: np.ndarray,
    mad_threshold: float = 3.0,
    occupancy_threshold: float = 0.3,
) -> dict:
    """
    Args:
        spectrogram_freq_by_time: (n_freq, n_time) raw array, as produced by
            fetch.py (frequency-major, matching FITS storage).
        mad_threshold: how many MADs above a channel's/column's own median
            counts as "anomalously bright" (same default as denoise_new.py).
        occupancy_threshold: a channel/column must be anomalously bright this
            fraction of the time/band to count as RFI, not just an occasional
            real burst passing through it.

    Returns:
        horizontal_rfi_frac: fraction of frequency channels flagged as a
            persistent narrowband carrier.
        vertical_rfi_frac: fraction of time columns flagged as a broadband
            instantaneous pulse.
        row_occupancy / col_occupancy: the raw per-channel / per-column
            occupancy arrays, in case a caller wants to plot them.
    """
    S = spectrogram_freq_by_time.astype(np.float64)

    # Horizontal RFI = a channel that's persistently brighter than the REST OF
    # THE BAND at a given moment. Baseline must therefore be computed ACROSS
    # FREQUENCY at each time step, not from the channel's own time history --
    # a channel with a rock-steady RFI carrier (constant offset at every time
    # step) has near-zero variance across TIME, so comparing it to its own
    # history hides it completely (checked this against a synthetic
    # constant-offset test row before settling on cross-channel comparison;
    # the same-channel version scored it 0.0 -- invisible).
    col_med = np.median(S, axis=0, keepdims=True)  # "normal" level across the band, per time step
    col_mad = np.median(np.abs(S - col_med), axis=0, keepdims=True)
    col_mad = np.where(col_mad == 0, 1e-9, col_mad)
    band_outlier = S > (col_med + mad_threshold * col_mad)
    row_occupancy = band_outlier.mean(axis=1)  # per freq channel: how often does it stick out from the band
    horizontal_rfi_frac = float((row_occupancy > occupancy_threshold).mean())

    # Vertical RFI = many channels spiking simultaneously relative to THEIR OWN
    # history (a broadband instantaneous pulse -- lightning, switching noise).
    # Here the right baseline is each channel's own time history.
    row_med = np.median(S, axis=1, keepdims=True)
    row_mad = np.median(np.abs(S - row_med), axis=1, keepdims=True)
    row_mad = np.where(row_mad == 0, 1e-9, row_mad)
    self_outlier = S > (row_med + mad_threshold * row_mad)
    col_occupancy = self_outlier.mean(axis=0)  # per time column: fraction of channels spiking at once
    vertical_rfi_frac = float((col_occupancy > occupancy_threshold).mean())

    return {
        "horizontal_rfi_frac": horizontal_rfi_frac,
        "vertical_rfi_frac": vertical_rfi_frac,
        "row_occupancy": row_occupancy,
        "col_occupancy": col_occupancy,
    }
