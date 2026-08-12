"""
SumThreshold-based RFI cleaning (Offringa et al.), adapted for solar radio
burst dynamic spectra.

Pipeline: robust 2D background fit -> per-channel normalized residual ->
continuous 2D "how burst-like" weight -> multi-scale SumThreshold ratio scan
(scaled by the burst weight throughout, so a real burst can't flag itself and
contributes no sharp edges to the scan) -> graduated suppression (no hard
0%/100% switch anywhere in the pipeline).

`sumthreshold_cleaning_wrapper` at the bottom matches
`denoise_new.advanced_rfi_cleaning_wrapper`'s call signature so it can be
swapped in directly in slicing_utils_new.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, uniform_filter1d


def _fill_nonfinite(data: np.ndarray) -> np.ndarray:
    """
    Replace NaN/Inf via per-channel linear interpolation along time, before
    any statistics are computed.

    Confirmed on a real own-station file: a single dropped time-sample (1
    row out of 416546) left NaN in 330 of 411 frequency channels for that
    one instant. Every median/MAD in this pipeline is computed per-channel
    across the FULL time axis (np.median etc. are not NaN-aware), so left
    alone, that one bad sample would poison those channels' background fit
    and normalization for the ENTIRE file -- confirmed: every window sliced
    from that file came out NaN, regardless of how far its burst window was
    from the actual bad sample. Interpolating first contains the damage to
    just the handful of originally-bad pixels.
    """
    if np.isfinite(data).all():
        return data
    data = data.copy()
    t = np.arange(data.shape[1])
    for row in data:
        bad = ~np.isfinite(row)
        if not bad.any():
            continue
        good = ~bad
        row[bad] = np.interp(t[bad], t[good], row[good]) if good.any() else 0.0
    return data


def robust_smooth_background(
    data: np.ndarray,
    freq_win: int = 9,
    time_win: int = 121,
    n_iter: int = 4,
    sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively fit a smooth 2D background, excluding outliers (bursts AND
    RFI, both are "not smooth") from the fit each round so neither pulls the
    background estimate toward itself.

    The outlier threshold is per-row (each frequency channel judged against
    its own median/MAD), not one global scalar for the whole spectrogram: a
    burst is rarely extreme relative to the WHOLE image's noise floor, only
    relative to its own channel's, so a global threshold silently keeps most
    of a burst's pixels in the fit and biases the background upward near it.
    """
    mask = np.zeros_like(data, dtype=bool)
    background = None
    for _ in range(n_iter):
        valid = (~mask).astype(np.float32)
        filled = np.where(mask, 0.0, data)
        num = uniform_filter(filled, size=(freq_win, time_win), mode="reflect")
        den = uniform_filter(valid, size=(freq_win, time_win), mode="reflect")
        background = num / np.maximum(den, 1e-6)

        residual = data - background
        row_med = np.median(residual, axis=1, keepdims=True)
        row_mad = np.median(np.abs(residual - row_med), axis=1, keepdims=True) * 1.4826 + 1e-9
        mask = np.abs(residual - row_med) > sigma * row_mad

    return background, mask


def per_row_normalize(residual: np.ndarray) -> np.ndarray:
    """
    Normalize each frequency row by its own MAD, so one global sigma
    threshold downstream behaves like a per-channel threshold (channels
    differ in noise level, MHz to MHz). MAD's 50%-breakdown robustness makes
    excluding burst columns first unnecessary -- a real burst is a small
    minority of the timeline.
    """
    row_med = np.median(residual, axis=1, keepdims=True)
    row_mad = np.median(np.abs(residual - row_med), axis=1, keepdims=True) * 1.4826 + 1e-9
    # np.median always returns float64 regardless of input dtype, which would
    # otherwise silently upcast this (large, freq x time) result -- cast back
    # explicitly rather than let every downstream array double in memory
    return ((residual - row_med) / row_mad).astype(residual.dtype)


def burst_weight_2d(
    residual_norm: np.ndarray,
    freq_smooth: int = 15,
    time_smooth: int = 21,
    sigma: float = 2.0,
    sharpness: float = 1.0,
) -> np.ndarray:
    """
    Continuous [0,1] "how burst-like is this pixel" weight. Genuinely 2D
    (smoothed in both frequency and time before thresholding) so it follows
    the real shape of an elevated region -- including a drifting Type III's
    diagonal track, which only occupies a narrow frequency band at any given
    time -- rather than a column-only decision that would protect (or fail
    to protect) an entire time-slice uniformly across every channel.
    """
    smoothed = uniform_filter(np.abs(residual_norm), size=(freq_smooth, time_smooth), mode="reflect")
    med = np.median(smoothed)
    mad = np.median(np.abs(smoothed - med)) * 1.4826 + 1e-9
    z = (smoothed - med) / mad
    weight = (1.0 / (1.0 + np.exp(-(z - sigma) * sharpness))).astype(residual_norm.dtype)
    if np.mean(weight > 0.5) > 0.5:
        # degenerate case: "most of the spectrogram looks like burst" means
        # the threshold is miscalibrated for this data (e.g. a globally
        # noisy station), not that there's really a burst everywhere
        weight = np.zeros_like(weight)
    return weight


def persistent_row_correction(residual, freq_win=9, sigma=3.0, sharpness=1.0):
    """
    Per-row (n_freq,) correction for channels persistently offset from their
    neighbors for the ENTIRE timeline, plus a continuous [0,1] confidence
    weight for how strongly each row triggered it.

    Why needed in addition to per_row_normalize: a row that's elevated for
    its whole length has that elevation baked into ITS OWN median/MAD, so
    normalizing against itself can't see it as anomalous -- confirmed on
    real data, a persistently-contaminated row's raw residual sat at
    mean~888 (the smooth 2D background fit didn't fully remove it either,
    since freq_win x time_win smoothing isn't primarily aimed at a single
    channel's constant DC-like offset) yet its OWN residual_norm showed
    essentially no outliers (std=0.96, 0.1% beyond 3 sigma) -- the
    elevation IS that row's "normal". Comparing each row's typical level
    against a smoothed CROSS-ROW (neighboring channel) reference instead --
    same idea as FPBS.analyze_persistent_bands earlier in this project,
    applied here as this pipeline's own row-level check -- catches what
    per-row self-normalization structurally cannot.

    Returns (correction, weight), both shape (n_freq,): subtract
    `weight * correction` (broadcast across time) from residual to flatten
    out the persistent excess before the rest of the pipeline runs.
    """
    from scipy.ndimage import median_filter
    row_level = np.median(residual, axis=1)
    local_expected = median_filter(row_level, size=freq_win, mode="reflect")
    dev = row_level - local_expected
    mad = np.median(np.abs(dev)) * 1.4826 + 1e-9
    z = np.abs(dev) / mad
    weight = 1.0 / (1.0 + np.exp(-(z - sigma) * sharpness))
    return dev.astype(residual.dtype), weight.astype(residual.dtype)


def vertical_rfi_weight(
    residual_norm: np.ndarray,
    coverage_threshold: float = 0.1,
    sigma: float = 3.0,
    sharpness: float = 20.0,
) -> np.ndarray:
    """
    Continuous [0,1] "how much does this time instant look like
    instantaneous broadband RFI" signal, based on CROSS-FREQUENCY coverage
    (what fraction of channels are simultaneously anomalous right now) --
    deliberately NOT time-smoothed, since coverage is specifically about
    "all at once".

    Why this is needed in addition to burst_weight_2d: magnitude alone
    can't separate a real burst from strong instantaneous broadband
    interference (e.g. electrical switching noise) -- on real data, a
    vertical RFI band's residual peaked far higher than a real burst's own
    region (~14 vs ~4.2 in per-row-normalized units), so a magnitude-based
    detector protects the RFI and leaves the real burst exposed to
    suppression, exactly backwards. Coverage is the distinguishing signal
    instead: genuine broadband interference can hit every channel in the
    receiver bandpass simultaneously; a real solar burst, even broadband,
    rarely does -- propagation/dispersion leaves some frequency structure
    -- so high simultaneous coverage is treated as RFI regardless of how
    strong it is.

    Takes `residual_norm` (already per-channel normalized, see
    per_row_normalize), NOT raw residual -- first attempt recomputed a
    fresh per-column (cross-frequency) median/MAD here instead, which is
    the wrong baseline: if RFI raises every channel by roughly the same
    amount, the column's own statistics shift right along with it, so
    nothing looks like an outlier relative to ITSELF even though the whole
    column is anomalous relative to what's normal for each channel. Each
    channel needs to be judged against its OWN typical level first (already
    done by per_row_normalize) -- only then does asking "how many channels
    are simultaneously elevated" mean anything. Same two-step logic
    denoise_new.py already uses (per-channel MAD-normalize in step1, THEN
    per-instant coverage in step2), just applied to this pipeline's own
    per-row-normalized residual instead of recomputing it separately.

    `coverage_threshold`/`sigma` defaults match denoise_new.py's
    already-validated vertical-RFI thresholds (Type 3 params); sigmoid'd
    instead of a hard cutoff to stay consistent with the rest of this
    pipeline's soft-switch suppression.
    """
    anomalous = residual_norm > sigma
    coverage = anomalous.mean(axis=0)
    return (1.0 / (1.0 + np.exp(-(coverage - coverage_threshold) * sharpness))).astype(residual_norm.dtype)


def sumthreshold_axis_ratio(
    residual: np.ndarray,
    axis: int,
    base_threshold: float,
    window_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    rho: float = 1.5,
    protect_weight: np.ndarray | None = None,
) -> np.ndarray:
    """
    Multi-scale combinatorial threshold along one axis (Offringa et al.),
    returning a continuous ratio = |windowed average| / that scale's
    threshold (max across scales) rather than a binary flag, so suppression
    downstream can be graduated instead of all-or-nothing.

    axis=1 (time): narrowband/persistent interference (elongated in time at
        one frequency). axis=0 (frequency): broadband/instantaneous
        interference (elongated in frequency at one time).

    `protect_weight`: continuous [0,1] burst weight (see burst_weight_2d).
    Scaling `work` by it BEFORE the windowed averages are computed (not just
    masking the final result) keeps a burst from contaminating any window
    that partially overlaps it, without introducing a sharp edge at the
    boundary the way a hard mask would.
    """
    ratio = np.zeros(residual.shape, dtype=residual.dtype)
    work = residual.copy()
    if protect_weight is not None:
        work = work * (1.0 - protect_weight)

    for M in window_sizes:
        thresh = float(base_threshold / (rho ** np.log2(M)))  # np.log2 returns an np.float64 scalar, which (unlike a Python float) is a "strong" dtype under NEP50 and would upcast every float32 array it touches
        if M == 1:
            avg = work
        else:
            avg = uniform_filter1d(work, size=M, axis=axis, mode="reflect")
        this_ratio = np.abs(avg) / thresh
        if M > 1:
            from scipy.ndimage import maximum_filter1d
            this_ratio = maximum_filter1d(this_ratio, size=M, axis=axis, mode="reflect")
        ratio = np.maximum(ratio, this_ratio)
        newly = this_ratio > 1.0
        work = np.where(newly, np.sign(work) * thresh, work)

    if protect_weight is not None:
        ratio = ratio * (1.0 - protect_weight)
    return ratio


def clean(
    data: np.ndarray,
    freq_win: int = 9,
    time_win: int = 121,
    bg_iter: int = 4,
    bg_sigma: float = 3.0,
    burst_sigma: float = 2.0,
    burst_sharpness: float = 1.0,
    base_threshold_sigma: float = 6.0,
    window_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    rho: float = 1.5,
    suppression_ramp: float = 2.0,
    vrfi_coverage_threshold: float = 0.1,
    vrfi_sigma: float = 3.0,
    known_burst_weight: np.ndarray | None = None,
    return_intermediates: bool = True,
) -> dict:
    """
    Full pipeline: robust background -> residual -> per-channel
    normalization -> continuous burst weight -> SumThreshold ratio scan
    (time axis + frequency axis) -> graduated suppression.

    `suppression_ramp`: suppression fraction = clip(ratio / suppression_ramp,
    0, 1) -- a pixel exactly at the classic SumThreshold flag threshold
    (ratio=1) gets 50% suppressed, not switched instantly from 0% to 100%;
    full suppression only applies once a pixel is clearly past threshold.

    `vrfi_coverage_threshold`/`vrfi_sigma`: passed straight through to
    vertical_rfi_weight() (see its docstring) -- exposed here mainly so
    event_review's parameter panel can adjust them per-event for cases with
    heavy residual instantaneous-broadband RFI the default 0.1/3.0 doesn't
    fully catch. Defaults match vertical_rfi_weight()'s own, so callers that
    don't pass these get identical behavior to before this was exposed.

    `known_burst_weight`: optional [0,1] array (same shape as `data`) of
    ground-truth burst protection, applied AFTER the vertical-RFI veto below
    (via elementwise max) so a labeled burst is always protected even if it
    happens to also look RFI-like. Pass this whenever the burst's real time
    range is already known (e.g. from a labeled catalog) instead of trusting
    auto-detection alone.

    `return_intermediates`: default True returns a dict with every stage
    (background, residual, masks, ...) so results can be inspected, not just
    the final `cleaned` array -- this is what makes debugging/plotting
    possible, so it stays the default. But holding every full-size (n_freq x
    n_time) array alive simultaneously until that dict is built is exactly
    what pushed real own-station files (hundreds of thousands of columns) to
    12-13GB peak RSS and got OOM-killed on a 15GB no-swap machine -- pass
    False (as the batch-processing wrapper below does) to drop each array as
    soon as later stages stop needing it, in exchange for only getting
    `cleaned` back.
    """
    data = _fill_nonfinite(data)
    background, _ = robust_smooth_background(data, freq_win, time_win, bg_iter, bg_sigma)
    residual = data - background
    if not return_intermediates:
        del background

    # flatten out any per-row persistent excess the smooth 2D background fit
    # didn't catch (see persistent_row_correction docstring) BEFORE
    # per-row normalization -- otherwise a persistently-contaminated row's
    # own MAD absorbs the excess and the rest of the pipeline never sees it
    row_dev, row_weight = persistent_row_correction(residual)
    residual = residual - (row_weight * row_dev)[:, np.newaxis]

    residual_norm = per_row_normalize(residual)

    vrfi_weight = vertical_rfi_weight(residual_norm, coverage_threshold=vrfi_coverage_threshold, sigma=vrfi_sigma)
    vrfi_weight_2d = np.broadcast_to(vrfi_weight, residual.shape)

    burst_weight = burst_weight_2d(residual_norm, sigma=burst_sigma, sharpness=burst_sharpness)
    # can't be treated as burst if it looks like instantaneous broadband RFI,
    # no matter how strong -- see vertical_rfi_weight docstring
    burst_weight = burst_weight * (1.0 - vrfi_weight_2d)
    if known_burst_weight is not None:
        burst_weight = np.maximum(burst_weight, known_burst_weight)
    protect = burst_weight > 0.5

    ratio_time = sumthreshold_axis_ratio(residual_norm, axis=1, base_threshold=base_threshold_sigma,
                                          window_sizes=window_sizes, rho=rho, protect_weight=burst_weight)
    ratio_freq = sumthreshold_axis_ratio(residual_norm, axis=0, base_threshold=base_threshold_sigma,
                                          window_sizes=window_sizes, rho=rho, protect_weight=burst_weight)
    ratio = np.maximum(ratio_time, ratio_freq)
    del ratio_time, ratio_freq  # never part of the returned dict either way, just locals the frame would otherwise pin until return
    if not return_intermediates:
        del residual_norm, burst_weight

    frac = np.clip(ratio / suppression_ramp, 0.0, 1.0)
    # vrfi_weight is independent, strong evidence (cross-frequency coverage,
    # not magnitude) that a pixel is broadband RFI -- once confident, don't
    # let the ratio-based ramp hold suppression back. Measured on real data:
    # a confirmed vertical RFI event (vrfi_weight~0.998) had raw |residual|
    # ~3125, and ratio-only suppression (84% at that event's ratio~1.85)
    # still left ~376 behind -- far above typical burst/background
    # magnitude, so the event stayed visually dominant despite an 8x
    # reduction. Suppressing at least as much as our RFI confidence fixes
    # that without weakening suppression anywhere ratio alone already
    # decided (max, not replace).
    # exclude protected (known-burst or confidently-auto-detected-burst)
    # pixels from this floor -- a real labeled burst can still trigger a
    # high vrfi_weight on its own (a strong, fast event can look broadband
    # instantaneous too), and ground truth must win over that heuristic
    frac = np.where(protect, frac, np.maximum(frac, vrfi_weight_2d))
    flagged = (ratio > 1.0) & ~protect
    cleaned = residual - frac * residual

    if not return_intermediates:
        return dict(cleaned=cleaned)

    return dict(
        background=background,
        residual=residual,
        residual_norm=residual_norm,
        protect_cols=protect,
        burst_weight=burst_weight,
        vertical_rfi_weight=vrfi_weight,
        ratio=ratio,
        suppression_frac=frac,
        flagged=flagged,
        cleaned=cleaned,
    )


# Method -> clean() parameter overrides. "fast" cuts the background fit's
# iteration count and the scan's window-size ladder (the two costs that
# actually scale with effort); "conservative" widens the suppression ramp
# and raises the flag threshold so only clearly-confident RFI gets touched.
_METHOD_PARAMS = {
    "comprehensive": dict(),
    "fast": dict(bg_iter=2, window_sizes=(1, 4, 16)),
    "conservative": dict(base_threshold_sigma=9.0, suppression_ramp=3.0),
}


# time_win=241 was tuned/validated on eCallisto test data sampled at 0.25s
# (i.e. a ~60s background-fit window). All of this pipeline's other
# thresholds are in sigma/ratio units and don't depend on sample rate, but
# time_win is a raw sample count -- at a finer sampling interval the same
# sample count covers a shorter physical duration, which (confirmed on a
# real own-station file, sampled at 0.1s) was short enough that a real
# burst no longer got a wide enough background-fit window to be told apart
# from the interference around it, and came out suppressed instead of
# preserved. Scale by the sampling interval instead of hardcoding a raw
# sample count so the EFFECTIVE physical window stays ~60s regardless of
# source.
_REFERENCE_TIME_WIN = 241
_REFERENCE_SAMPLE_INTERVAL_S = 0.25


def sumthreshold_cleaning_wrapper(
    spectral_data,
    burst_start_idx: int | None = None,
    burst_end_idx: int | None = None,
    burst_type: int | None = None,
    method: str = "comprehensive",
    known_burst_margin: int = 10,
    sample_interval_s: float = 0.1,
):
    """
    Drop-in replacement for denoise_new.advanced_rfi_cleaning_wrapper --
    same call signature, same (time, frequency) input/output orientation.

    `burst_type` is accepted for interface compatibility but unused: unlike
    the old cleaner's type-specific thresholds, burst_weight_2d adapts to
    the actual data rather than a burst-type lookup table.

    `burst_start_idx`/`burst_end_idx` (time-axis positions, from the
    catalog) are treated as ground truth and force-protected across every
    frequency channel -- widened by `known_burst_margin` samples on each
    side since catalog times are human-curated approximations, not exact
    pixel boundaries. Passing these is strongly recommended: auto-detection
    alone (burst_weight_2d) can be outcompeted by strong broadband RFI (see
    vertical_rfi_weight docstring) and isn't a substitute for a known label.

    `sample_interval_s`: this station's own CSVs are 0.1s/sample by default
    (matches BurstFixedWindowSlicer's hardcoded sampling_interval) -- pass
    the real value if a given file differs.
    """
    is_dataframe = isinstance(spectral_data, pd.DataFrame)
    arr = spectral_data.values if is_dataframe else np.asarray(spectral_data)
    # input is (time, frequency); this pipeline works in (frequency, time)
    data = arr.T.astype(np.float32)
    del arr  # real own-station files run ~500k+ rows; the float64 copy above is ~2x `data`'s size and unneeded past this line

    known_burst_weight = None
    if burst_start_idx is not None and burst_end_idx is not None:
        lo = max(0, burst_start_idx - known_burst_margin)
        hi = min(data.shape[1], burst_end_idx + known_burst_margin)
        known_burst_weight = np.zeros(data.shape[1], dtype=data.dtype)
        known_burst_weight[lo:hi] = 1.0
        known_burst_weight = np.broadcast_to(known_burst_weight, data.shape)

    time_win = round(_REFERENCE_TIME_WIN * _REFERENCE_SAMPLE_INTERVAL_S / sample_interval_s)
    time_win = max(time_win // 2 * 2 + 1, 3)  # keep it odd, same convention as the tuned reference

    # REVERTED: window_sizes was widened to (1..128) hoping to cover a wide
    # vertical RFI event's tapering edges, but the effective per-scale
    # threshold shrinks fast (thresh = base_threshold / rho**log2(M) --
    # ~0.7 sigma at M=128), making the ratio scan pick up any broad,
    # moderately-elevated structure as flaggable -- including a real burst
    # wherever burst_weight_2d's own protection is imperfect. Measured on
    # real data: widening to 128 increased a real burst's own suppression
    # (0.25 -> 0.37 mean frac in its region) while achieving the EXACT same
    # vertical-RFI suppression as the default ladder alone (|cleaned| mean
    # 1.2072 vs 1.2074) -- vertical_rfi_weight's suppression floor (below)
    # already fully handles wide vertical RFI on its own, so the wider
    # ladder bought nothing and cost real-signal fidelity.
    params = dict(time_win=time_win, base_threshold_sigma=12.0, burst_sigma=2.0)
    params.update(_METHOD_PARAMS.get(method, {}))
    # return_intermediates=False: batch processing only needs `cleaned`, and
    # keeping every intermediate array alive is what pushes real files to
    # 12-13GB peak RSS (see clean()'s docstring) -- fine on a small test crop
    # but OOM-kills on a 15GB no-swap machine for real, full-length files
    result = clean(data, known_burst_weight=known_burst_weight, return_intermediates=False, **params)

    cleaned = result["cleaned"].T  # back to (time, frequency)
    return cleaned
