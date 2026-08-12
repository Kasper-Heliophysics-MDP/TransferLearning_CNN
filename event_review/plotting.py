"""
plotting.py

Shared imshow rendering for the review UI. Two lessons from this session's
denoising work, both directly load-bearing here:

1. Raw (intensity) and cleaned (signed residual around 0) are different kinds
   of data -- each gets its own percentile-based scale (viridis / RdBu_r),
   matching the convention already established in ecallisto_grabber/preview.py
   and every comparison image produced this session.
2. When comparing MULTIPLE cleaned results against each other (default
   production cleaning vs. a live-reprocessed attempt with different
   parameters), those two MUST share one color scale, not each auto-scale
   independently -- independent percentile scaling repeatedly produced false
   "this version looks cleaner" impressions this session (the v0 edge-spike
   case, the ramp-sweep case) when the true difference was the opposite or
   nonexistent. `cleaned_scale()` is exposed separately from the plotting
   calls specifically so a caller can compute it once from one result and
   reuse it for a second.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def raw_scale(raw: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(raw, [2, 98]))


def cleaned_scale(cleaned: np.ndarray) -> float:
    return float(np.percentile(np.abs(cleaned), 99))


def render_pair(
    raw: np.ndarray,
    cleaned: np.ndarray,
    sample_interval_s: float,
    freq_mhz: np.ndarray | None,
    event_indices: tuple[int, int] | None,
    protect_indices: tuple[int, int] | None,
    cleaned_lim: float | None = None,
    titles: tuple[str, str] = ("raw", "cleaned"),
) -> plt.Figure:
    """
    One figure, two side-by-side panels: raw (viridis) | cleaned (RdBu_r).

    `event_indices`: the catalog's own (start_col, end_col) for the burst,
    drawn as a narrow red/black shaded band on both panels.
    `protect_indices`: the actual protected window after applying
    known_burst_margin, drawn as a wider, lighter shaded band -- the gap
    between the two bands is exactly where a real signal can fall outside
    protection and get suppressed (the SWISS-Landschlacht case this tool
    exists to catch).
    `cleaned_lim`: pass in a value (from a previous cleaned_scale() call) to
    force this panel onto that scale instead of computing its own -- use
    when rendering a reprocessed result that should be visually comparable
    to the default one already shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    vmin, vmax = raw_scale(raw)
    extent = None
    if freq_mhz is not None and len(freq_mhz) == raw.shape[0]:
        extent = [0, raw.shape[1] * sample_interval_s, freq_mhz.min(), freq_mhz.max()]
    axes[0].imshow(raw, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title(titles[0], fontsize=10)

    lim = cleaned_lim if cleaned_lim is not None else cleaned_scale(cleaned)
    axes[1].imshow(cleaned, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=extent)
    axes[1].set_title(titles[1], fontsize=10)

    for ax in axes:
        _mark_windows(ax, event_indices, protect_indices, raw.shape[1], sample_interval_s, bool(extent))
        ax.set_xlabel("s" if extent else "col", fontsize=8)
        if extent:
            ax.set_ylabel("MHz", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


def _mark_windows(ax, event_indices, protect_indices, n_cols, sample_interval_s, in_seconds):
    def to_x(col):
        return col * sample_interval_s if in_seconds else col

    if protect_indices is not None:
        lo, hi = protect_indices
        ax.axvspan(to_x(max(0, lo)), to_x(min(n_cols, hi)), color="black", alpha=0.08, zorder=0)
    if event_indices is not None:
        lo, hi = event_indices
        ax.axvline(to_x(lo), color="red", lw=1.0, alpha=0.7, zorder=1)
        ax.axvline(to_x(hi), color="red", lw=1.0, alpha=0.7, zorder=1)
