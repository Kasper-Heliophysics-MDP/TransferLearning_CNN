"""
range_select.py

Drag-to-annotate: a Plotly spectrogram where dragging a horizontal range sets
the burst start/end, replacing the two number inputs as the primary way to
correct a burst's time range.

Why this exists (measured, not a UI preference):
The two number inputs made correcting the START and the END cost the same
keystrokes but carry very different urgency. The catalog's END is obviously
wrong -- 36% of events are zero-width, so the end box visibly sits on top of
the start -- and it got corrected 93% of the time. The catalog's START looks
like a plausible number, so it was silently accepted 38% of the time. The
consequence is systematic: across 281 corrections, boxes whose start was never
touched have a median width of 90s versus 30s for boxes whose start was moved,
because the untouched ones still carry the catalog's ~30s-early start (of the
174 events whose start WAS moved, 96% moved it later, median +30s).

A drag sets both edges in one gesture, so "I only fixed half of it" stops being
possible. That is the actual fix -- snapping and grid size were never the
problem (quantizing to 1s adds 0.29s of noise against a human jitter of ~7.8s).

Implementation notes:
- go.Image, not go.Heatmap: the window is 200x3600, and a heatmap of 720k
  points is painfully slow in the browser. Colour-mapping to RGB first makes it
  a single image trace.
- Values snap to SNAP_S (1s). Finer is false precision -- the underlying data
  is 0.25s per column and human repeatability is ~8s.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib import cm

SNAP_S = 1.0
MAX_COLS = 1800  # display downsample; 3600 native columns is more than the browser needs


def _to_rgb(raw: np.ndarray) -> np.ndarray:
    """(freq, time) -> uint8 RGB, per-row median subtracted then percentile-clipped.

    Same treatment the exported training PNGs get, so what is dragged on is what
    the model will see -- not a differently-scaled picture."""
    x = raw.astype(np.float32)
    x -= np.median(x, axis=1, keepdims=True)
    lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (cm.viridis(x)[:, :, :3] * 255).astype(np.uint8)


def range_selector(raw: np.ndarray, sample_interval_s: float, freq_mhz, key: str,
                   start_s: float, end_s: float, height: int = 460) -> tuple[float, float] | None:
    """Draw the spectrogram with the current range marked; return a new
    (start_s, end_s) if the user dragged one, else None.

    `start_s`/`end_s` are drawn as vertical lines plus a shaded band so the
    current answer stays visible while dragging a new one.
    """
    n_freq, n_t = raw.shape
    step = max(1, n_t // MAX_COLS)
    rgb = _to_rgb(raw[:, ::step])
    dx = sample_interval_s * step
    total_s = n_t * sample_interval_s

    fig = go.Figure(go.Image(z=rgb, x0=dx / 2, dx=dx, y0=0, dy=1))
    fig.add_vrect(x0=start_s, x1=max(end_s, start_s + 0.5), fillcolor="red",
                  opacity=0.18, line_width=0)
    for xv, nm in ((start_s, "start"), (end_s, "end")):
        fig.add_vline(x=xv, line=dict(color="red", width=2))
    if freq_mhz is not None and len(freq_mhz) == n_freq:
        ticks = np.linspace(0, n_freq - 1, 5).astype(int)
        fig.update_yaxes(tickmode="array", tickvals=ticks,
                         ticktext=[f"{freq_mhz[i]:.0f}" for i in ticks], title="MHz")
    fig.update_xaxes(range=[0, total_s], title="秒(从窗口开始算)")
    # plotly.js locks an Image trace to square pixels at render time (the
    # constraint is added client-side, so the Python-side scaleanchor reads as
    # None and looks fine). A 200x480 window then renders as a narrow sliver in
    # a wide container, which is useless for dragging a time range precisely.
    # scaleanchor=False is the documented way to remove a DEFAULT constraint,
    # letting the spectrogram stretch to the full container width.
    fig.update_yaxes(scaleanchor=False, constrain="range")
    fig.update_layout(dragmode="select", selectdirection="h", height=height,
                      margin=dict(l=55, r=10, t=10, b=45), showlegend=False,
                      xaxis=dict(constrain="range"))

    ev = st.plotly_chart(fig, key=key, on_select="rerun", selection_mode="box",
                         config={"displayModeBar": False}, use_container_width=True)

    box = (ev or {}).get("selection", {}).get("box") or []
    if not box:
        return None
    xs = box[0].get("x") or []
    if len(xs) < 2:
        return None
    a, b = sorted(float(v) for v in xs[:2])
    a = float(np.clip(round(a / SNAP_S) * SNAP_S, 0.0, total_s))
    b = float(np.clip(round(b / SNAP_S) * SNAP_S, 0.0, total_s))
    if b - a < SNAP_S:            # a click rather than a drag -- ignore, don't
        return None               # silently collapse the range to nothing
    return a, b
