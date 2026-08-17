"""
zero_shot_transfer.py

Run the e-Callisto-trained detector on own-station windows, unchanged, and
measure what survives the domain gap. No training, no fine-tuning -- this is the
cheap measurement that says whether transfer is worth building for.

THE DOMAIN GAP, CONCRETELY
--------------------------
                        own station        training data
    band                15.996-24.004      15.0-86.625 MHz
    channels            411                200
    channel spacing     0.0195 MHz         0.358 MHz        (18x coarser)
    sample interval     0.1 s              0.25 s

The time axis is not a problem: both are 15-minute windows resized to the same
width, so seconds-per-pixel matches by construction. The frequency axis is the
whole problem, and there is no way to satisfy both of these at once:

  * keep the physical scale, and the own station's entire 8 MHz band occupies
    72 of 640 pixel rows -- the model sees a thin strip in a blank frame, which
    nothing in COCO or in training resembles;
  * fill the frame, and 8 MHz is stretched over the height that 71.6 MHz
    occupied in training, so a given drift rate lands on a pixel slope 8.9x
    steeper than the model learned. Drift slope is the primary Type II/III cue,
    so this is not a cosmetic difference.

Both are rendered here rather than argued about. `band` is physically faithful
and out of distribution; `stretch` is in-distribution-looking and physically
wrong. Which one degrades less is a fact about the model, not something
derivable from first principles, and it decides what the Phase 3 design has to
be: if `band` wins, the path is padding/masking and the training data can stay
as it is; if `stretch` wins, the model is keying on appearance rather than
physical slope, and the honest fix is retraining on a band-matched crop.

Detection is scored on 1-D time overlap, the same convention evaluate.py uses:
every box is full height, so the frequency axis carries no positional
information and IoU is purely temporal.

Usage:
    python transfer/zero_shot_transfer.py <weights> [--out-dir transfer/out]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "detection"))
from build_yolo_dataset import render_png     # noqa: E402
from evaluate import iou_1d, maxconf_collapse  # noqa: E402

# the band the detector was trained on; a rendered row maps linearly onto it
TRAIN_FMAX, TRAIN_FMIN = 86.625, 15.0
SIZE = 640


def render_band(arr: np.ndarray, fmhz: np.ndarray) -> np.ndarray:
    """Own-station band placed at its true position in the training band's
    frame. Everything outside it is filled with the band's own median level, so
    the empty region reads as featureless background rather than as a hard
    black edge the detector could latch onto."""
    hi, lo = float(fmhz.max()), float(fmhz.min())
    r0 = int(round((TRAIN_FMAX - hi) / (TRAIN_FMAX - TRAIN_FMIN) * SIZE))
    r1 = int(round((TRAIN_FMAX - lo) / (TRAIN_FMAX - TRAIN_FMIN) * SIZE))
    r0, r1 = max(0, r0), min(SIZE, max(r1, r0 + 1))
    band = render_png(arr, (r1 - r0, SIZE))
    canvas = np.full((SIZE, SIZE), int(np.median(band)), np.uint8)
    canvas[r0:r1] = band
    return canvas


def render_stretch(arr: np.ndarray, fmhz: np.ndarray) -> np.ndarray:
    """Own-station band stretched to the full frame height -- what you get by
    treating the two sources as interchangeable images."""
    return render_png(arr, (SIZE, SIZE))


MODES = {"band": render_band, "stretch": render_stretch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--window-dir", default="data/burst_data/own_windows")
    ap.add_argument("--out-dir", default="transfer/out")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--min-overlap", type=float, default=0.1)
    args = ap.parse_args()

    win = pd.read_csv(os.path.join(args.window_dir, "windows.csv")).set_index("file_name")
    box = pd.read_csv(os.path.join(args.window_dir, "boxes.csv"))
    gt: dict[str, list] = {}
    for b in box.itertuples():
        n = int(win.loc[b.file_name].n_cols)
        gt.setdefault(b.file_name, []).append((b.box_start_col / n, b.box_end_col / n))

    model = YOLO(args.weights)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(win)} own-station windows, {len(box)} boxes\n")

    rows = []
    for mode, fn in MODES.items():
        imgs, names = [], []
        for f in win.index:
            arr = np.load(os.path.join(args.window_dir, f))
            fq = np.load(os.path.join(args.window_dir, "freq-" + f[:-4] + ".npy"))
            imgs.append(cv2.cvtColor(fn(arr, fq), cv2.COLOR_GRAY2BGR))
            names.append(f)
        cv2.imwrite(os.path.join(args.out_dir, f"sample_{mode}.png"),
                    cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY))

        preds: dict[str, list] = {}
        for i in range(0, len(imgs), 16):
            for nm, res in zip(names[i:i + 16],
                               model.predict(imgs[i:i + 16], conf=args.conf,
                                             imgsz=SIZE, verbose=False)):
                preds[nm] = maxconf_collapse(
                    [(0, float(b.conf), *b.xyxyn[0].tolist()[::2]) for b in res.boxes],
                    args.min_overlap)

        n_gt = hit = n_pred = 0
        errs, confs = [], []
        for f, truths in gt.items():
            p = preds.get(f, [])
            n_gt += len(truths); n_pred += len(p)
            confs += [c for _, c, _, _ in p]
            for t0, t1 in truths:
                best = None
                for _, c, x0, x1 in p:
                    if iou_1d(x0, x1, t0, t1) >= args.min_overlap:
                        d = abs((x0 + x1) / 2 - (t0 + t1) / 2)
                        if best is None or d < best:
                            best = d
                if best is not None:
                    hit += 1
                    errs.append(best * 900.0)
        rows.append(dict(mode=mode, detected=f"{hit}/{n_gt}", rate=hit / max(n_gt, 1),
                         preds=n_pred,
                         dt_median=float(np.median(errs)) if errs else float("nan"),
                         conf_max=max(confs) if confs else 0.0,
                         conf_median=float(np.median(confs)) if confs else 0.0))

    r = pd.DataFrame(rows)
    print(f"{'mode':10s} {'detected':>10s} {'rate':>7s} {'preds':>7s} "
          f"{'|dt| med':>9s} {'conf max':>9s} {'conf med':>9s}")
    for x in r.itertuples():
        print(f"{x.mode:10s} {x.detected:>10s} {x.rate*100:6.0f}% {x.preds:7d} "
              f"{x.dt_median:8.1f}s {x.conf_max:9.3f} {x.conf_median:9.3f}")
    r.to_csv(os.path.join(args.out_dir, "zero_shot.csv"), index=False)
    print(f"\nsample renders + zero_shot.csv in {args.out_dir}/")


if __name__ == "__main__":
    main()
