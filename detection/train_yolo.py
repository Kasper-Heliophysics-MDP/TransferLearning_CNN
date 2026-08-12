"""
train_yolo.py

Phase 1 training: COCO-pretrained YOLOv8 -> solar radio burst detection on
15-minute e-Callisto windows. See TRAINING_PLAN.md.

Two stages, as planned: freeze the backbone and train the detection head first
(the COCO features are a starting point, and letting the whole network move on
a small, noisy dataset from the first step tends to wash them out), then
unfreeze and fine-tune end to end at a lower LR.

Usage:
    python detection/train_yolo.py <data.yaml> --name phase1 \
        [--model yolov8s.pt] [--freeze-epochs 10] [--epochs 60] [--imgsz 640]

    # smoke test: proves the whole path runs, not that it learns anything
    python detection/train_yolo.py <data.yaml> --name smoke \
        --freeze-epochs 1 --epochs 2 --model yolov8n.pt
"""

from __future__ import annotations

import argparse
import os

from ultralytics import YOLO

# Freezing 10 layers covers the YOLOv8 backbone and leaves neck + head
# trainable (ultralytics' own transfer-learning convention).
BACKBONE_LAYERS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="path to data.yaml from build_yolo_dataset.py")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--name", default="phase1")
    ap.add_argument("--project", default="detection/runs")
    ap.add_argument("--freeze-epochs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--rect", action="store_true",
                    help="rectangular batches: keep the images' own aspect ratio instead of "
                         "letterboxing to a square. Needed for the 640x1280 wide dataset, "
                         "otherwise ultralytics pads it back to 1280x1280 and burns 2x the "
                         "compute for identical content resolution.")
    ap.add_argument("--patience", type=int, default=100,
                    help="early-stop after this many epochs without val improvement. "
                         "The first Phase 1 run peaked ~3 epochs into stage 2 and then only "
                         "overfit (val loss rose monotonically for 57 more), so leaving this "
                         "at ultralytics' default of 100 just burns time on a small dataset.")
    args = ap.parse_args()

    # absolute, so ultralytics doesn't nest it under its own settings runs_dir
    args.project = os.path.abspath(args.project)

    # Spectrograms are not photographs: a burst's identity is its
    # time-frequency shape, so the geometric augmentations that help on COCO
    # actively destroy the label here. Vertical flip reverses the frequency
    # drift direction (the single most important Type II/III cue), horizontal
    # flip reverses time, and mosaic splices unrelated windows together along
    # the time axis. All off, deliberately.
    common = dict(data=args.data, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project=args.project, exist_ok=True,
                  patience=args.patience, rect=args.rect,
                  fliplr=0.0, flipud=0.0, mosaic=0.0, degrees=0.0,
                  shear=0.0, perspective=0.0, scale=0.0, translate=0.0)

    if args.freeze_epochs > 0:
        print(f"=== stage 1: backbone frozen, {args.freeze_epochs} epochs ===")
        model = YOLO(args.model)
        model.train(epochs=args.freeze_epochs, freeze=BACKBONE_LAYERS, lr0=1e-3,
                    name=f"{args.name}_frozen", **common)
        # Ask the trainer where it actually wrote, never reconstruct the path:
        # ultralytics prepends its own settings `runs_dir`/detect to a RELATIVE
        # `project`, so project="detection/runs" really lands in
        # "runs/detect/detection/runs/...". Constructing the path by hand
        # crashed stage 2 with FileNotFoundError after stage 1 had already
        # spent its epochs. `project` is made absolute above, which stops the
        # nesting, but reading save_dir is what makes this correct regardless.
        weights = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
        print(f"stage 1 weights: {weights}")
    else:
        weights = args.model

    print(f"=== stage 2: full fine-tune from {weights}, {args.epochs} epochs ===")
    model = YOLO(weights)
    results = model.train(epochs=args.epochs, lr0=1e-4, name=args.name, **common)
    print(results)


if __name__ == "__main__":
    main()
