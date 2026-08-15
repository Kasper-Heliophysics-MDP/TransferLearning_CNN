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
    ap.add_argument("--workers", type=int, default=8,
                    help="dataloader workers. This box has 4 cores, so ultralytics' default of 8 "
                         "is already oversubscribed; drop it to 2-3 when something else needs CPU "
                         "(the ASSA scrape decodes FITS between downloads). The GPU is the "
                         "bottleneck here anyway -- fewer workers costs little.")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed. Needed to measure run-to-run variance: with only "
                         "169 val boxes, an eval flag that should be neutral already moved "
                         "AP50 by 15%%, which is the same size as the gaps between the four "
                         "Phase 1 arms -- so the noise floor has to be measured before any "
                         "further comparison means anything.")
    ap.add_argument("--rect", action="store_true",
                    help="rectangular batches: keep the images' own aspect ratio instead of "
                         "letterboxing to a square. Needed for the 640x1280 wide dataset, "
                         "otherwise ultralytics pads it back to 1280x1280 and burns 2x the "
                         "compute for identical content resolution.")
    ap.add_argument("--optimizer", default="auto",
                    help="ultralytics optimizer. LEAVE THIS AT 'auto' AND --lr0-* DOES NOTHING: "
                         "auto picks the optimizer AND overrides lr0 from the iteration count, so "
                         "the 2026-08-14 batch silently ran every arm at AdamW(lr=0.002) and the "
                         "two LR arms came back bit-identical to the control. Set an explicit "
                         "optimizer (AdamW/SGD) for any run whose learning rate is supposed to "
                         "mean something. The effective LR is printed at startup either way.")
    ap.add_argument("--lr0-stage1", type=float, default=1e-3,
                    help="initial LR while the backbone is frozen. Higher than stage 2 on "
                         "purpose: only the neck+head move, and they start from random-ish "
                         "detection weights on a 1-class problem COCO never saw.")
    ap.add_argument("--lr0-stage2", type=float, default=1e-4,
                    help="initial LR for the full fine-tune. Both LRs were hard-coded at these "
                         "values through every experiment so far, i.e. the search space has "
                         "never been entered -- they are a starting guess, not a tuned result.")
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
    if args.optimizer == "auto" and (args.lr0_stage1 != 1e-3 or args.lr0_stage2 != 1e-4):
        print("!! WARNING: --lr0-* was set but --optimizer is 'auto', which overrides lr0.\n"
              "!! This run will IGNORE the learning rate you asked for. Pass --optimizer AdamW\n"
              "!! (or SGD) to make it take effect. See the --optimizer help text.")

    common = dict(data=args.data, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project=args.project, exist_ok=True,
                  patience=args.patience, rect=args.rect, seed=args.seed,
                  optimizer=args.optimizer, workers=args.workers,
                  fliplr=0.0, flipud=0.0, mosaic=0.0, degrees=0.0,
                  shear=0.0, perspective=0.0, scale=0.0, translate=0.0)

    if args.freeze_epochs > 0:
        print(f"=== stage 1: backbone frozen, {args.freeze_epochs} epochs ===")
        model = YOLO(args.model)
        model.train(epochs=args.freeze_epochs, freeze=BACKBONE_LAYERS,
                    lr0=args.lr0_stage1, name=f"{args.name}_frozen", **common)
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

    print(f"=== stage 2: full fine-tune from {weights}, {args.epochs} epochs, "
          f"lr0={args.lr0_stage2} ===")
    model = YOLO(weights)
    results = model.train(epochs=args.epochs, lr0=args.lr0_stage2, name=args.name, **common)
    print(results)


if __name__ == "__main__":
    main()
