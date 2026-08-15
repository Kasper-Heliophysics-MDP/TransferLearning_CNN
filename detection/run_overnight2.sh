#!/bin/bash
# Overnight 2 (2026-08-15): new baseline on the gap-augmented dataset.
#
#   nohup bash detection/run_overnight2.sh > detection/overnight2.log 2>&1 &
#
# RUNS ALONGSIDE THE ASSA SCRAPE ON PURPOSE. The two barely contend:
#   GPU  - scrape uses none (it is network + FITS decode); T4 was 0% idle.
#   RAM  - scrape holds 0.38 GB; 13 GB free after stopping streamlit. Training
#          peaked around 3-4 GB in the last batch.
#   CPU  - the only real contention. 4 cores total, and ultralytics defaults to
#          8 dataloader workers, which would starve the scrape's decode step.
#          --workers 2 leaves it room. The GPU is the bottleneck for training
#          anyway, so this costs little.
#   DISK - 65 GB free; scrape needs ~1 GB more, each run ~40 MB.
#
# DO NOT start the review app while this runs: streamlit held 4 GB, and the
# previous batch OOM-killed it twice (exit 137).
#
# WHY 4 SEEDS AND NOTHING CLEVER: sigma was measured at 0.056 AP50 on 2026-08-14,
# five times the value assumed before. At that noise level nothing smaller than
# a ~0.11 swing is readable, so extra arms would produce numbers no one can
# interpret. The job tonight is to re-establish a baseline on the CHANGED val
# (the gap boxes moved the val day split; 0.361 is no longer comparable) and to
# re-measure sigma at the new data scale. One clean resolution arm is appended
# because the previous attempt confounded four variables at once.

cd /home/ubuntu/Desktop/TransferLearning_CNN || exit 1
set -u

DATA=data/ecallisto/yolo8_single/data.yaml
WIDE=data/ecallisto/yolo8_wide/data.yaml
for f in "$DATA" "$WIDE"; do [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done

if pgrep -f "train_yolo.py" > /dev/null; then
  echo "FATAL: a trainer is already running"; ps -eo pid,cmd | grep "[t]rain_yolo.py"; exit 1
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free_mb" -lt 10000 ] && { echo "FATAL: only ${free_mb}MiB VRAM free"; exit 1; }
echo "pre-flight OK: ${free_mb}MiB VRAM free"
echo "ASSA scrape at start: $(grep -c '^\[' detection/scrape_assa.log) station-days"

started=$(date +%s)
run () {
  local name=$1 data=$2; shift 2
  echo ""; echo "########## $name  |  $(date -Is) ##########"
  local t0=$(date +%s)
  python3 -u detection/train_yolo.py "$data" \
    --name "$name" --model yolov8s.pt --project detection/runs \
    --freeze-epochs 5 --epochs 30 --patience 10 --imgsz 640 --batch 16 \
    --workers 2 "$@"
  echo "##### $name rc=$? in $((($(date +%s)-t0)/60)) min | scrape at $(grep -c '^\[' detection/scrape_assa.log)"
}

# --- A. new baseline + noise floor on the changed val (4 seeds) ---
for s in 0 1 2 3; do run "n_seed$s" "$DATA" --seed "$s"; done

# --- B. resolution, this time as a SINGLE variable. The 2026-08-14 o_wide run
#     changed width, imgsz, batch (16->8) AND rect together and came out 4.1
#     sigma worse, which was therefore uninterpretable. Here only the render
#     size and imgsz change; batch stays 16. rect must stay on (without it
#     ultralytics pads 640x1280 back to 1280x1280 and doubles the compute for
#     identical content) -- that is the one residual difference, and it moved
#     AP50 by 0.001 when measured directly at eval time. ---
run n_wide "$WIDE" --seed 0 --imgsz 1280 --rect

echo ""
echo "OVERNIGHT2 DONE | $(date -Is) | total $((($(date +%s)-started)/60)) min"
echo "ASSA scrape at end: $(grep -c '^\[' detection/scrape_assa.log) station-days"
