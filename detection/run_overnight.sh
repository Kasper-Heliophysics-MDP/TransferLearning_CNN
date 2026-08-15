#!/bin/bash
# Overnight batch (2026-08-14): contaminated-negative removal, noise floor,
# hyperparameters, resolution.  Plan: OVERNIGHT_PLAN.md.
#
#   nohup bash detection/run_overnight.sh > detection/overnight.log 2>&1 &
#
# Sequential on purpose -- one T4.  No polling/wait logic anywhere: the last
# round lost 4 hours to a `pgrep -f "train_yolo.py"` guard that matched its own
# bash command line and deadlocked.  Sequential execution needs no guard.
#
# DATASET: yolo7_single, NOT yolo6_single.  yolo6 applied
# --drop-contaminated-negatives to val as well as train, shrinking val 581->565.
# The 16 removed val windows are exactly the ones the detector fires on, so that
# dataset lifts AP by deleting false positives -- nothing to do with the training
# change being measured.  yolo7 is built with --contaminated-negatives-scope
# train: val is byte-identical to the one v5_single scored 0.361 on (verified
# with diff -r on both images/ and labels/), train differs only by the 74 dropped
# windows.  One variable.
#
# NOTE boxes.csv is deliberately NOT refreshed: review_status.csv has ~90 reviews
# newer than it, but absorbing them would move the val set and break every
# comparison against 0.361.  Refresh after this batch is read, not before.

cd /home/ubuntu/Desktop/TransferLearning_CNN || exit 1
set -u

DATA=data/ecallisto/yolo7_single/data.yaml
WIDE=data/ecallisto/yolo7_wide/data.yaml

for f in "$DATA" "$WIDE"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
# Guard against a second trainer, NOT against any GPU process at all: this box
# permanently runs dcvagent (remote desktop, ~300MiB) and Xorg, and every run so
# far has coexisted with them.  Checking "is anything on the GPU" fails instantly
# here.  `pgrep -f train_yolo.py` is safe from THIS script -- its own cmdline is
# "bash detection/run_overnight.sh", which does not contain the pattern.  The
# 4-hour deadlock last round came from a separate waiter script whose own
# cmdline did contain it.
if pgrep -f "train_yolo.py" > /dev/null; then
  echo "FATAL: a trainer is already running:"; ps -eo pid,etime,cmd | grep "[t]rain_yolo.py"; exit 1
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$free_mb" -lt 10000 ]; then
  echo "FATAL: only ${free_mb}MiB VRAM free, need ~10GB:"; nvidia-smi; exit 1
fi
echo "pre-flight OK: ${free_mb}MiB VRAM free, no trainer running"

started=$(date +%s)

run () {  # run <name> <data.yaml> <extra args...>
  local name=$1 data=$2; shift 2
  echo ""
  echo "########## $name  |  $(date -Is) ##########"
  local t0=$(date +%s)
  python3 -u detection/train_yolo.py "$data" \
    --name "$name" --model yolov8s.pt --project detection/runs \
    --freeze-epochs 5 --epochs 30 --patience 10 --imgsz 640 --batch 16 "$@"
  local rc=$?
  echo "##### $name finished rc=$rc in $((($(date +%s)-t0)/60)) min"
  # no `set -e`: a crash in one arm must not take the remaining arms with it
}

# --- A. does dropping the 74 contaminated train negatives help? ---
#     control: v5_single AP50 0.361 (same val, same 570 train boxes)
run o_clean_neg "$DATA" --seed 0

# --- B. noise floor at the CURRENT data scale (old sigma=0.011 was measured on
#     243 train boxes; we are at 570).  With A's seed 0 that is n=4.
#     Nothing else in this batch is interpretable until this lands. ---
for s in 1 2 3; do run "o_seed$s" "$DATA" --seed "$s"; done

# --- C. hyperparameters, never once searched.  --lr0-stage1/--lr0-stage2 were
#     added today; before that both LRs were hard-coded, so the two LR arms
#     below are the first real entry into that space. ---
run o_lr_hi    "$DATA" --seed 0 --lr0-stage2 3e-4
run o_lr_lo    "$DATA" --seed 0 --lr0-stage2 3e-5
run o_nofreeze "$DATA" --seed 0 --freeze-epochs 0 --epochs 40
run o_ep60     "$DATA" --seed 0 --epochs 60 --patience 20

# --- D. time resolution: same boxes, same windows, only the render is
#     640x1280 instead of 640x640 (labels verified byte-identical).
#     AP@0.75 is the sensitive one here. ---
run o_wide "$WIDE" --seed 0 --imgsz 1280 --rect --batch 8

echo ""
echo "OVERNIGHT DONE  |  $(date -Is)  |  total $((($(date +%s)-started)/60)) min"
