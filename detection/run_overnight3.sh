#!/bin/bash
# Overnight 3 (2026-08-15): does Australia-ASSA rescue Type V?
#   nohup bash detection/run_overnight3.sh > detection/overnight3.log 2>&1 &
#
# THE QUESTION. Type V has been the one flat failure of Phase 1: 27 clean
# Arecibo samples in total, which cannot feed training and evaluation at once,
# and the measured result was AP 0.048 with 15% detection. ASSA was scraped for
# this single reason, and its 155 II/V boxes have now been reviewed by hand
# (149 usable, 6 discarded, 136 with corrected times). This asks whether that
# was enough.
#
# THE COMPARISON IS SINGLE-VARIABLE BY CONSTRUCTION:
#   val    byte-identical between arms (verified with diff -r over images/ and
#          labels/): 1096 images, 287 boxes, II 47 / III 187 / V 33.
#   III    582 training boxes in BOTH arms -- untouched.
#   II/V   50/14 (Arecibo only)  ->  105/34 (plus the reviewed ASSA boxes).
#
# ASSA NEGATIVES ARE DELIBERATELY EXCLUDED (4784 windows dropped). Including
# them would confound "more II/V labels" with "4784 more negative windows from
# a second station", and worse: no gap mining has been done on ASSA, the
# catalog omits roughly 36% of bursts, so those windows would enter as empty
# labels covering real bursts. That is reverse supervision, already measured
# here to be worse than label noise. Only ASSA's 73 reviewed positive windows
# are added.
#
# MULTI-CLASS on purpose: the Type V question cannot be asked of a single-class
# detector, so this gives up the ~19% AP edge single-class showed.
#
# 3 seeds per arm because sigma is somewhere in 0.018-0.056 depending on which
# measurement you trust, and a one-seed difference at this scale means nothing.
# Type V will be the noisiest column regardless -- 33 val boxes.

cd /home/ubuntu/Desktop/TransferLearning_CNN || exit 1
set -u

A=data/ecallisto/yolo9_arec/data.yaml
B=data/ecallisto/yolo9_pos/data.yaml
for f in "$A" "$B"; do [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done

# guard against a second trainer only; this box permanently runs dcvagent and
# Xorg on the GPU, so "is anything using the GPU" is always true here
if pgrep -f "train_yolo.py" > /dev/null; then
  echo "FATAL: a trainer is already running"; exit 1
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$free_mb" -lt 10000 ]; then echo "FATAL: only ${free_mb}MiB VRAM free"; exit 1; fi
echo "pre-flight OK: ${free_mb}MiB VRAM free"

started=$(date +%s)
run () {
  local name=$1 data=$2; shift 2
  echo ""; echo "########## $name | $(date -Is) ##########"
  local t0=$(date +%s)
  python3 -u detection/train_yolo.py "$data" --name "$name" --model yolov8s.pt \
    --project detection/runs --freeze-epochs 5 --epochs 30 --patience 10 \
    --imgsz 640 --batch 16 --workers 4 "$@"
  echo "##### $name rc=$? in $((($(date +%s)-t0)/60)) min"
}

for s in 0 1 2; do run "p2_arec_s$s" "$A" --seed "$s"; done
for s in 0 1 2; do run "p2_assa_s$s" "$B" --seed "$s"; done

echo ""
echo "OVERNIGHT3 DONE | $(date -Is) | total $((($(date +%s)-started)/60)) min"
