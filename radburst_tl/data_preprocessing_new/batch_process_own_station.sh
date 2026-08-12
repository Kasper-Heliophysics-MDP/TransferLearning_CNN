#!/usr/bin/env bash
# Runs every burst_list.csv row through process_one_burst.py, one fresh
# python process per row (see process_one_burst.py's docstring for why:
# real files run 12-13GB peak RSS on a 15GB no-swap machine, too tight to
# trust in-process GC between files).
set -uo pipefail

CATALOG="/home/ubuntu/Desktop/TransferLearning_CNN/data/burst_data/csv/original/burst_list_240330_240729.csv"
CSV_DIR="/home/ubuntu/Desktop/TransferLearning_CNN/data/burst_data/csv/original"
OUT_DIR="/home/ubuntu/Desktop/TransferLearning_CNN/data/burst_data/csv/gan_training_windows_sumthreshold"
# "fast" chosen over "comprehensive" after measuring real peak RSS on the
# largest file (530k rows): fast=7.9GB, comprehensive=12.3GB against a 13GB
# budget on this no-swap machine -- comprehensive DID succeed but with only
# ~700MB margin, too tight to trust unattended across 74 sequential bursts.
# Re-run specific important bursts with comprehensive afterward if the fast
# output needs it (this is also what TRAINING_RESULTS_ANALYSIS.md's own
# recommended workflow suggests: fast pass first, selective comprehensive
# re-pass after visual QA).
METHOD="fast"
LOG="/home/ubuntu/Desktop/TransferLearning_CNN/radburst_tl/data_preprocessing_new/batch_run.log"
N_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$CATALOG')))")

cd "$(dirname "$0")"
: > "$LOG"
echo "Starting batch: $N_ROWS burst rows, method=$METHOD, $(date)" | tee -a "$LOG"

for i in $(seq 0 $((N_ROWS - 1))); do
    echo "--- index $i / $((N_ROWS - 1)) : $(date +%H:%M:%S) ---" >> "$LOG"
    python3 process_one_burst.py \
        --catalog "$CATALOG" \
        --csv-dir "$CSV_DIR" \
        --out-dir "$OUT_DIR" \
        --index "$i" \
        --method "$METHOD" >> "$LOG" 2>&1
done

echo "Batch finished: $(date)" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$LOG"
grep "^RESULT" "$LOG" | tee -a "$LOG"
