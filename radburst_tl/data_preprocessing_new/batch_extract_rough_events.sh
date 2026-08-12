#!/usr/bin/env bash
# One process per FILE (not per burst -- extract_events_own_station.py
# denoises a file once and crops every burst in it), same OS-level memory
# reclaim rationale as batch_process_own_station.sh.
set -uo pipefail

CATALOG="/home/ubuntu/Desktop/TransferLearning_CNN/data/burst_data/csv/original/burst_list_240330_240729.csv"
CSV_DIR="/home/ubuntu/Desktop/TransferLearning_CNN/data/burst_data/csv/original"
OUT_DIR="/home/ubuntu/Desktop/TransferLearning_CNN/data/burst_data/rough_events"
METHOD="fast"
LOG="/home/ubuntu/Desktop/TransferLearning_CNN/radburst_tl/data_preprocessing_new/batch_extract_rough_events.log"
N_FILES=$(python3 -c "import pandas as pd; print(pd.read_csv('$CATALOG')['file_name'].nunique())")

cd "$(dirname "$0")"
: > "$LOG"
echo "Starting rough-crop batch: $N_FILES unique files, method=$METHOD, $(date)" | tee -a "$LOG"

for i in $(seq 0 $((N_FILES - 1))); do
    echo "--- file_index $i / $((N_FILES - 1)) : $(date +%H:%M:%S) ---" >> "$LOG"
    python3 extract_events_own_station.py \
        --catalog "$CATALOG" \
        --csv-dir "$CSV_DIR" \
        --out-dir "$OUT_DIR" \
        --file-index "$i" \
        --method "$METHOD" >> "$LOG" 2>&1
done

echo "Batch finished: $(date)" | tee -a "$LOG"
echo "=== SUMMARY ===" >> "$LOG"
grep "^RESULT" "$LOG" | sort -u -t= -k2 -n >> "$LOG"
