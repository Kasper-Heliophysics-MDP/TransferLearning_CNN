#!/usr/bin/env bash
# Phase 1 (Arecibo) and phase 2 (Australia-ASSA) each hit their global
# --limit before ever reaching another station -- both have enough
# matching station-days on their own to soak up a 500-day budget alone.
# Phase 3 runs each remaining station in its OWN scrape.py invocation with
# a small per-station limit, so every station is guaranteed some data
# instead of hoping a shared limit rolls over to it.
set -uo pipefail
cd "$(dirname "$0")"

STATIONS=(SWISS-Landschlacht EGYPT-Alexandria SWISS-MUHEN ROSWELL-NM SPAIN-PERALEJOS GREENLAND)
LIMIT=65

for st in "${STATIONS[@]}"; do
    echo "=== $(date) starting $st (limit $LIMIT) ==="
    python3 scrape.py \
        --start 2021-01-01 --end 2024-03-31 --types II III V \
        --stations "$st" \
        --out-dir ../data/ecallisto/raw_events \
        --limit "$LIMIT" >> scrape_run_phase3.log 2>&1
    echo "=== $(date) finished $st ==="
done

echo "Phase 3 all done at $(date)."
