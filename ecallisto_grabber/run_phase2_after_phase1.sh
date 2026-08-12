#!/usr/bin/env bash
# Waits for the currently-running phase-1 scrape (Arecibo-Observatory, which
# turned out to have >=500 matching station-days on its own and consumed the
# whole --limit 500) to exit, then immediately starts phase 2: the other 7
# curated stations, which phase 1 never got to. Same --limit 500 bound.
set -uo pipefail

cd "$(dirname "$0")"

echo "Waiting for phase-1 scrape (PID 320540, Arecibo-Observatory) to finish..."
while kill -0 320540 2>/dev/null && ps -p 320540 -o cmd= | grep -q "scrape.py"; do
    sleep 60
done
echo "Phase 1 done at $(date). Starting phase 2 (other 7 stations)."

python3 scrape.py \
    --start 2021-01-01 --end 2024-03-31 --types II III V \
    --stations Australia-ASSA SWISS-Landschlacht EGYPT-Alexandria SWISS-MUHEN ROSWELL-NM SPAIN-PERALEJOS GREENLAND \
    --out-dir ../data/ecallisto/raw_events \
    --limit 500 > scrape_run_phase2.log 2>&1

echo "Phase 2 done at $(date)."
