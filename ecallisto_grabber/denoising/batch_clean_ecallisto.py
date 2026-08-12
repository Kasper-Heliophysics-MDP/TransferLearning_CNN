"""
Batch-clean every file in data/ecallisto/raw_events/metadata.csv and write
results to data/ecallisto/cleaned_events/ (raw_events left untouched).

Deliberately does NOT use clean_all_events() from clean_ecallisto_events.py:
that function returns {file_name: full_result_dict} for every row, which is
fine interactively (one file, inspect all intermediate arrays) but at this
scale (7000+ files) would hold all of them in memory simultaneously -- even
though each file's own arrays are tiny (~200 x few-hundred to few-thousand),
accumulating all of them at once does not stay tiny. Streams to disk and
lets each file's arrays be garbage collected before starting the next
instead.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from clean_ecallisto_events import clean_event_file

RAW_DIR = "/home/ubuntu/Desktop/TransferLearning_CNN/data/ecallisto/raw_events"
OUT_DIR = "/home/ubuntu/Desktop/TransferLearning_CNN/data/ecallisto/cleaned_events"
METHOD = "comprehensive"  # crops are tiny (~200 x hundreds-few-thousand) -- no memory/time reason to default to fast here


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = pd.read_csv(os.path.join(RAW_DIR, "metadata.csv"), dtype=str)
    meta.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)

    n_ok, n_missing, n_error, n_nonfinite = 0, 0, 0, 0
    t0 = time.time()
    for i, row in meta.iterrows():
        npy_path = os.path.join(RAW_DIR, row["file_name"])
        if not os.path.exists(npy_path):
            n_missing += 1
            continue
        try:
            result = clean_event_file(npy_path, row, method=METHOD)
        except Exception as e:
            print(f"ERROR {row['file_name']}: {e!r}", flush=True)
            n_error += 1
            continue

        cleaned = result["cleaned"]
        if not np.isfinite(cleaned).all():
            print(f"NON-FINITE {row['file_name']}", flush=True)
            n_nonfinite += 1
        np.save(os.path.join(OUT_DIR, row["file_name"]), cleaned)
        n_ok += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(meta)}] ok={n_ok} missing={n_missing} error={n_error} "
                  f"nonfinite={n_nonfinite} elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. ok={n_ok} missing={n_missing} error={n_error} nonfinite={n_nonfinite}")


if __name__ == "__main__":
    main()
