"""
apply_gap_verdicts.py

Turn confirmed catalog-gap candidates (mine_catalog_gaps.py + a human filling
the `verdict` column) into two concrete outputs.

WHAT A CONFIRMED GAP IS WORTH, AND WHY IT SPLITS IN TWO
-------------------------------------------------------
A confirmed candidate says "there is a real burst here that the catalog never
recorded". That fact is worth two different things, with very different risk:

  1. REMOVE WRONG SUPERVISION (safe, applied here).
     A window with no catalog entry is used as an empty-label training image --
     it actively teaches "no burst here" while pointing at one. PHASE1_RESULTS
     calls this worse than label noise, because it is reverse signal rather
     than missing signal. Dropping such a window costs nothing: it is pure
     subtraction, and no new coordinates are invented.

  2. ADD A POSITIVE BOX (NOT done here, deliberately).
     Writing the model's own box into boxes.csv would train the detector on
     coordinates it produced itself -- and box extent is precisely its weakest
     axis (AP@0.75 0.16, and maxconf keeps the most CONFIDENT box in a cluster,
     not the best-placed one). That is self-distillation on the one dimension
     already known to be bad. Confirmed gaps that deserve to become boxes go
     through event_review's drag-to-annotate so a human sets the times; this
     script only emits the worklist for that.

So: verdicts feed the contaminated-negatives list immediately, and a separate
worklist for hand annotation. Nothing invented by the model enters training
geometry.

Safety, matching event_review/merge_subset_reviews.py:
  * DRY RUN BY DEFAULT -- nothing is written without --apply.
  * Every file written is backed up (timestamped) first.
  * Additive only: existing entries are preserved, never rewritten.

Usage:
    python detection/apply_gap_verdicts.py detection/gap_review/gap_candidates.csv [--apply]
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime

import pandas as pd

NEG_CSV = "data/ecallisto/windows/contaminated_negatives.csv"
BACKUP_DIR = "event_review/backups"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts", help="gap_candidates.csv with the verdict column filled")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--neg-csv", default=NEG_CSV)
    ap.add_argument("--worklist", default="detection/gap_review/to_annotate.csv",
                    help="confirmed gaps that should get a hand-drawn box in event_review")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="ignore confirmed rows below this confidence (0 = use all)")
    args = ap.parse_args()

    df = pd.read_csv(args.verdicts)
    df["verdict"] = df["verdict"].astype(str).str.strip().str.lower()

    known = {"y", "n", "?", ""}
    bad = sorted(set(df.verdict) - known - {"nan"})
    if bad:
        print(f"!! unrecognised verdict values (expected y/n/?): {bad}")
        print("!! fix the CSV -- refusing to guess what these mean")
        return

    n_total = len(df)
    filled = df[df.verdict.isin(["y", "n", "?"])]
    yes = filled[filled.verdict == "y"]
    if args.min_conf > 0:
        dropped = (yes.conf < args.min_conf).sum()
        yes = yes[yes.conf >= args.min_conf]
        print(f"--min-conf {args.min_conf}: ignoring {dropped} confirmed row(s) below it")

    print(f"{n_total} candidates, {len(filled)} reviewed "
          f"({n_total - len(filled)} still blank)")
    if len(filled):
        print("  " + "  ".join(f"{k}={v}" for k, v in
                               filled.verdict.value_counts().to_dict().items()))
    if not len(yes):
        print("\nno confirmed gaps -- nothing to do")
        return

    # (1) pure-negative windows: the catalog lists nothing at all, so the whole
    #     window is currently an empty-label training image and every confirmed
    #     burst in it is reverse supervision.
    pure_neg = yes[yes.n_catalog_boxes == 0]
    # (2) windows that already carry catalog boxes: they are NOT negatives, so
    #     dropping them would throw away good boxes. These only ever become an
    #     annotation task.
    has_boxes = yes[yes.n_catalog_boxes > 0]

    existing = set()
    if os.path.exists(args.neg_csv):
        existing = set(pd.read_csv(args.neg_csv)["file_name"])
    new_neg = sorted(set(pure_neg.file_name) - existing)

    print(f"\nconfirmed gaps: {len(yes)}")
    print(f"  on windows with NO catalog entry : {len(pure_neg)} "
          f"-> {len(new_neg)} new contaminated negatives "
          f"({len(set(pure_neg.file_name)) - len(new_neg)} already listed)")
    print(f"  on windows that DO have entries  : {len(has_boxes)} "
          f"-> annotation worklist only (dropping them would discard good boxes)")
    print(f"\nworklist for event_review hand-annotation: {len(yes)} row(s)")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    if new_neg:
        if os.path.exists(args.neg_csv):
            bak = os.path.join(BACKUP_DIR, f"contaminated_negatives_{stamp}.csv")
            shutil.copy2(args.neg_csv, bak)
            print(f"backed up {args.neg_csv} -> {bak}")
            cur = pd.read_csv(args.neg_csv)
        else:
            cur = pd.DataFrame(columns=["file_name"])
        add = pd.DataFrame({"file_name": new_neg})
        for c in cur.columns:
            if c != "file_name":
                add[c] = ""
        out = pd.concat([cur, add[cur.columns]], ignore_index=True)
        out.to_csv(args.neg_csv, index=False)
        print(f"wrote {args.neg_csv}: {len(cur)} -> {len(out)} rows")
    else:
        print("no new contaminated negatives to add")

    os.makedirs(os.path.dirname(args.worklist), exist_ok=True)
    if os.path.exists(args.worklist):
        shutil.copy2(args.worklist, os.path.join(BACKUP_DIR, f"to_annotate_{stamp}.csv"))
    cols = [c for c in ["id", "file_name", "date", "conf", "start_s", "end_s",
                        "n_catalog_boxes", "split", "note"] if c in yes.columns]
    yes[cols].to_csv(args.worklist, index=False)
    print(f"wrote {args.worklist}: {len(yes)} row(s) to hand-annotate")
    print("\nNOTE: rebuild the dataset for the negative drop to take effect, and run\n"
          "      refresh_boxes.py after the hand annotation to pull the new boxes in.")


if __name__ == "__main__":
    main()
