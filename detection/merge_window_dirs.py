"""
merge_window_dirs.py

Symlink two or more scraped window directories into one, so build_yolo_dataset
(which takes a single --window-dir) can build a cross-station dataset.

Symlinks rather than copies: the arrays are 5-6 GB per station and identical to
the originals. windows.csv / boxes.csv are concatenated for real, because
build_yolo_dataset reads them as one table.

NOTE ON THE VAL SPLIT, which is the thing that makes cross-station safe here:
choose_val_days groups by DATE ALONE, not by (station, date). With Arecibo and
ASSA sharing 351 observation dates that is the correct behaviour and not a
coincidence worth "fixing" -- the same solar burst is frequently recorded by
both stations on the same day, so splitting a date across train and val would
put two recordings of one physical event on opposite sides. Grouping by date
keeps them together.

Usage:
    python detection/merge_window_dirs.py data/ecallisto/windows_merged \\
        data/ecallisto/windows data/ecallisto/windows_assa [--apply]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    wins, boxes, links, freq = [], [], [], []
    for src in args.sources:
        w = pd.read_csv(os.path.join(src, "windows.csv"), dtype={"date": str})
        b = pd.read_csv(os.path.join(src, "boxes.csv"), dtype={"date": str})
        wins.append(w); boxes.append(b)
        for f in w.file_name:
            p = os.path.abspath(os.path.join(src, f))
            if os.path.exists(p):
                links.append((f, p))
        md = os.path.join(src, "meta")
        if os.path.isdir(md):
            freq += [(f, os.path.abspath(os.path.join(md, f))) for f in os.listdir(md)]
        print(f"{src}: {len(w)} windows, {len(b)} boxes, {w.location.nunique()} station(s)")

    W = pd.concat(wins, ignore_index=True)
    B = pd.concat(boxes, ignore_index=True)
    dup = W.file_name.duplicated().sum()
    if dup:
        raise SystemExit(f"{dup} duplicate window file_name across sources")
    print(f"\nmerged: {len(W)} windows, {len(B)} boxes, "
          f"stations {sorted(W.location.unique())}")
    print(f"  box types: {B.type.value_counts().to_dict()}")
    print(f"  reviewed usable boxes: {(B.review_status == 'usable').sum()}")
    print(f"  {len(links)} array symlink(s), {len(freq)} frequency axis file(s)")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    os.makedirs(os.path.join(args.out_dir, "meta"), exist_ok=True)
    for name, target in links:
        link = os.path.join(args.out_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(target, link)
    for name, target in freq:
        link = os.path.join(args.out_dir, "meta", name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(target, link)
    W.to_csv(os.path.join(args.out_dir, "windows.csv"), index=False)
    B.to_csv(os.path.join(args.out_dir, "boxes.csv"), index=False)
    print(f"\nwrote {args.out_dir}/")


if __name__ == "__main__":
    main()
