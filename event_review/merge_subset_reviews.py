"""
merge_subset_reviews.py

Merge reviews done in a SUBSET directory (e.g. data/ecallisto/fix_start/) back
into the parent review_status.csv.

Subset directories exist because the review app can filter by station/type/
status but not by "this record has a specific defect". To re-review a targeted
set -- such as the 107 events whose burst START was left at the catalog default
-- the events get symlinked into their own directory with their own
review_status.csv, reviewed there, then merged back by this script.

Safety, because this writes to hand-made data that cost hours:
  * DRY RUN BY DEFAULT. Nothing is written without --apply.
  * The parent CSV is backed up (timestamped) before any write.
  * Only file_names present in the subset are ever touched; a row that somehow
    isn't in the parent is reported and skipped rather than appended.
  * Rows whose content is unchanged are left alone, so reviewed_at doesn't
    churn.

Usage:
    python event_review/merge_subset_reviews.py data/ecallisto/fix_start \
        data/ecallisto/raw_events [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

COLS = ["status", "notes", "override_params_json", "reviewed_at",
        "manual_burst_range_json", "problem_tags", "auto_retry_done"]


def _range(js: str):
    if not js:
        return None
    d = json.loads(js)
    return float(d["start_s"]), float(d["end_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir"); ap.add_argument("parent_dir")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    sub_p = os.path.join(args.subset_dir, "review_status.csv")
    par_p = os.path.join(args.parent_dir, "review_status.csv")
    sub = pd.read_csv(sub_p, dtype=str, keep_default_na=False).set_index("file_name")
    par = pd.read_csv(par_p, dtype=str, keep_default_na=False).set_index("file_name")
    print(f"subset {len(sub)} 条   parent {len(par)} 条")

    missing = [f for f in sub.index if f not in par.index]
    if missing:
        print(f"⚠️  {len(missing)} 条在 parent 里不存在,将跳过(本脚本只更新,不新增)")

    changed, dstart, dwidth, unchanged = [], [], [], 0
    for f in sub.index:
        if f not in par.index:
            continue
        if all(str(sub.at[f, c]) == str(par.at[f, c]) for c in COLS):
            unchanged += 1
            continue
        o, n = _range(par.at[f, "manual_burst_range_json"]), _range(sub.at[f, "manual_burst_range_json"])
        if o and n:
            dstart.append(n[0] - o[0]); dwidth.append((n[1] - n[0]) - (o[1] - o[0]))
        changed.append(f)

    print(f"\n有改动 {len(changed)} 条,无改动 {unchanged} 条")
    if dstart:
        ds, dw = np.array(dstart), np.array(dwidth)
        print(f"  起点移动: 中位 {np.median(ds):+.0f}s   往后挪的占 {(ds > 0).mean() * 100:.0f}%"
              f"   (预期:catalog 起点偏早,应普遍往后)")
        print(f"  宽度变化: 中位 {np.median(dw):+.0f}s   变窄的占 {(dw < 0).mean() * 100:.0f}%"
              f"   (预期:去掉左边多含的一段,应普遍变窄)")
    for f in changed[:8]:
        o, n = _range(par.at[f, "manual_burst_range_json"]), _range(sub.at[f, "manual_burst_range_json"])
        so = f"{o[0]:.0f}-{o[1]:.0f}" if o else "-"
        sn = f"{n[0]:.0f}-{n[1]:.0f}" if n else "-"
        print(f"    {f[26:52]:28s} {so:>12s} -> {sn:<12s} [{par.at[f,'status']} -> {sub.at[f,'status']}]")
    if len(changed) > 8:
        print(f"    ... 还有 {len(changed) - 8} 条")

    if not args.apply:
        print("\n[DRY RUN] 没有写入任何东西。确认无误后加 --apply 再跑一次。")
        return
    if not changed:
        print("\n没有需要写入的改动。")
        return

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    bak = f"event_review/backups/review_before_merge_{stamp}.csv"
    os.makedirs(os.path.dirname(bak), exist_ok=True)
    shutil.copy(par_p, bak)
    for f in changed:
        for c in COLS:
            par.at[f, c] = sub.at[f, c]
    par.reset_index().to_csv(par_p, index=False)
    print(f"\n已写入 {len(changed)} 条改动到 {par_p}")
    print(f"写入前的备份: {bak}")


if __name__ == "__main__":
    main()
