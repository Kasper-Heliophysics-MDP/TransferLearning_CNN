"""
compare_repeat.py

Measure how repeatable the human time annotation is -- the number that sets the
real upper bound on AP for this whole approach.

Why it matters: boxes are full frequency height, so IoU is purely temporal, and
two equal-width intervals offset by d have IoU=(w-d)/(w+d). IoU>=0.5 therefore
demands d <= w/3, which for the 25th-percentile 30-second burst is +/-10s. If a
human re-annotating the same event lands more than that away from their own
earlier answer, then no amount of extra annotation can push AP50 past that
agreement rate -- the labels would be arguing with themselves. Measuring this
BEFORE investing days of review is the point.

Workflow:
  1. data/ecallisto/repeat_test/ holds 30 events (symlinks) with NO
     review_status.csv, so the tool pre-fills times from the catalog and the
     earlier human answer is invisible. The earlier answers live outside that
     directory, in event_review/backups/repeat_test_reference.csv.
  2. Point the review tool's sidebar directory box at repeat_test/ and
     re-annotate the 30 blind.
  3. Run this.

Usage: python detection/compare_repeat.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

TEST_DIR = "data/ecallisto/repeat_test"
REF = "event_review/backups/repeat_test_reference.csv"


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def main():
    ref = pd.read_csv(REF)
    new_path = os.path.join(TEST_DIR, "review_status.csv")
    if not os.path.exists(new_path):
        print(f"{new_path} 还不存在 -- 先在 repeat_test 目录上重标那 30 条")
        return
    new = pd.read_csv(new_path, dtype=str, keep_default_na=False)
    df = ref.merge(new[["file_name", "status", "manual_burst_range_json"]], on="file_name", how="left")
    df["manual_burst_range_json"] = df["manual_burst_range_json"].fillna("")

    rows = []
    for x in df.itertuples():
        if not x.manual_burst_range_json:
            rows.append((x.file_name, x.type, np.nan, np.nan, np.nan, "未重标"))
            continue
        j = json.loads(x.manual_burst_range_json)
        n0, n1 = float(j["start_s"]), float(j["end_s"])
        o0, o1 = float(x.old_start_s), float(x.old_end_s)
        rows.append((x.file_name, x.type, iou_1d(o0, o1, n0, n1),
                     n0 - o0, (n1 - n0) - (o1 - o0), getattr(x, "status", "")))
    d = pd.DataFrame(rows, columns=["file_name", "type", "iou", "d_start", "d_width", "status"])
    done = d.dropna(subset=["iou"])
    print(f"重标完成 {len(done)} / {len(d)} 条\n")
    if done.empty:
        return

    print("=== 人工标注自我一致性(这就是 AP 的真实天花板) ===")
    print(f"  中位 IoU              {done.iou.median():.3f}")
    for t in (0.5, 0.75, 0.9):
        print(f"  IoU>={t:.2f} 的比例      {(done.iou >= t).mean() * 100:5.0f}%"
              + ("   <- AP50 的上限" if t == 0.5 else ""))
    print(f"\n  起点偏差 |Δ|: 中位 {done.d_start.abs().median():5.1f}s   p80 {done.d_start.abs().quantile(.8):5.1f}s"
          f"   最大 {done.d_start.abs().max():5.1f}s")
    print(f"  宽度偏差 |Δ|: 中位 {done.d_width.abs().median():5.1f}s   p80 {done.d_width.abs().quantile(.8):5.1f}s")
    print(f"  起点是否有系统性偏移: 中位 {done.d_start.median():+.1f}s "
          f"(接近 0 = 没有系统漂移,只是随机抖动)")

    print("\n=== 一致性最差的 5 条(值得肉眼回看,可能是本身就难判的事件) ===")
    for x in done.nsmallest(5, "iou").itertuples():
        print(f"  IoU {x.iou:.2f}  Δstart {x.d_start:+6.1f}s  Δwidth {x.d_width:+6.1f}s  {x.file_name[26:60]}")

    print("\n=== 结论 ===")
    ceil = (done.iou >= 0.5).mean()
    print(f"  这套标注方式的 AP50 上限约 {ceil:.2f}。")
    if ceil < 0.8:
        print("  => 上限偏低。继续堆标注量的收益有限,应先改进标注方式本身")
        print("     (例如:放宽到 IoU=0.3 报告、或改用'重叠检出+时间偏差'口径、")
        print("      或为窄 burst 规定统一的起止判据以减少主观抖动)。")
    else:
        print("  => 上限够高,标注方式本身没问题,继续扩大标注量是有效投入。")


if __name__ == "__main__":
    main()
