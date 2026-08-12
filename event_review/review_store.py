"""
review_store.py

Persistence for human review decisions. One review_status.csv per data-source
directory, colocated with that directory's own metadata.csv -- same
file_name-keyed-join convention used everywhere else in this repo
(batch_processing.py, extract_events_own_station.py, process_one_burst.py).

Every classification click writes immediately (no separate "submit" step) via
a write-to-temp-then-os.replace atomic swap, so a crash mid-session loses at
most the row being edited right now, never prior work.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import pandas as pd

REVIEW_COLUMNS = [
    "file_name", "status", "notes", "override_params_json", "reviewed_at",
    "manual_burst_range_json", "problem_tags", "auto_retry_done",
]

# internal codes, decoupled from the UI's Chinese labels so relabeling the
# UI later doesn't touch stored data or any downstream filtering logic
STATUS_USABLE = "usable"
STATUS_CASE_BY_CASE = "case_by_case"
STATUS_FLAGGED = "flagged_for_retry"
STATUS_DISCARD = "discard"

STATUS_LABELS = {
    STATUS_USABLE: "直接可用",
    STATUS_CASE_BY_CASE: "可用但需case by case处理",
    STATUS_FLAGGED: "有已知问题(待自动重试)",
    STATUS_DISCARD: "弃用",
}

# "有直接问题" sub-tags, seeded from real review notes across 190+ eCallisto
# + 25 own-station events reviewed in this session -- not a guess from the
# dev log alone. TIME_WRONG dominates real occurrences (seen on nearly every
# station reviewed); the others are documented failure modes but were seen
# less often in this first pass.
TAG_TIME_WRONG = "time_wrong"
TAG_BURST_FAINT = "burst_faint"
TAG_HORIZONTAL_RFI = "horizontal_rfi"
TAG_VERTICAL_RFI = "vertical_rfi"

PROBLEM_TAG_LABELS = {
    TAG_TIME_WRONG: "标注时间不准(比实际burst早/晚)",
    TAG_BURST_FAINT: "burst偏淡/被抹除",
    TAG_HORIZONTAL_RFI: "横向RFI残留重",
    TAG_VERTICAL_RFI: "竖向RFI残留重",
}


def _review_path(out_dir: str) -> str:
    return os.path.join(out_dir, "review_status.csv")


def load_review(out_dir: str) -> pd.DataFrame:
    """Load review_status.csv for a data-source directory, or an empty frame
    with the right columns if it doesn't exist yet (first time reviewing this
    directory)."""
    path = _review_path(out_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    # keep_default_na=False: with only dtype=str, pandas still turns a truly
    # empty CSV field into NaN (a float) rather than "" -- and float('nan')
    # is truthy in Python, so `if row["override_params_json"]` downstream
    # doesn't catch it, and json.loads(nan) crashes. This is what upsert_review
    # actually writes for an unset field (empty string, see _atomic_write), so
    # reading it back as "" instead of NaN is just symmetry, not new behavior.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[REVIEW_COLUMNS]


def upsert_review(
    out_dir: str,
    file_name: str,
    status: str,
    notes: str = "",
    override_params: dict | None = None,
    manual_burst_range: dict | None = None,
    problem_tags: list[str] | None = None,
    auto_retry_done: bool = False,
) -> None:
    """Insert or update one file's review row and write atomically. Latest
    write wins -- no history log (deliberate v1 scope cut, not an oversight:
    the point is triage state, not an audit trail).

    `manual_burst_range` is saved regardless of `status` -- unlike
    `override_params` (a case-by-case cleaning-parameter tweak, only
    meaningful when status=case_by_case), a corrected burst time range is a
    fact about the event itself. Real review data confirmed the catalog time
    is off (usually early) across nearly every station, not a rare edge
    case, so an event corrected this way can legitimately end up "usable"
    with the correction still needing to be saved.
    """
    df = load_review(out_dir)
    override_json = json.dumps(override_params) if override_params else ""
    manual_range_json = json.dumps(manual_burst_range) if manual_burst_range else ""
    tags_str = ",".join(problem_tags) if problem_tags else ""
    new_row = {
        "file_name": file_name,
        "status": status,
        "notes": notes or "",
        "override_params_json": override_json,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manual_burst_range_json": manual_range_json,
        "problem_tags": tags_str,
        "auto_retry_done": "true" if auto_retry_done else "",
    }

    if file_name in df["file_name"].values:
        idx = df.index[df["file_name"] == file_name][0]
        for k, v in new_row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    _atomic_write(out_dir, df)


def get_review(out_dir: str, file_name: str) -> dict | None:
    """The stored review row for one file, or None if unreviewed."""
    df = load_review(out_dir)
    match = df[df["file_name"] == file_name]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    row["override_params"] = json.loads(row["override_params_json"]) if row["override_params_json"] else {}
    row["manual_burst_range"] = json.loads(row["manual_burst_range_json"]) if row["manual_burst_range_json"] else {}
    row["problem_tags_list"] = row["problem_tags"].split(",") if row["problem_tags"] else []
    return row


def progress_counts(out_dir: str, total: int, done_names: set[str] | None = None) -> dict:
    """{'reviewed': n, 'remaining': n, 'usable': n, 'case_by_case': n, 'discard': n}.

    `reviewed` counts `done_names` if given (app.py's reviewed-minus-pending-
    retry set) rather than every row in review_status.csv, so a flagged,
    not-yet-retried event doesn't inflate the "done" count -- it isn't done,
    see STATUS_FLAGGED."""
    df = load_review(out_dir)
    counts = df["status"].value_counts().to_dict()
    reviewed = len(done_names) if done_names is not None else len(df)
    return {
        "reviewed": reviewed,
        "remaining": max(0, total - reviewed),
        STATUS_USABLE: counts.get(STATUS_USABLE, 0),
        STATUS_CASE_BY_CASE: counts.get(STATUS_CASE_BY_CASE, 0),
        STATUS_DISCARD: counts.get(STATUS_DISCARD, 0),
    }


def _atomic_write(out_dir: str, df: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = _review_path(out_dir)
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".review_status_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            df.to_csv(f, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
