"""
app.py

Streamlit entrypoint for the burst review tool. Launch with:
    streamlit run event_review/app.py

Per event: raw | cleaned side by side (shared color scale within the cleaned
comparison, see plotting.py), a burst time range (catalog-derived, or
human-corrected -- see below) with the actual protected window shown too,
a 4-way classification, and a parameter panel (including a manual time-range
override) to interactively fix cases where the default protection window
clips real signal or is simply wrong.

Manual time correction: real review across 190+ eCallisto + 25 own-station
events (this session) found the catalog-derived burst time is off (usually
early) on nearly every station reviewed -- not a rare edge case. So the
burst range used for protection can be corrected here directly, independent
of which final classification gets chosen (an event corrected this way can
legitimately end up "usable", not just "case-by-case").

4-way classification (STATUS_* in review_store.py):
- usable: use as-is (production result, or a corrected/reprocessed result
  the reviewer is satisfied with).
- case_by_case: reviewer manually tuned parameters right now; those
  parameters are saved as the final override.
- flagged_for_retry ("有已知问题"): reviewer recognizes a specific, tagged
  problem (see review_store.PROBLEM_TAG_LABELS) but doesn't want to hand-tune
  it right now. Saved with problem_tags, does NOT count as reviewed yet, and
  is sorted to the end of the queue. Revisiting it auto-applies
  data_access.SUGGESTED_PARAMS_BY_TAG for its tags as the new baseline
  preview -- reviewer then confirms usable/case_by_case/discard from there.
- discard: excluded.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import data_access as da
import review_store as rs
from plotting import render_pair, cleaned_scale

st.set_page_config(page_title="Burst Review", layout="wide")

DEFAULT_DIRS = {
    da.SOURCE_OWN_STATION: os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "burst_data", "rough_events")
    ),
    da.SOURCE_ECALLISTO: os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "ecallisto", "raw_events")
    ),
}
SOURCE_LABELS = {da.SOURCE_OWN_STATION: "Own-station", da.SOURCE_ECALLISTO: "eCallisto scrape output"}

_ACTION_LABELS = {
    rs.STATUS_USABLE: "直接可用 (1)",
    rs.STATUS_CASE_BY_CASE: "case-by-case (2)",
    "flagged": "有已知问题 (3)",
    rs.STATUS_DISCARD: "弃用 (4)",
}

# Best-effort keyboard shortcuts: Streamlit has no native shortcut support.
# st.components.v1.html renders in its own iframe, so the listener has to
# attach to window.parent.document to see keys pressed on the actual page,
# not the (invisible, unfocused) iframe itself. Guarded by a flag on
# window.parent so re-running this block on every Streamlit rerun doesn't
# stack up duplicate listeners (confirmed necessary -- without the guard,
# each rerun adds one, and after N reruns a single keypress fires N times).
# Fragile by nature: matches on visible button text, so it breaks silently
# if a button's label text changes. Treat as a nice-to-have, not load-bearing.
_SHORTCUT_JS = f"""
<script>
if (!window.parent.__reviewShortcutsInstalled) {{
    window.parent.__reviewShortcutsInstalled = true;
    const keyMap = {{
        "1": {_ACTION_LABELS[rs.STATUS_USABLE]!r},
        "2": {_ACTION_LABELS[rs.STATUS_CASE_BY_CASE]!r},
        "3": {_ACTION_LABELS["flagged"]!r},
        "4": {_ACTION_LABELS[rs.STATUS_DISCARD]!r}
    }};
    window.parent.document.addEventListener("keydown", function(e) {{
        if (["TEXTAREA", "INPUT"].includes(e.target.tagName)) return;
        const label = keyMap[e.key];
        if (!label) return;
        const buttons = window.parent.document.querySelectorAll("button");
        for (const b of buttons) {{
            if (b.innerText.trim() === label) {{ b.click(); break; }}
        }}
    }});
}}
</script>
"""
st.components.v1.html(_SHORTCUT_JS, height=0)


# ---------------------------------------------------------------------------
# sidebar: source, filters, navigation
# ---------------------------------------------------------------------------

st.sidebar.header("数据源")
source = st.sidebar.radio("来源", list(SOURCE_LABELS), format_func=lambda s: SOURCE_LABELS[s])
out_dir = st.sidebar.text_input("目录", value=DEFAULT_DIRS[source])

if not os.path.exists(os.path.join(out_dir, "metadata.csv")):
    st.sidebar.error(f"{out_dir} 下没有 metadata.csv")
    st.stop()

meta = da.load_metadata(out_dir)
review_df = rs.load_review(out_dir)
reviewed_names = set(review_df["file_name"])
status_by_name = {r["file_name"]: r["status"] for _, r in review_df.iterrows()}
# "reviewed" for progress/queue purposes excludes flagged-pending-retry --
# those aren't done, they're deferred (see module docstring)
pending_retry_names = set(
    review_df[(review_df["status"] == rs.STATUS_FLAGGED) & (review_df["auto_retry_done"] != "true")]["file_name"]
)
done_names = reviewed_names - pending_retry_names

st.sidebar.header("筛选")
stations = sorted(meta["location"].dropna().unique())
sel_stations = st.sidebar.multiselect("站点", stations, default=stations)
types = sorted(meta["type"].dropna().unique())
default_types = [t for t in types if t != "0"] or types
sel_types = st.sidebar.multiselect("类型 (0 = 负样本)", types, default=default_types)
status_options = ["未审核", rs.STATUS_LABELS[rs.STATUS_USABLE], rs.STATUS_LABELS[rs.STATUS_CASE_BY_CASE],
                   rs.STATUS_LABELS[rs.STATUS_FLAGGED], rs.STATUS_LABELS[rs.STATUS_DISCARD]]
sel_status = st.sidebar.multiselect("审核状态", status_options, default=status_options)

filtered = meta[meta["location"].isin(sel_stations) & meta["type"].isin(sel_types)]


def _status_label(file_name: str) -> str:
    s = status_by_name.get(file_name)
    return "未审核" if s is None else rs.STATUS_LABELS.get(s, s)


filtered = filtered[filtered["file_name"].apply(lambda f: _status_label(f) in sel_status)]

st.sidebar.header("进度")
counts = rs.progress_counts(out_dir, total=len(meta), done_names=done_names)
st.sidebar.metric("已审完 / 总数", f"{counts['reviewed']} / {len(meta)}")
st.sidebar.caption(
    f"直接可用 {counts[rs.STATUS_USABLE]} · case-by-case {counts[rs.STATUS_CASE_BY_CASE]} · "
    f"弃用 {counts[rs.STATUS_DISCARD]} · 待重试(不计入已审) {len(pending_retry_names)}"
)

if filtered.empty:
    st.warning("当前筛选条件下没有事件。")
    st.stop()

# queue order: not-yet-touched first, flagged-pending-retry last (so cycling
# through naturally reaches them after everything else has a first pass),
# fully-resolved in between (still visitable for re-check, just not prioritized)
def _queue_rank(f: str) -> int:
    if f in pending_retry_names:
        return 2
    if f not in reviewed_names:
        return 0
    return 1


file_names = sorted(filtered["file_name"].tolist(), key=_queue_rank)

state_key = f"current_{source}_{out_dir}"
if state_key not in st.session_state or st.session_state[state_key] not in file_names:
    st.session_state[state_key] = file_names[0]

st.sidebar.header("导航")
col_prev, col_next = st.sidebar.columns(2)
cur_idx = file_names.index(st.session_state[state_key])
if col_prev.button("◀ 上一个", use_container_width=True) and cur_idx > 0:
    st.session_state[state_key] = file_names[cur_idx - 1]
    st.rerun()
if col_next.button("下一个 ▶", use_container_width=True) and cur_idx < len(file_names) - 1:
    st.session_state[state_key] = file_names[cur_idx + 1]
    st.rerun()

selected = st.sidebar.selectbox(
    f"事件 ({cur_idx + 1}/{len(file_names)})", file_names,
    index=cur_idx,
    format_func=lambda f: f"{'⏳ ' if f in pending_retry_names else ('✓ ' if f in done_names else '')}{f}",
)
if selected != st.session_state[state_key]:
    st.session_state[state_key] = selected
    st.rerun()

file_name = st.session_state[state_key]
row = filtered[filtered["file_name"] == file_name].iloc[0]

# ---------------------------------------------------------------------------
# main: event header + raw/cleaned + classification
# ---------------------------------------------------------------------------

st.title("Burst 人工审核")
st.caption(
    f"**{row['location']}** · {row['date']} · type={row['type']} · "
    f"事件时间 {row.get('event_start_time', '')}–{row.get('event_end_time', '')} · "
    f"uncertain={row.get('uncertain', '')} · sample_interval_s={row['sample_interval_s']}"
)

try:
    event = da.load_event(out_dir, row.to_dict(), source)
except FileNotFoundError as e:
    st.error(f"无法加载:{e}")
    st.stop()

if event["raw"].shape[1] == 0:
    # confirmed real cause once (own-station raw reconstruction returning an
    # empty slice for a session crossing midnight, now fixed at the source --
    # see _elapsed_since_session_start), but surfacing this as a clear error
    # instead of letting the number_input below crash on max_value=0 either
    # way, in case some other data issue produces the same symptom later
    st.error("这条事件的 raw 数据是空的(0 列),大概率是原始数据定位/切片有问题,不是这条事件本身没数据。"
              "建议跳过、检查日志,不要在这条上继续操作。")
    st.stop()

existing = rs.get_review(out_dir, file_name)
is_pending_retry = file_name in pending_retry_names
sample_interval_s = event["sample_interval_s"]

# ---------------------------------------------------------------------------
# manual burst time-range correction -- see module docstring on why this
# exists and isn't hidden in a collapsed panel: real review found the
# catalog time wrong on nearly every station, not a rare case worth burying.
# ---------------------------------------------------------------------------

auto_indices = event["event_indices"]
manual_saved = existing["manual_burst_range"] if existing else {}

if auto_indices is not None:
    default_start_s = manual_saved.get("start_s", auto_indices[0] * sample_interval_s)
    default_end_s = manual_saved.get("end_s", auto_indices[1] * sample_interval_s)
    max_s = event["raw"].shape[1] * sample_interval_s

    st.subheader("Burst 实际时间段(可修正)")
    mc1, mc2, mc3 = st.columns([1, 1, 2])
    manual_start_s = mc1.number_input("开始(秒)", min_value=0.0, max_value=max_s,
                                       value=float(default_start_s), step=1.0, key=f"mstart_{file_name}")
    manual_end_s = mc2.number_input("结束(秒)", min_value=0.0, max_value=max_s,
                                     value=float(default_end_s), step=1.0, key=f"mend_{file_name}")
    is_manual_default = (
        abs(manual_start_s - auto_indices[0] * sample_interval_s) < 1e-6
        and abs(manual_end_s - auto_indices[1] * sample_interval_s) < 1e-6
    )
    mc3.caption("默认值来自 catalog 自动标注(或之前保存的修正)。改动会替代红线标记,并作为下方保护窗口的依据。"
                if is_manual_default else "⚠️ 已修正,和 catalog 自动标注不同 -- 保存时会记录这个修正后的时间段。")
    effective_indices = (round(manual_start_s / sample_interval_s), round(manual_end_s / sample_interval_s))
else:
    effective_indices = None
    is_manual_default = True

# ---------------------------------------------------------------------------
# cleaning parameters -- auto-seeded from problem tags if this event is a
# pending auto-retry, otherwise production defaults
# ---------------------------------------------------------------------------

method = da.PRODUCTION_METHOD[source]
preset = da.default_params(method)
if is_pending_retry:
    suggested = {}
    for tag in existing.get("problem_tags_list", []):
        suggested.update(da.SUGGESTED_PARAMS_BY_TAG.get(tag, {}))
    preset = {**preset, **suggested}
    st.info(f"这条之前被标记为「有已知问题」({', '.join(rs.PROBLEM_TAG_LABELS.get(t, t) for t in existing['problem_tags_list'])}),"
            f"下面已经按建议参数重新处理,看看是否好转。")

def _apply_params_to_sliders(fname: str, params: dict) -> None:
    """Push a preset into the slider widgets.

    Writing st.session_state inside an on_click callback is the supported way
    to change a widget's value: the callback runs BEFORE the widgets are
    rebuilt on the next rerun, whereas assigning to a widget's key after that
    widget has already been instantiated in the same run raises. Nothing here
    touches review_status.csv -- this is UI state only, so already-reviewed
    records cannot be altered by pressing it.
    """
    for state_key, param in (("margin", "known_burst_margin"), ("bts", "base_threshold_sigma"),
                             ("ramp", "suppression_ramp"), ("bs", "burst_sigma"),
                             ("vcov", "vrfi_coverage_threshold"), ("vsig", "vrfi_sigma")):
        if param in params:
            st.session_state[f"{state_key}_{fname}"] = params[param]


minimal_default = da.uses_minimal_view(row["type"])
show_full_clean = False
if minimal_default:
    show_full_clean = st.checkbox(
        "改用完整去噪版(默认对 Type II/V 只做逐行中位数扣除)",
        value=False, key=f"fullclean_{file_name}",
        help="Type II/V 的 burst 通常在 raw 上就很明显,而完整去噪在实测里有 2/8 把微弱的 V 抹掉、"
             "1/8 引入硬边界断裂。默认改成只扣通道直流偏移、不做任何抑制。"
             "但有 1/8 的情况完整去噪确实更清晰,勾这里可以切回去看。",
    )
use_minimal = minimal_default and not show_full_clean

with st.expander("调整清洗参数(仅影响下方预览,不影响已保存的分类)", expanded=is_pending_retry and not use_minimal):
    if use_minimal:
        st.caption("当前是最小处理模式(不做去噪),下面的滑块对预览没有作用。"
                   "勾上上面的「改用完整去噪版」才会生效。")
    gentle = da.gentle_params_for(row["type"])
    if gentle is not None:
        gc1, gc2 = st.columns([1, 2])
        gc1.button(f"套用 Type {row['type']} 轻度去噪", key=f"gentle_{file_name}",
                   on_click=_apply_params_to_sliders, args=(file_name, gentle),
                   help="把下面六个滑块一次性设成对 Type II/V 更宽松的一组值(少压制、多保护)。"
                        "只改预览,不改任何已保存的记录。")
        gc2.button("恢复生产默认", key=f"reset_{file_name}",
                   on_click=_apply_params_to_sliders,
                   args=(file_name, {**da.default_params(da.PRODUCTION_METHOD[source]),
                                     "known_burst_margin": da.DEFAULT_KNOWN_BURST_MARGIN}))
        st.caption(
            "⚠️ 长 Type II 上这些滑块可能完全看不出变化:保护窗口内它们的影响精确为 0,"
            "而且 burst 主要是在更上游的背景拟合阶段被吸收的(实测吸收 97%),滑块够不着那一步。"
        )
    margin = st.slider(
        "known_burst_margin(样本数)", 0, 400, preset.get("known_burst_margin", da.DEFAULT_KNOWN_BURST_MARGIN),
        step=5, key=f"margin_{file_name}",
        help="保护窗口在burst时间两侧各扩展多少个采样点。",
    )
    c1, c2, c3 = st.columns(3)
    base_threshold_sigma = c1.slider("base_threshold_sigma", 3.0, 20.0, preset["base_threshold_sigma"],
                                      key=f"bts_{file_name}")
    suppression_ramp = c2.slider("suppression_ramp", 1.0, 5.0, preset["suppression_ramp"],
                                  key=f"ramp_{file_name}")
    burst_sigma = c3.slider("burst_sigma", 0.5, 5.0, preset["burst_sigma"], key=f"bs_{file_name}")
    c4, c5 = st.columns(2)
    vrfi_coverage_threshold = c4.slider(
        "vrfi_coverage_threshold", 0.02, 0.5, preset["vrfi_coverage_threshold"], step=0.02,
        key=f"vcov_{file_name}", help="竖向(瞬时宽带)RFI 判定阈值:多大比例的频率通道同时异常才算。调低=更容易判定为RFI。",
    )
    vrfi_sigma = c5.slider("vrfi_sigma", 1.0, 6.0, preset["vrfi_sigma"], step=0.5, key=f"vsig_{file_name}")

    is_default_params = (
        margin == da.DEFAULT_KNOWN_BURST_MARGIN
        and base_threshold_sigma == da.default_params(method)["base_threshold_sigma"]
        and suppression_ramp == da.default_params(method)["suppression_ramp"]
        and burst_sigma == da.default_params(method)["burst_sigma"]
        and vrfi_coverage_threshold == da.default_params(method)["vrfi_coverage_threshold"]
        and vrfi_sigma == da.default_params(method)["vrfi_sigma"]
    )

needs_reprocess = (is_pending_retry or not is_default_params or not is_manual_default) and not use_minimal

if use_minimal:
    # no clean() call at all -- nothing here can suppress anything, so there
    # are no override params to record either
    cleaned_to_show = da.minimal_view(event["raw"])
    protect_to_show = da.protect_indices_for(effective_indices, margin, event["raw"].shape[1])
    active_overrides = None
elif not needs_reprocess:
    cleaned_to_show = event["cleaned"]
    protect_to_show = event["protect_indices"]
    active_overrides = None
else:
    reprocess_raw = event["raw"]
    window_sizes = da.default_params(method)["window_sizes"]
    result = da.reprocess(
        reprocess_raw, sample_interval_s, effective_indices,
        known_burst_margin=margin, base_threshold_sigma=base_threshold_sigma,
        suppression_ramp=suppression_ramp, burst_sigma=burst_sigma, window_sizes=window_sizes,
        vrfi_coverage_threshold=vrfi_coverage_threshold, vrfi_sigma=vrfi_sigma,
    )
    cleaned_to_show = result["cleaned"]
    protect_to_show = da.protect_indices_for(effective_indices, margin, reprocess_raw.shape[1])
    active_overrides = dict(known_burst_margin=margin, base_threshold_sigma=base_threshold_sigma,
                             suppression_ramp=suppression_ramp, burst_sigma=burst_sigma,
                             vrfi_coverage_threshold=vrfi_coverage_threshold, vrfi_sigma=vrfi_sigma)
    st.info("预览已重新处理(未保存)。")

if use_minimal:
    # must NOT reuse the production cleaned's colour scale: a row-median-
    # subtracted raw sits at raw's own magnitude (tens of counts), an order
    # above a suppressed residual, so forcing it onto that scale saturates the
    # whole panel. The "share one scale" rule exists for comparing two CLEANED
    # results with each other; this isn't one.
    right_lim = cleaned_scale(cleaned_to_show)
    # ASCII only: the matplotlib font here has no CJK glyphs, CJK titles
    # render as tofu boxes (confirmed by rendering it)
    right_title = "raw - row median (NO denoising)"
else:
    right_lim = cleaned_scale(event["cleaned"])
    right_title = "cleaned (reprocessed preview)" if needs_reprocess else "cleaned"

fig = render_pair(
    event["raw"], cleaned_to_show, sample_interval_s, event["freq_mhz"],
    effective_indices, protect_to_show, cleaned_lim=right_lim,
    titles=("raw", right_title),
)
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# classification -- 4 direct action buttons (click = save with current
# notes/tags + advance), each wired to a keyboard shortcut (see _SHORTCUT_JS)
# ---------------------------------------------------------------------------

st.divider()
notes = st.text_area("备注(可选)", value=(existing["notes"] if existing else ""), key=f"notes_{file_name}")

st.caption("勾选适用的问题标签(仅「有已知问题」分类会用到,可多选):")
tag_cols = st.columns(len(rs.PROBLEM_TAG_LABELS))
selected_tags = []
default_tags = existing.get("problem_tags_list", []) if existing else []
for col, (tag, label) in zip(tag_cols, rs.PROBLEM_TAG_LABELS.items()):
    if col.checkbox(label, value=(tag in default_tags), key=f"tag_{tag}_{file_name}"):
        selected_tags.append(tag)


def _advance():
    remaining = [f for f in file_names if f not in (done_names | {file_name})]
    if remaining:
        st.session_state[state_key] = remaining[0]
    elif cur_idx < len(file_names) - 1:
        st.session_state[state_key] = file_names[cur_idx + 1]


def _manual_range_to_save():
    if effective_indices is None or is_manual_default:
        return None
    return {"start_s": manual_start_s, "end_s": manual_end_s}


b1, b2, b3, b4 = st.columns(4)
if b1.button(_ACTION_LABELS[rs.STATUS_USABLE], use_container_width=True):
    rs.upsert_review(out_dir, file_name, rs.STATUS_USABLE, notes,
                      override_params=active_overrides, manual_burst_range=_manual_range_to_save())
    _advance()
    st.rerun()
if b2.button(_ACTION_LABELS[rs.STATUS_CASE_BY_CASE], use_container_width=True, type="primary"):
    rs.upsert_review(out_dir, file_name, rs.STATUS_CASE_BY_CASE, notes,
                      override_params=active_overrides, manual_burst_range=_manual_range_to_save())
    _advance()
    st.rerun()
if b3.button(_ACTION_LABELS["flagged"], use_container_width=True):
    if not selected_tags:
        st.warning("选「有已知问题」之前请至少勾选一个问题标签。")
    else:
        rs.upsert_review(out_dir, file_name, rs.STATUS_FLAGGED, notes,
                          manual_burst_range=_manual_range_to_save(),
                          problem_tags=selected_tags, auto_retry_done=is_pending_retry)
        _advance()
        st.rerun()
if b4.button(_ACTION_LABELS[rs.STATUS_DISCARD], use_container_width=True):
    rs.upsert_review(out_dir, file_name, rs.STATUS_DISCARD, notes)
    _advance()
    st.rerun()
