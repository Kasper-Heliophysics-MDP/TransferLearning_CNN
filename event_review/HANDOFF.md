# event_review 模块:使用说明 / 代码结构 / 开发进度与计划

写给下一个用这个模块的人(或下一次会话)看的。根目录的 `HANDOFF.md` 是整个项目的交接文档,这份只讲
`event_review/` 这一个子模块。启动方式见 `README.md`,这里补的是 README 没有的:为什么这么设计、
代码怎么分工、现在到底做完了多少、接下来具体要做什么。

## 这个模块解决什么问题

同一套清洗参数(`sumthreshold_denoise.clean()` 的 `base_threshold_sigma`、`suppression_ramp`、
`known_burst_margin` 等)不能可靠地适配所有站点/事件,数字指标本身也靠不住(eCallisto 站点筛选阶段
验证过,旧的 `rfi_quality.py` 排名对"大范围中等强度"RFI 不敏感)。这个工具把"人眼判断"这一步固化成
可持续、可追溯、可复用参数的流程。**这个工具已经不是纸上谈兵**——用它做的真实批量审核先后抓出过两个
自动化流程/数值指标都没发现的真实问题(own-station 的 known_burst_weight bug、ROSWELL-NM 数据质量、
catalog 标注时间系统性不准),细节见根目录 `HANDOFF.md`。

## 代码结构

```
event_review/
├── app.py            # Streamlit 入口,页面布局和交互逻辑
├── data_access.py     # 两种数据源的 raw/cleaned 加载,含本站"raw 重建"逻辑、建议参数表
├── plotting.py         # raw|cleaned 并排画图,色标处理
├── review_store.py     # 审核结果持久化(review_status.csv),状态/标签定义
├── requirements.txt    # 只有 streamlit
└── README.md           # 快速上手(启动命令 + 基本用法,写得早,细节没有这份新)
```

### `app.py`
页面主体。侧边栏:数据源单选(own-station / eCallisto)→ 目录输入框 → 站点/类型/审核状态筛选 →
进度统计(**"已审完"不包括"有已知问题待重试"的,那些是排到队尾的未完成项**)→ 上一个/下一个 +
事件下拉(`⏳`=待重试、`✓`=已审完、无标记=未审核)。

导航顺序:未审核 → 已审完(可重新查看) → 有已知问题待重试(排最后),不是原始 catalog 顺序,见
`_queue_rank()`。

主区域从上到下:
1. **Burst 实际时间段(数字输入,常驻显示,不是折叠的)**——开始/结束两个秒数输入框,默认值来自
   catalog 自动标注(或之前保存过的人工修正)。改动会替代红线标记,并影响 `known_burst_weight`。
   **这个不是隐藏在折叠面板里的可选功能**——真实审核发现 catalog 标注时间在几乎每个站点上都大量
   出现偏差(通常偏早),常驻显示是刻意的。
2. "调整清洗参数"折叠面板——`known_burst_margin`/`base_threshold_sigma`/`suppression_ramp`/
   `burst_sigma`/`vrfi_coverage_threshold`/`vrfi_sigma` 六个滑块(后两个是这轮新加的,原来
   `vertical_rfi_weight` 的这两个参数硬编码在 `clean()` 内部调不到,现在提成了可选参数,默认值跟原来
   硬编码值一致,不影响任何现有调用方)。如果这条事件是"待重试"状态,面板默认展开且滑块初始值是
   根据 `problem_tags` 从 `data_access.SUGGESTED_PARAMS_BY_TAG` 查出来的建议值,不是生产默认值。
3. raw | cleaned 对比图。
4. 备注 + 问题标签多选(仅"有已知问题"分类会用)+ 四个分类按钮。

`needs_reprocess`(是否要现场重新跑 `clean()`,而不是直接用生产结果)现在由三个条件的"或"决定:参数
滑块偏离默认值 / 时间段被人工修正 / 这条是待重试状态。

**分类从三选一变成四选一,而且不再是"单选+另外点保存",是四个按钮直接触发保存并跳下一个**:
- `直接可用` / `case-by-case` / `弃用`:含义不变。
- `有已知问题(待自动重试)`(新):选之前要先勾选至少一个问题标签(标注时间不准/burst偏淡被抹除/
  横向RFI残留重/竖向RFI残留重,`review_store.PROBLEM_TAG_LABELS`)。保存后这条**不计入已审进度**,
  排到队尾。下次翻到时,面板自动套用该标签对应的建议参数重新预览,人工再判断是否好转。

四个按钮绑定了数字键 1/2/3/4 快捷键(注入的 JS,监听 `window.parent.document` 的 keydown 再模拟点击
对应文字的 button——**这是个 hack,矩阵按钮文案匹配,不保证在所有浏览器/环境下都生效,已验证过重复
挂载监听器不会叠加触发多次,但没有做过完整跨浏览器测试**)。

**手动时间修正的保存逻辑**:不管最后点哪个分类按钮,只要时间段跟自动检测值不同,`manual_burst_range`
都会一起存下来(跟 `override_params` 不一样——override 只有 case_by_case 才存,manual_burst_range
不管选哪个分类都存,因为这是"这个事件的真实情况",不是"case by case 的临时处理")。

### `data_access.py`
两种数据源的核心差异,这是全模块最不直观、最容易踩坑的一块:

- **eCallisto**(`data/ecallisto/raw_events/*.npy`):`.npy` 本身就是真·raw。`load_event()` 调
  `clean_ecallisto_events.clean_event_file()` 现场清洗——**没有复用**批量跑好的
  `data/ecallisto/cleaned_events/`,每次都重新算一遍(单次 ~0.05 秒,7000+ 文件批量跑也就十分钟量级,
  现场重算的开销可以忽略;好处是审核时看到的和面板里能调的是同一条代码路径,不会因为读了预算好的
  文件而对不上)。
- **own-station**(`data/burst_data/rough_events/*.npy`):这里的 `.npy` **已经是清洗过的**产物,磁盘
  上从来没存过真正的 raw own-station crop。`load_event()` 要看 raw,得从
  `data/burst_data/csv/original/` 里对应的原始 CSV 按需要的时间范围重新读出来
  (`_load_own_station_raw_slice()`,只读需要的行范围,不整个文件加载)。

  定位到正确原始 CSV(`resolve_own_station_csv()`)要处理拼写不一致(`SequenceMatcher` 模糊匹配,
  阈值 0.8)和同日多录制 session 消歧(读候选文件 Time 列范围确认覆盖)。**已实测**:74 个本站事件
  全部走一遍,74/74 成功解析。

`_METHOD_PRESETS` / `default_params()`:生产参数预设,现在包含 `vrfi_coverage_threshold`/
`vrfi_sigma`。

`SUGGESTED_PARAMS_BY_TAG`(新):问题标签 → 建议参数覆盖的映射,给"有已知问题"分类的自动重试用。
**这是第一版猜测,写的时候零真实审核数据支撑**,只是基于对算法机制的已知理解(比如
`horizontal_rfi` 标签目前没有靠谱的参数解法,因为 `persistent_row_correction` 只修每行常数偏移不修
纹理,这是文档已经记录过的结构性限制,不是没调好参数)。跑几轮真实自动重试之后应该回来看这张表准不
准,尤其是 `horizontal_rfi` 这条大概率成功率偏低。

`reprocess()` 现在接受 `event_indices` 参数指向**调用方决定的**任何范围(可以是人工修正后的),这个
函数本身不关心是不是"真的" catalog 时间,只管拿给它的范围去建 `known_burst_weight`。

### `plotting.py`
没有大改。`render_pair()` 画 raw(viridis)| cleaned(RdBu_r)对比图,两条踩过的坑:raw/cleaned 配色
逻辑不同、多个 cleaned 结果互相比较必须共用同一色标(不然会有"看起来更干净"的假象)。图上红线标记
的时间(以及黑色保护窗口阴影)现在可能是人工修正后的范围,不一定是 catalog 原始值。

### `review_store.py`
持久化层。`review_status.csv` 按数据源目录各存一份。**这轮加了三列**:
`manual_burst_range_json`(人工修正的时间段,`{"start_s":.., "end_s":..}`)、`problem_tags`(逗号分隔
的标签)、`auto_retry_done`(布尔,标记这条"有已知问题"是否已经自动重试过一轮,避免同一条被反复用
同一套建议参数无限重试)。新列通过 `load_review()` 里"缺列就补空字符串"的逻辑,对旧版
`review_status.csv`(只有原来 5 列)向后兼容,不需要手动迁移。

新增 `STATUS_FLAGGED = "flagged_for_retry"`,`PROBLEM_TAG_LABELS` 定义四个标签的中文文案(标签取值
是这次会话真实审核 190+ eCallisto + 25 own-station 记录里实际出现的问题类型总结出来的,不是凭空
设计)。

`load_review()` 有个 pandas 坑修过:只用 `dtype=str` 读 CSV,真正空的字段还是会变成 `NaN`(float)
不是 `""`,而 `bool(float('nan'))` 在 Python 里是 `True`,导致 `if row["override_params_json"]`
这种判断拦不住,后面 `json.loads(nan)` 直接崩——**这是真实审核时踩出来的 bug,不是理论推演发现的**,
加了 `keep_default_na=False` 修好。

**没有历史记录**——同一个 `file_name` 重新审核会直接覆盖旧记录,v1 明确的范围裁剪:目的是记录"当前
判断",不是审计轨迹。多人协作审核前需要重新设计(会互相覆盖)。

## Type II/V 的特殊处理(2026-08-12 新增)

### 右面板默认不再去噪
`row["type"]` 是 II 或 V 时,右面板默认显示 `data_access.minimal_view(raw)`(**只做逐行中位数扣除,
零抑制**),而不是 `clean()` 的结果。面板上方有「改用完整去噪版」勾选框可以切回去。

**为什么**:长 Type II 上 `clean()` 会在**背景拟合阶段**丢掉 burst(实测吸收 97%),而且保护窗口内
六个滑块的影响精确为 0 —— 也就是说面板上你能调的任何东西都救不回来。完整的实测数据、以及"把
`known_burst_weight` 传进 `robust_smooth_background`"这个朴素修法为什么**是错的**(背景会整个塌成 0),
见 `ecallisto_grabber/开发日志.md` 模块四。

**为什么是默认值而不是删除**:肉眼比对 8 条已审 usable 的 II/V,cleaned 有 2/8 抹掉微弱 burst、1/8
引入硬边界断裂,但**也有 1/8 明显比 raw 更清晰**。所以保留切换开关。

**色标坑**:最小处理版的量级是 raw 自身量级(几十),比抑制过的残差高一个数量级,**必须用它自己的
`cleaned_scale()`,不能复用 production cleaned 的色标**(会整片饱和)。plotting.py 里"多个 cleaned
结果共用色标"那条规则是为了让两个**去噪**结果可比,最小处理版不属于那种情况。

**标题必须用 ASCII**:matplotlib 在这台机器上的字体没有中文字形,中文标题会渲染成方块(已实测)。

### Type II/V 的「轻度去噪」一键按钮
参数面板顶部对 II/V 多两个按钮:`套用 Type X 轻度去噪` / `恢复生产默认`,一次性设好六个滑块
(`data_access.GENTLE_PARAMS_BY_TYPE`)。每个参数的方向都是从 `clean()` 代码读出来的,不是猜的。
**但这组值没有经过效果验证**——试过用能量比值指标验证,长 Type II 上全返回 NaN、Type V 上反而报告
"更轻的设置更差",跟本项目早就记过的"自动化 burst 存活指标不可靠"是同一个失效模式,所以没有拿它
反推参数。按钮只是省拖滑块的力气,判断仍然只能靠眼睛。

只在勾选「改用完整去噪版」之后才有意义(最小处理模式下滑块本来就不参与计算)。

### 对已有审核记录的影响:零
这两处改动都只影响**渲染**,不写 `review_status.csv`,`review_store.py` 一行未动,
`_METHOD_PRESETS` / `DEFAULT_KNOWN_BURST_MARGIN` 生产默认值未动(动了会让 `is_default_params` 判定
漂移,进而影响那些存了 `override_params` 的 case_by_case 记录的含义)。改动前后 `cmp` 逐字节比对过
两个 `review_status.csv`,均 UNCHANGED,备份在 `review_status_backup_*_20260812T020026.csv`。

## 使用说明

```bash
pip install -r event_review/requirements.txt
streamlit run event_review/app.py --server.headless true --server.address localhost --server.port 8501
```

这台机器是 SSH 远程,Streamlit 默认只监听本机 localhost,本地浏览器打不开——需要 SSH 端口转发
(`ssh -L 8501:localhost:8501 ...`,或 VS Code / Cursor Remote-SSH 之类工具的自动端口转发),然后本地
浏览器开 `http://localhost:8501`。

**改代码之后要重启进程**(`pkill -f "streamlit run app.py"` 再重新起),不是刷新浏览器就行——
`@st.cache_data` 缓存在进程内存里,代码换了但底层数据文件也换了的情况下,缓存不会自动失效,刷新页面
可能还看到旧结果(这个也是真实踩过的坑,不是猜的)。

操作流程:
1. 侧边栏选数据源、筛选站点/类型/审核状态。
2. 检查"Burst 实际时间段"两个数字框,跟红线对不对得上,对不上就改。
3. 默认清洗效果不理想就展开"调整清洗参数"现场调。
4. 看 raw|cleaned 对比图,填备注,如果是"有已知问题"就先勾问题标签,点四个分类按钮之一(或按数字键
   1-4)——保存并自动跳下一条。

## 当前进度

- **核心数据流验证过**(不是只读代码猜的):`review_store.py` 的存读逻辑(含新的 manual_burst_range/
  problem_tags 字段)、`data_access.reprocess()` 带人工时间段+新竖向RFI参数,都用真实数据跑过并断言
  结果正确。
- **浏览器端交互(按钮点击、快捷键、滑块)靠用户实测反馈迭代**,不是我自己能验证的部分——已知这套
  键盘快捷键实现是 hack,没做过完整测试。
- **真实审核已经产出成果,不是空转**:第一轮用旧版三分类工具审了 223 条(25 本站 + 198 eCallisto),
  发现了 known_burst_weight bug、ROSWELL-NM 该弃用、catalog 时间系统性不准这三件事,细节和处理结果
  见根目录 `HANDOFF.md`。这 223 条记录后来应用户要求清空重新开始(用新版四分类工具),备份在
  `event_review/review_status_backup_*.csv`。
- **下游"生成最终数据"的脚本仍然不存在**——这是唯一还没做的核心缺口,见下面 Phase C。

## 下一步计划

### 用新版工具重新审核(进行中)
本站 74 条建议全审(量不大)。eCallisto 建议还是优先 positive、按站点分层抽样,不用一次性冲全量
(现在 7 站 ~7330 条,量比第一轮更大了,GREENLAND 因为审核证实质量最好已经补抓到 1029 条)。

重点关注"标注时间不准"这个标签的出现频率——如果继续像第一轮那样几乎每个站点都大量出现,可能需要
考虑批量/半自动的时间修正方案(比如用信号强度做一个粗略的自动重新定位,人工只做确认),而不是每条
都手动输入数字。

### 这个模块在新架构里的角色变了(2026-08-11)
根目录 `TRAINING_PLAN.md` 把架构改成了"COCO预训练 YOLO + 15 分钟窗口目标检测",训练数据的基本单元
从"事件 crop"变成"15 分钟窗口",而且要**重新抓取**(现有 crop 有框居中标签泄漏,见 `TRAINING_PLAN.md`
结论5)。这对本模块的影响:

- **审核记录不作废,反而更值钱了**。`manual_burst_range_json` 现在是**框标注的真值来源**,不再只是
  "清洗时保护哪一段"。事件身份通过 `(location, date, event_start_time, event_end_time, type)` 映射到
  新窗口上,换切法不影响。
- **注意 `manual_burst_range` 存的是"相对旧 crop 起点的秒数"**,而旧 crop 起点 = `event_start - 60s`
  (`extract_events.py` 的 `buffer_s`)。换算成绝对时间必须加回这个偏移,别直接当成绝对时刻用。
- `status=discard` 现在的用途是"把这个窗口整体排除"(既不能当正样本也不能当干净负样本),不只是
  "这条事件不要"。
- **`data/ecallisto/cleaned_events/` 对被修正过的事件是坏的**:生产清洗用 catalog 时间建
  `known_burst_weight`,而人工修正说明那个时间错了 —— 132 条有修正的事件里 42% 的真实 burst 区域完全
  没被保护。**所以审核时看到的"burst 偏淡/被抹除",有一部分是这个原因造成的假象,不是去噪算法的问题。**
  给 `burst_faint` 标签调参数之前,先确认时间段是不是对的。

### 写下游脚本(唯一必须写代码的部分,还没开始)
**注意:下面这个 `materialize_reviewed_dataset.py` 的设计是针对"事件 crop"数据单元写的,在新架构下
需要重新设计**——实际的转换逻辑现在归 `ecallisto_grabber/extract_windows.py`(15 分钟窗口 + 框)。
保留原设计备查,里面关于"四种 status 分别怎么处理"的判断在新脚本里依然成立:

新建脚本(暂定 `event_review/materialize_reviewed_dataset.py`),读 `review_status.csv` +
`metadata.csv`:
- `status=discard` → 排除
- `status=usable` → 有 `manual_burst_range` 就用它重新跑 `clean()`,没有就用生产默认结果
- `status=case_by_case` → 用存下来的 `override_params`(+ 如果有的话 `manual_burst_range`)重新跑
- `status=flagged_for_retry` → 这个状态本身不该出现在最终产出里(理论上应该都已经流转成上面三种之一
  了),脚本应该对遗留的 flagged 状态报警/跳过,而不是静默处理成任何一种

own-station 需要先重建 raw 再 reprocess(复用 `data_access._load_own_station_raw_slice()`)。输出
目录待定,遵循"新产出不覆盖旧数据"惯例。

## 已知的小问题/待办

- `SUGGESTED_PARAMS_BY_TAG` 是零数据支撑的第一版猜测,需要用真实自动重试结果校准。
- eCallisto 的 `load_event()` 现场重跑清洗、不复用批量生成的 `cleaned_events/`——有意为之(见上文
  "代码结构"),如果审核规模变得很大、现场重算成本上升,可以重新评估。
- `review_store.py` 没有审核历史/审核人记录,多人协作会互相覆盖。
- 键盘快捷键的 JS hack 没有做过完整测试,依赖按钮文案匹配,文案一改就失效。

## 目录漏记事件的人工标注子集(`data/ecallisto/gap_annotate/`,2026-08-14)

`event_review/make_gap_subset.py` 把 `detection/gap_review/to_annotate.csv`(126 条经人工确认、
目录漏记的 burst)做成一个可直接拖框的子集目录,跟 `fix_start/` 同一套路(软链 + 自己的
metadata.csv + 自己的 review_status.csv)。

启动:`streamlit run event_review/app.py` → 侧边栏 eCallisto → 目录填 `data/ecallisto/gap_annotate`

### 三个和 `fix_start/` 不一样的地方,都是被数据逼出来的

1. **数据单元是 15 分钟窗口,不是单事件 crop**。`load_event` 的 eCallisto 分支按
   `start_time`/`event_start_time` 的差值算 burst 在数组里的位置,所以只要
   `start_time` = 窗口起点、`event_start_time` = 窗口起点 + 模型预测的秒数,整套机制原样可用,
   不需要改 `event_review` 的加载逻辑。实测 `clean()` 在 200×3600 上只要 **0.3 秒**,不是瓶颈。

2. **一个候选一个条目,不是一个窗口一个条目**。`manual_burst_range_json` 只存**一个**
   `{start_s, end_s}`,而 117 个窗口里有 **8 个含 2-3 个漏记 burst**。按窗口建条目会把第二个
   burst 悄悄丢掉。所以文件名是 `gap<id>-<窗口名>.npy`,多个候选软链到同一个数组,
   同一个窗口会出现多次、每次预填不同的区间 —— 这是对的,它们是各自独立的标注。

3. **`type` 写成 `"gap"`,并加进了 `MINIMAL_VIEW_TYPES`**。这些 burst 目录里没有,类型**真的未知**,
   写成 "III"(最常见)是在编数据。`"gap"` 在 `event_review` 里只影响两处分支:
   `gentle_params_for()` 返回 None(用生产默认值),以及现在会走最小处理模式。
   **最小处理模式是刻意的**:`clean()` 的保护窗口会拿**模型猜的时间**去建 ——
   正是当初让 `cleaned_events/` 对已修正事件失效的那个失败模式,而这里它会在修正**正在进行时**发生。
   逐行中位数扣除既不可能抹掉 burst,也正是审核图上判断用的那个渲染。

### 一个已修的坑:预填时间不要丢小数

`shift()` 最初用 `%H:%M:%S` 格式化,**把 0.1 秒精度截断成整秒**,实测 **126 条里 112 条**的预填框
偏早 2-4 列(0.25s/列,最多 1 秒)。这个量级远低于人自身的重复性(σ≈7.8s)、任何指标都测不出来,
但**恰恰因为如此才必须修**:预填值是"看着像个正常数字"就被直接接受的那个数(已实测 38% 的修正
从没动过起点),它里面的任何系统性偏移都会原样进标注。`event_burst_indices` 本来就按有没有 `.`
自动选解析格式,保留小数零成本。修完 126 条**全部零偏差**。

### ⚠️ 写回路径还没有,标注完不会自动进训练集

`merge_subset_reviews.py` 是按 `file_name` 合并回父目录 `review_status.csv` 的,而这些
`gap<id>-...` 的 file_name **在 `raw_events/review_status.csv` 里根本不存在**(它们不是事件 crop),
按其设计会被"报告并跳过"。`refresh_boxes.py` 也只认目录事件,不知道漏记框的存在。

**所以"标注 → boxes.csv"这一段是还没写的新代码。** 标完之后需要一个脚本把
`gap_annotate/review_status.csv` 里的 `manual_burst_range_json` 转成 `boxes.csv` 的新增行
(需要决定 `type` 怎么填 —— 类型未知是个真问题,单类检测不受影响,多分类需要)。
