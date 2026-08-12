# 会话交接:下一步该干什么

写给下一个对话开头看的。这个文件只讲"现在卡在哪、下一步做什么",完整的设计历史/踩过的坑看
`ecallisto_grabber/开发日志.md`,不在这里重复。

## 项目目标(一句话)

先在 eCallisto 全网数据上 transfer learning 预训练,再迁移到自己站点的小数据集上做太阳射电暴分类(Type II/III/V)。另有 DCGAN 数据增强分支(`radburst_tl/` 下)。

## 重要决定:去噪和"细切+缩放到固定像素"是两个独立步骤,不再绑在一起

`BurstFixedWindowSlicer.resize_window` 会把裁出来的窗口用 `cv2.resize` 同时压缩两个轴到 128×128
(频率 3.2×、时间 18.75×)。问题是本站频段 ~16-24MHz,eCallisto 各站点频段差异很大且常常不重叠
(比如 AUSTRIA-UNIGRAZ 是 45-81MHz,SWISS-SCAN 甚至是 10610-11246MHz 微波段)——如果每个来源独立
resize,同样的真实漂移率在不同站点会变成不同的像素斜率,这对 Type III 这种"斜率即特征"的分类是
致命的。`ecallisto_grabber/extract_events.py` 从一开始就刻意不做这件事。own-station 这边也已经改成
同样的策略:**去噪 + 按物理时长粗裁剪,原生分辨率保存,不做固定窗口细切,不做 resize**。等以后设计
出跨站点频段对齐方案后,细切+缩放作为统一的下游步骤一次性对两个来源生效。**这个对齐方案还没设计**,
是目前最大的未决问题。

## 现状总览

own-station 和 eCallisto 现在都完成了"去噪 + 粗裁剪"这一步。

> **重要:下面"卡在频段对齐"这个判断已经作废,不要再照它执行。** 架构在 `TRAINING_PLAN.md` 里改成了
> "COCO预训练 YOLO + 固定时长画布做目标检测",频段对齐降级成 Phase 3(迁移到本站)才需要面对的问题,
> **不再是阻塞项**。当前实际的工作是重抓 Arecibo 的 15 分钟窗口并跑通 Phase 1。以 `TRAINING_PLAN.md`
> 为准,这份文档保留下面的历史记录只是为了说明数据是怎么来的。

~~own-station 和 eCallisto 现在都完成了"去噪 + 粗裁剪"这一步,数据已经就绪,**卡在频段对齐方案没设计**,
细切+缩放、正式训练都要等这个定下来。~~

### own-station
- **`data/burst_data/rough_events/`**(标准产出):74 个原生分辨率、已去噪 `.npy`,不 resize。**这批
  是修过 known_burst_weight bug 之后重新生成的第二版**——第一版
  `extract_events_own_station.py` 有个严重 bug:全文件去噪那一步根本没把 catalog 里的真实 burst
  时间传进去做保护,等于每个文件的每一个 burst 全靠不可靠的自动检测(`burst_weight_2d`)硬扛,后果
  是不少真实 burst 信号被直接抹成空白。这个 bug 是靠 `event_review` 工具的**真实审核**发现的——6
  条真实审核里 5 条报告"burst被白框抹去",不是靠读代码或数值指标发现的,又一次印证"要在真实数据上
  验证"这条原则。已修复(现在会先扫一遍这个文件里全部 catalog burst,建一个覆盖所有 burst 的
  `known_burst_weight`,再跑一次 `clean()`),74 个文件已用修复后的代码重新生成,修复前后对比图证实
  burst 区域从"基本空白"变成"清晰可见结构"。**如果你看到的 `rough_events` 时间戳早于这次修复,那是
  旧的有 bug 版本,不能用。**
- `data/burst_data/csv/gan_training_windows_sumthreshold/`(258→259 个 128×128 window,新去噪算法+
  旧细切逻辑)和 `gan_training_windows_128/`(旧算法基线)保留但不是标准流程,能应急喂纯站内 SpecGAN
  训练。

### eCallisto(这次会话新完成的部分)
- **站点筛选**:原计划直接用 `rfi_quality.py` 的排名,但验证发现它对"大范围中等强度"RFI 不敏感——
  实测 ALASKA-HAARP 旧指标显示"中等"(h=0.13),新算法测出 **92.9%** 像素需要重度抑制(11 个候选里
  最差),肉眼看原始频谱图也确实布满持续性横纹,两边吻合。同样问题出在 NORWAY-EGERSUND(60%)和 BIR
  (52%)。最初选了新旧指标都认可干净、且频段跟本站 16-24MHz 有重叠的 8 个站点,抓取完之后靠
  `event_review` 工具的真实人工审核又做了一轮调整(见下一条)。
- **规模化抓取分三段完成**,原因是 `scrape.py` 的 `--limit` 是全局的,按 (station,date) 字母序处理,
  第一个站点如果自己的可抓天数就够多,会把额度全部吃掉,轮不到后面的站点——Arecibo-Observatory 和
  Australia-ASSA 各自吃满了一整批 500:
  - Phase 1: Arecibo-Observatory,500 station-day
  - Phase 2: Australia-ASSA,500 station-day
  - Phase 3: 剩余 6 站每站单独跑 `--limit 65`(保证每站都有数据,不再赌顺序)
- **人工审核发现 ROSWELL-NM 数值指标筛选阶段漏检**:数值指标(新旧都测过)对 ROSWELL-NM 评分中等,
  没有被筛选阶段排除;但真实人工审核 14 条,**14 条全部弃用**,0 例外。已从 `raw_events`/
  `cleaned_events`/metadata/review_status 里完全移除。这是"数值指标 + 单样本抽查"两道筛选都没拦住、
  只有大批量真实审核才发现的问题,进一步说明筛选阶段的抽样(每站 1 个样本)不够,量大之后还是要靠
  真实审核兜底。
- **GREENLAND 人工审核证实是这批里信号质量最好的站点**(跟筛选阶段数值指标结论一致),已补抓到该
  站点全部可用范围:从最初的 298 条(phase 3 的 65 天上限)扩到 **1029 条**。
- **最终 7 站合计约 7330 个事件**(Australia-ASSA、Arecibo-Observatory、SWISS-Landschlacht、
  EGYPT-Alexandria、SWISS-MUHEN、SPAIN-PERALEJOS、GREENLAND;ROSWELL-NM 已移除),存在
  `data/ecallisto/raw_events/`。
- **去重逻辑在新站点上继续验证有效**:Arecibo-Observatory、Australia-ASSA 都触发了"同一时刻多接收机
  重复文件"的去重(之前只在 ALASKA-HAARP、AUSTRIA-UNIGRAZ 上见过),以及 Australia-ASSA 的"当天混用
  200/400 通道配置"(开发日志提过的已知问题),都被现有逻辑正确处理,没有新故障模式。
- **批量清洗完成**:`ecallisto_grabber/denoising/batch_clean_ecallisto.py`(新增,仓库里没有对应旧
  文件——`clean_ecallisto_events.py` 自带的 `clean_all_events()` 是给交互式画图用的,会把全部文件的
  完整中间结果字典同时留在内存里,这么大批量这样跑内存会爆,即使单个文件很小;新脚本流式处理,清洗
  完立刻存盘再丢弃)。全部事件用 comprehensive 模式(eCallisto 单个 crop 只有 200×几百到几千,跟
  own-station 动辄 50 万行完全不是一个量级,内存和耗时都不是问题),**0 失败、0 非有限值**。输出在
  `data/ecallisto/cleaned_events/`(比 raw 大是因为 raw 是原始整数编码、cleaned 是 float32 残差,
  预期之内)。

### `event_review/`——人工审核工具,这次会话从"写完没用过"变成实际驱动决策的工具
独立子模块,详细的代码结构/设计/进度看 `event_review/HANDOFF.md`,这里只记关键结论:

- **靠真实审核抓出了一个严重 bug**(见上面 own-station 部分的 known_burst_weight 问题),不是数值
  指标或读代码发现的。
- **靠真实审核(190+ eCallisto + 25 own-station)发现一个跨站点的系统性问题**:catalog 标注的 burst
  时间点经常不准(通常偏早),这个问题在 Arecibo、Australia-ASSA、SWISS-Landschlacht、SWISS-MUHEN、
  SPAIN-PERALEJOS、GREENLAND 上都大量出现,不是某个站点特有的。为此新增了**人工时间段修正**功能
  (图上方两个数字输入框,替代 catalog 自动标注,修正后会实际影响 `known_burst_weight`,不管最后
  分类选哪个都会保存)。
- 分类从三选一扩展成四选一(直接可用/case-by-case/**有已知问题(待自动重试)**/弃用),新增问题标签
  多选(标注时间不准/burst偏淡被抹除/横向RFI残留重/竖向RFI残留重)、按钮+快捷键(1-4)、点击即保存
  并跳下一条。"有已知问题"这个分类不计入审核进度、排到队尾,下次翻到时会自动套用该标签对应的建议
  参数重新预览。
- 为了让"竖向RFI残留重"这个标签真能对应到可调参数,`sumthreshold_denoise.clean()` 新增了
  `vrfi_coverage_threshold`/`vrfi_sigma` 两个可选参数(之前硬编码在函数内部,外部完全调不到),默认值
  跟原硬编码值一致,不影响任何现有调用方。
- **之前测试阶段积累的 223 条审核记录已应用户要求清空重新开始**(备份在
  `event_review/review_status_backup_*.csv`,没有真的丢),当前 `review_status.csv` 是全新的、用
  新版四分类工具审核的状态。
- **下游脚本仍未实现**:读 `review_status.csv` 按分类结果(含人工修正的时间段、case_by_case 的
  override 参数)生成最终训练数组,这一步还没写,等审核积累到一定量再做。

## 下一步

### 当前真正在做的事:见 `PHASE1_RESULTS.md`(结果+瓶颈+明天计划)和 `TRAINING_PLAN.md`(方案背景)

**Phase 1 四组已跑完**:Type III 检出率 95%、时间中位偏差 10.7s,但 AP50 只有 0.235 ——
因为标注本身的天花板就是 0.485。瓶颈在标注精度和验证集规模,不在模型。四组之间的差异落在噪声内,
排不出名次。完整数字、五个瓶颈、三处更正、明天的 P0/P1/P2 全在 `PHASE1_RESULTS.md`。

### 原计划(部分已被上面推翻)
架构改成目标检测之后,工作重心从"数据预处理"转到"构造检测训练集"。当前顺序:

1. 重抓 Arecibo 的 **15 分钟窗口**(9 小时 / 6.6 GB),因为现有 crop 有致命的**框居中标签泄漏**
   (1694 个正样本框心 100% 落在图正中央,标准差 0 —— crop 定义就是"事件 ± 60s"的必然结果)。
   在这份数据上跑出来的 mAP 无法解读,Phase 1 也就失去意义
2. 跑通 Phase 1(Arecibo 单配置 + YOLOv8),拿一个可解读的 mAP
3. Phase 2/3 见 `TRAINING_PLAN.md`

**`data/ecallisto/cleaned_events/` 对被人工修正过的事件是坏的,不要直接用来训练**:生产清洗用的是
catalog 时间建 `known_burst_weight`,而人工修正恰恰说明那个时间错了 —— 132 个有修正的事件里,42%
的真实 burst 区域**完全没有被保护**(纯算术统计 + 肉眼确认 burst 被抹掉)。这不是代码 bug(算法忠实
保护了传给它的区间),是数据过期。Phase 1 直接用 raw。

### 已降级(不再是阻塞项):跨站点频段对齐
只在 Phase 3(迁移到本站)才需要。技术方案已查证(公共频率网格插值),7 个站点跟本站 16-24MHz 全部
有 7-8MHz 重叠。细节和已知限制见 `TRAINING_PLAN.md`。留下当初的开放问题备查:
- 统一到哪个物理频段/分辨率(本站 16-24MHz;eCallisto 站点间跨度大,同一站点不同天配置还可能不一样)
- 完全不重叠频段的数据要怎么处理
- 频率轴插值/重采样策略,同样要在真实数据上肉眼验证,不要只看数值

### 仍在做:用 `event_review` 批量人工审核
本站 25 条 + eCallisto 190+ 条已经审过一轮(旧版三分类工具,记录已按用户要求清空重来,备份见上面
eCallisto 部分),现在用新版四分类+人工时间修正的工具重新开始。继续审下去,重点关注:
- "标注时间不准"这个标签出现频率——如果继续像第一轮那样几乎每个站点都大量出现,可能要考虑做一个
  批量/自动化的时间修正方案,而不是每条都手动调
- 攒够数据后把下游脚本(读 review_status.csv 生成最终训练数组)写出来

### 其他,不阻塞但值得做
- 是否需要扩大 eCallisto 站点/时间范围,取决于架构决定(用户提到过要不要参考 ImageNet 预训练+少量
  同领域微调这种思路,而不是单纯堆量——还没深入讨论,两篇参考论文在仓库根目录)

## 提醒(避免走回头路)

- **数据集构造方式本身会制造标签泄漏,而且不会报错**——"事件 ± 固定 buffer"裁出来的 crop,框心必然
  100% 在正中央(实测 std=0.0000,n=1694)。做检测/定位任务前一定要先统计一遍标签在图里的位置分布,
  这是一行代码的检查,但漏掉的话整个 Phase 1 的 mAP 都是废数字。**同理适用于任何"以目标为中心裁剪"
  的数据集**,包括以后给本站做窗口构造的时候
- **原始 FITS 没有缓存过**——`extract_events.py` 是边下边裁只存 crop。任何"想换个切法"的需求都要
  重新下载。这次已经决定改成缓存"事件所在 15 分钟槽 ± 2 槽"(全 7 站才 19.8 GB),以后别再只存裁好的
  结果。`fetch_station_day()` 反正是整天下载,缓存粒度只影响磁盘不影响耗时
- **估算存储前先看真实的 files/day**——最初按 24 小时观测估 Arecibo 全天缓存 29 GB,实测中位数只有
  49 个文件/天(约 12 小时,站点只在本地白天观测),真实 15.2 GB,差了一倍
- 根目录 `burst_data.zip`(17.6 GB)是 `data/burst_data/` 的压缩备份,**用户已确认以后可以删除**,
  磁盘紧张时是现成的余量

- **改动之后要在真实数据上重新生成图确认,不要凭"逻辑上应该没问题"下结论**——这次的内存 bug、NaN
  bug、站点筛选的旧指标盲区,全部是靠"在真实数据上跑一次"和"检查全部输出"发现的,不是靠读代码推理
  出来的。
- **"全文件一次性去噪、之后按 catalog 时间裁剪出多个 burst"这种效率优化,必须确认 catalog 时间真的
  作为 known_burst_weight 传进了去噪那一步,不能只在裁剪阶段用**——`extract_events_own_station.py`
  就在这里栽过一次:时间索引算出来了,但只用来定位裁剪范围,从没喂给 `clean()`,等于全文件在没有任何
  ground truth 保护的情况下跑自动检测。**这个 bug 是靠人工审核工具(`event_review`)第一批真实审核
  就抓出来的**,再次证明"肉眼判断"比"看起来逻辑没问题"可靠——写完新流程之后,找几个真实样本肉眼过一
  遍再批量跑,不要批量跑完才第一次看结果。
- **`clean()` 返回 dict 里的中间数组是为了调试/画图设计的,批处理场景一定要用
  `return_intermediates=False`,而且不要用 `clean_all_events()` 处理大批量**——它会把所有文件的
  结果同时留在内存里,单个文件多小都扛不住数量堆起来,`batch_clean_ecallisto.py` 是流式处理的正确
  参考写法。
- **任何统计量(median/MAD/mean)只要不是显式 NaN-aware,一个坏点就可能污染一整行/一整个通道,而
  不是只污染坏点附近**。
- **resize/固定尺寸化是"跨站点对齐之后"才该做的事**,不要在对齐方案定下来之前又把它加回标准产出
  流程里。
- **RFI 质量筛选一定要用当前验证过的算法重新测,不能信旧的 `rfi_quality.py` 排名**——对"大范围中等
  强度、分散式"RFI 不敏感,这次在 ALASKA-HAARP/NORWAY-EGERSUND/BIR 上又验证了一遍(第一次是这次会话
  之前就发现的)。
- **就算数值指标(新旧都测过)+ 单样本人工抽查都通过,批量之后还是可能出问题**——ROSWELL-NM 筛选
  阶段两道指标都没拦住,真实批量审核 14 条却 14 条全弃用。单站点抽 1 个样本不能代表这个站点的整体
  质量,后续如果还要筛选新站点,批量抓取前最好先审个 10+ 条样本再决定,不要只信 1 条抽查 + 数值分数。
- **`scrape.py` 的 `--limit` 是全局额度,不是按站点分配**——站点列表按字母序处理,数据量大的站点会
  吃掉整批额度,想保证多站点覆盖必须每站单独跑。
- 内存:15GB 机器、无 swap,own-station 最大文件(126万行)comprehensive 模式峰值 12.3GB 余量很紧,
  批量跑倾向用 fast 模式;eCallisto 单个 crop 很小,内存完全不是问题,批量清洗直接用 comprehensive。
- 肉眼比较比数值指标可靠——有视觉判断入口就优先用视觉判断。
- own-station 和 eCallisto 用的 `known_burst_weight` 机制是**必需项不是可选优化**——纯自动检测在
  不同站点上可靠度差异很大(盲测保留比例测出过 20%~74% 的跨度),没有 ground truth 时不能直接信。
