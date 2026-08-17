# 太阳射电暴自动检测(SunRISE Ground Radio Lab)

在 e-Callisto 全网频谱数据上训练一个太阳射电暴(Type II / III / V)**检测器**,
再迁移到本站接收机。COCO 预训练 YOLOv8 + 15 分钟固定窗口目标检测。

**结果与结论看 [`RESULTS.md`](RESULTS.md)。** 这份只讲仓库怎么组织、代码怎么跑。

| | |
|---|---|
| 数据 | 15,358 个 15 分钟窗口 ≈ 3,840 小时,3 个台站 |
| Type III | 检出率 **92%**,时间中位误差 **5.0 s**,F1 **0.57** |
| Type V(引入第二台站后)| 检出率 **4% → 55%** |
| 跨仪器零样本迁移 | **77%** 检出,无微调 |
| 目录漏记事件 | 模型发现 + 人工确认 **126** 个 |

---

## 文档地图

文档分四类。**新来的人按这个顺序读:`RESULTS.md` → `TRAINING_PLAN.md` → 具体模块的 HANDOFF。**

### 结果与方案(根目录)

| 文件 | 内容 |
|---|---|
| **[`RESULTS.md`](RESULTS.md)** | **结果汇总。所有数字、被推翻的结论、方法上的坑。先读这个** |
| [`TRAINING_PLAN.md`](TRAINING_PLAN.md) | 方案设计:为什么是检测不是分类、bounding box 怎么定义、分阶段计划、文献依据 |
| [`PHASE1_RESULTS.md`](PHASE1_RESULTS.md) | Phase 1 到 Phase 3 的**完整实验流水账**,含每一轮的原始数字和当时的判断。`RESULTS.md` 是它的提炼版 |
| [`HANDOFF.md`](HANDOFF.md) | 早期数据处理交接:own-station / e-Callisto 的去噪与抓取历史 |
| [`OVERNIGHT_PLAN.md`](OVERNIGHT_PLAN.md) | 通宵批次的设计与实际执行记录 |

### 子模块说明

| 文件 | 内容 |
|---|---|
| [`event_review/HANDOFF.md`](event_review/HANDOFF.md) | 人工审核工具的设计、代码结构、两个审核子集的构造方式 |
| [`event_review/README.md`](event_review/README.md) | 审核工具快速上手 |
| [`ecallisto_grabber/README.md`](ecallisto_grabber/README.md) | 抓取模块用法 |
| [`ecallisto_grabber/开发日志.md`](ecallisto_grabber/开发日志.md) | 抓取/去噪的踩坑记录(真实数据上发现的 bug) |

### 历史路线(已不在主线)

改用目标检测架构之前的分类 + GAN 增强方案。保留备查,未并入当前流程。

- `dcgan/` —— DCGAN / SpecGAN 数据增强
- `radburst_tl/` —— 早期迁移学习训练脚本

---

## 目录结构

```
detection/          检测主线:数据集构造、训练、评测、漏记挖掘
transfer/           本站迁移实验(Phase 3)
event_review/       人工审核工具(Streamlit)+ 审核子集生成
ecallisto_grabber/  e-Callisto 抓取 + sumthreshold 去噪
data/               频谱数组与标注(大部分不入 git,见 .gitignore)
```

`data/` 里只有**手工产生、不可再生**的 CSV 进版本控制(审核记录、漏记框、
污染负样本名单)。几 GB 的 `.npy` 和可再生的 YOLO 数据集目录都不入库,
理由写在 [`.gitignore`](.gitignore) 顶部。

---

## 主要流程

### 1. 抓取

```bash
python ecallisto_grabber/scrape_windows.py \
    --start 2021-01-01 --end 2024-03-31 --stations Arecibo-Observatory \
    --types II III V --out-dir data/ecallisto/windows
```

整天下载、按 FITS 边界切 15 分钟窗口存 `.npy`,可断点续传。
**注意 `--limit` 是全局额度不是按站分配**,多站点必须每站单独跑。

### 2. 人工审核

```bash
streamlit run event_review/app.py
```

侧边栏选数据源和目录。核心功能是**在频谱图上横向拖一段直接设定 burst 起止**——
目录标注的时间系统性偏早(实测中位 +40 秒),这是主要的修正对象。

针对性重审子集:

```bash
python event_review/make_box_subset.py data/ecallisto/windows_assa \
    --types II V --out-dir data/ecallisto/assa_iiv --apply
```

### 3. 构造数据集 + 训练

```bash
python detection/build_yolo_dataset.py data/ecallisto/windows data/ecallisto/yolo8_single \
    --train-source clean --min-window-frac 0.99 --single-class \
    --contaminated-negatives-scope train --stations Arecibo-Observatory

python detection/train_yolo.py data/ecallisto/yolo8_single/data.yaml \
    --name run1 --model yolov8s.pt --freeze-epochs 5 --epochs 30 --patience 10
```

三个不是默认值的决定(改之前先读脚本 docstring):
**按 station-day 划分而非按窗口**、**验证集只收人工审核过的干净窗口**、**几何增广全部关闭**
(频谱图翻转会破坏频率漂移方向这个核心判据)。

⚠️ **跑 ≥3 个种子**。噪声底 σ 在 0.018–0.056 之间,单次结果不可解读。

⚠️ `--optimizer` 保持默认 `auto` 时 **ultralytics 会覆盖 `--lr0-*`**,学习率不生效
(有启动警告)。要真正搜学习率必须显式指定优化器。

### 4. 评测

```bash
python detection/evaluate.py detection/runs/run1/weights/best.pt \
    data/ecallisto/yolo8_single/data.yaml --imgsz 640 --conf 0.05 --maxconf
```

双口径输出:标准 AP@0.30/0.50/0.75,加上**运营指标**(检出率、时间偏差、虚警率)——
AP 会把"完全漏掉"和"找到了但偏 21 秒"压成同一个零,而这两件事对使用者完全不同。

**所有对比必须锁死口径**:`--rect` 和 `--maxconf` 都能大幅改变数字。

### 5. 用模型找目录漏记的事件

```bash
python detection/mine_catalog_gaps.py detection/runs/run1/weights/best.pt \
    data/ecallisto/windows detection/gap_review --min-conf 0.10
# 人工填 gap_candidates.csv 的 verdict 列后:
python detection/apply_gap_verdicts.py detection/gap_review/gap_candidates.csv --apply
python event_review/make_gap_subset.py --apply          # 在审核工具里画框
python detection/apply_gap_annotations.py --type-override gap0067=II --apply
```

参考目录漏记约 36%,这条链路是补充干净标注最快的路径。
**模型画的框不会直接进训练集**——它只是预填值,时间范围由人工重画。

### 6. 本站迁移

```bash
python transfer/make_own_windows.py --apply        # 连续 CSV -> 真实 15 分钟窗口
python transfer/zero_shot_transfer.py detection/runs/run1/weights/best.pt
```

对比两种渲染:物理正确的频段定位 vs 朴素拉伸。**实测朴素拉伸远好于物理对齐**
(77% vs 14%),细节见 `RESULTS.md`。

---

## 环境

- Tesla T4(15 GB),PyTorch 2.12 + CUDA,ultralytics 8.4.80
- 机器只有 **4 个 CPU 核、15 GB 内存、无 swap**。训练默认 8 个 dataloader worker 会
  超额占用;与其他任务并行时用 `--workers 2`。审核工具占约 4 GB,训练期间不要开
  (曾被 OOM 杀掉两次)。

## 一个反复踩到的坑

`pgrep -f <脚本名>` / `pkill -f <脚本名>` **会匹配到调用它的 shell 自己的命令行**。
本项目因此损失过 4 小时(等待脚本自锁)和两次误杀。写等待/清理逻辑时用端口
(`ss -lptn 'sport = :8501'`)或更具体的匹配模式,不要用脚本名。
