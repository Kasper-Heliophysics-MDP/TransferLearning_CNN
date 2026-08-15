# 交接:通宵运行方案(2026-08-13)

> **2026-08-14 04:49 已实际启动。下面第二节的方案有三处被改掉了,读结果前先看文末
> "实际执行(2026-08-14)"一节** —— 尤其是**不要用 `yolo6_single` 评测**,它的 val 被
> `--drop-contaminated-negatives` 改过,和 0.361 那个基线不可比。

写给下一次会话/明早的人。上一次会话的 context 快满了,所以把"接下来跑什么、怎么跑、
跑完看什么"完整写下来。**这份文档里的脚本还没有运行过**,需要人确认后再启动。

结果与瓶颈诊断看 `PHASE1_RESULTS.md`,方案背景看 `TRAINING_PLAN.md`,代码踩坑看
`ecallisto_grabber/开发日志.md`。

---

## 一、还需要继续审核吗?

**需要,但优先级已经从"最高"降到"第二",而且该换个审法。**

当前审核量:**1097 条,721 条有人工时间修正**。

| 类型 | 已审 | 剩余 | 判断 |
|---|---|---|---|
| Type II | 53/53(100%) | 0 | **审完了,Arecibo 历史上就这么多** |
| Type V | 27/27(100%) | 0 | **同上** |
| Type III | 1016/1614(63%) | **598** | 还有价值,但边际收益在下降 |

### 为什么优先级降了

1. **稀有类已到顶。** II/V 全审完了,而 V 只有 29 个干净样本 —— val 拿 20 个,train 就只剩 9 个,
   **训练和评估无法同时满足**。这不是审核能解决的,是数据源问题(要靠 Phase 2 引入
   Australia-ASSA,它独有 55 个 II + 32 个 V)。
2. **目录本身漏记 36%。** 继续审目录里已有的事件,补不上目录没记的那部分 ——
   而那部分按测量是**已记录事件的 36%**。

### 更高价值的替代:用模型反向找目录漏记的事件

实测:模型在 **conf≥0.1 上判断"这里有 burst"的准确率是 24/24**(另一批 16 个负样本窗口里
14/16 正确)。所以可以**让模型扫全部窗口,把"高置信度但目录没有"的检出列出来,人工只做确认**。

这比从头审快得多,而且直接补的是目录的 36% 缺口 —— 那是当前最大的一块空白。
脚本还没写(见下面"第四部分:还没做的事")。

### 结论
- **今晚不需要审核**,通宵方案是纯 GPU 任务
- 明天如果要审,**优先做"模型辅助找漏记"而不是继续按目录顺序审 Type III**

---

## 二、通宵方案(约 6-8 小时,纯 GPU,无人值守)

### 设计意图

四轮实验下来,还有三件"该做但一直没做"的事,都是**只有跑才知道答案、且互不依赖**的:

1. **污染负样本剔除的效果** —— `yolo6_single` 已经建好(剔了 90 个含 burst 的假负样本),
   但**还没训练过**,不知道有没有用。
2. **超参一次都没搜过** —— 学习率、epoch、冻结策略全是随手定的,这在小数据集上通常值几个点。
3. **640×1280 的公平对比** —— 之前测过一次,但用的是刷新前的旧数据,和分辨率混在一起,
   结论作废(见 `PHASE1_RESULTS.md`)。

外加一件**必须做的**:任何结论都要有噪声底做参照,而当前噪声底(σ=0.011)是在
**旧数据、243 训练框**上测的,现在是 570 框,得重测。

### 脚本

保存为 `detection/run_overnight.sh`,内容如下(**尚未创建**):

```bash
#!/bin/bash
# 通宵实验:污染剔除效果 + 噪声底 + 超参 + 分辨率
# 用法: nohup bash detection/run_overnight.sh > detection/overnight.log 2>&1 &
cd /home/ubuntu/Desktop/TransferLearning_CNN
set -u

run () {  # run <name> <data.yaml> <extra args...>
  local name=$1 data=$2; shift 2
  echo "########## $name ##########"
  python3 -u detection/train_yolo.py "$data" \
    --name "$name" --model yolov8s.pt --project detection/runs \
    --freeze-epochs 5 --epochs 30 --patience 10 --imgsz 640 --batch 16 "$@"
}

# --- A. 污染负样本剔除的效果(vs v5_single 基线 AP50 0.361) ---
run o_clean_neg data/ecallisto/yolo6_single/data.yaml --seed 0

# --- B. 噪声底:同配置换 3 个种子(当前数据规模下重测) ---
for s in 1 2 3; do run o_seed$s data/ecallisto/yolo6_single/data.yaml --seed $s; done

# --- C. 超参:学习率 x 冻结策略(6 组) ---
#   train_yolo.py 现在把 lr 写死(stage1 1e-3 / stage2 1e-4),要先加 --lr0 参数,见下
run o_lr3   data/ecallisto/yolo6_single/data.yaml --seed 0 --freeze-epochs 0 --epochs 40
run o_lr4   data/ecallisto/yolo6_single/data.yaml --seed 0 --freeze-epochs 10 --epochs 40
run o_ep60  data/ecallisto/yolo6_single/data.yaml --seed 0 --epochs 60 --patience 20

# --- D. 分辨率:640x1280 的公平对比(同一份 boxes,只改渲染尺寸) ---
python3 detection/build_yolo_dataset.py data/ecallisto/windows data/ecallisto/yolo6_wide \
  --overwrite --train-source clean --min-window-frac 0.99 --single-class --img-size 640 1280
run o_wide data/ecallisto/yolo6_wide/data.yaml --seed 0 --imgsz 1280 --rect --batch 8

echo "OVERNIGHT DONE"
```

### 启动前必须先做的两件事

1. **确认没有其他训练在跑**(GPU 只有一块 T4):
   ```bash
   ps -eo pid,etime,cmd | grep "[t]rain_yolo.py"
   ```
2. **`train_yolo.py` 目前把学习率写死了**,C 组要真正搜超参需要先加 `--lr0` / `--lr0-stage2`
   两个参数(现在是 stage1 `lr0=1e-3`、stage2 `lr0=1e-4`)。**不加也能跑**,只是 C 组变成
   "只搜冻结策略和 epoch 数",价值减半。

### ⚠️ 三个已经踩过的坑,别再踩

1. **`pgrep -f "train_yolo.py"` 会匹配到自己的 bash 命令行**。今晚这个脚本用顺序执行而不是
   轮询等待,就是为了绕开它 —— 上一轮有个等待脚本因此自锁了 4 小时,把后续实验全挡住。
   如果非要写等待逻辑,用 `pgrep -f "train_yolo.py --data"` 之类更具体的模式,或者检查 GPU 占用。
2. **内存只有 15GB,训练会把 streamlit 挤掉**(已经被 OOM 杀过两次,exit 137)。
   通宵期间不要开审核工具。
3. **磁盘剩 70GB**。每个 `detection/runs/*` 约 40MB,这批 9 个跑完约 360MB,没问题。
   但 `data/ecallisto/yolo*` 已经有 12 个数据集目录,**该清理了**(见第五部分)。

---

## 三、明早看什么

### 1. 先确认都跑完了
```bash
grep -c "OVERNIGHT DONE" detection/overnight.log
ls -d detection/runs/o_*/ | wc -l          # 应为 9(不含 _frozen)
```

### 2. 统一口径评测(必须锁死 `rect`,否则数字不可比)

**这是踩过的坑**:`rect` 是 ultralytics `val()` 的开关,默认 `True`,而它能让同一份权重的
AP50 从 0.176 变到 0.203。**所有对比都必须显式传同一个值。**

```bash
for n in o_clean_neg o_seed1 o_seed2 o_seed3 o_lr3 o_lr4 o_ep60; do
  echo "=== $n ==="
  python3 detection/evaluate.py detection/runs/$n/weights/best.pt \
    data/ecallisto/yolo6_single/data.yaml --imgsz 640 --conf 0.05 --class-agnostic \
    2>&1 | grep -E "^burst|detected|false alarms"
done
python3 detection/evaluate.py detection/runs/o_wide/weights/best.pt \
  data/ecallisto/yolo6_wide/data.yaml --imgsz 1280 --conf 0.05 --class-agnostic
```

### 3. 判读标准(先算噪声底,再看差异)

用 `o_seed1/2/3` 加上 `o_clean_neg`(seed 0)算标准差 σ。**然后:**

- **差异 < 2σ 的一律当作没有差异** —— 上一轮就是因为没先算 σ 而把 `rect` 造成的抖动
  当成了真实差异,得出"C 方案赢了"的错误结论,后来收回。
- **n=4 估的 σ 本身很不准**(上一轮 n=3 时 95% 区间是 [0.006, 0.069]),
  所以宁可保守。

### 4. 各组分别回答什么

| 组 | 对照 | 回答 |
|---|---|---|
| `o_clean_neg` | v5_single AP50 **0.361** | 剔除 90 个污染负样本有没有用 |
| `o_seed1/2/3` | 彼此 | 当前数据规模下的噪声底 σ |
| `o_lr3` / `o_lr4` / `o_ep60` | `o_clean_neg` | 冻结策略和 epoch 数值不值得调 |
| `o_wide` | `o_clean_neg` | 时间分辨率翻倍是否改善定位(AP@0.75 最敏感) |

### 5. 别忘了 maxconf 后处理

`evaluate.py` **还没有**内置 maxconf(每簇只保留置信度最高的框)。实测它把
F1 从 0.311 提到 **0.522**,是能部署的口径。上面的评测命令报的是原始口径,
两者相差很大,**别把原始口径当成最终能力**。把它写进 `evaluate.py` 是明天的 P1。

---

## 四、还没做的事(按价值排序)

1. **模型辅助找目录漏记事件**(脚本未写)。模型 conf≥0.1 时准确率 24/24。
   做法:扫全部 6532 个窗口 → 列出高置信度但目录没有的检出 → 人工确认 → 补进标注。
   **这是当前补充干净标注最快的路径**,也直接补目录的 36% 缺口。
2. **把 maxconf 写进 `evaluate.py`**(F1 0.311 → 0.522)。
3. **Phase 2:引入 Australia-ASSA**。它独有 55 个 II + 32 个 V,是唯一能救 Type V 的路
   (Arecibo 的 27 个样本无论怎么分都不够)。抓取约 17 小时 / 7.9 GB,
   `scrape_windows.py` 现成可用。
4. **一维时间轴分割**替代目标检测框(`PHASE1_RESULTS.md` P2 第 1 条)。
   框在 640 图上是 21-49 像素宽 × 640 高,长宽比 1:13~1:30,YOLO 不是为这种形状设计的。
   这是结构性改动,不是调参。

---

## 五、顺手该清理的

```bash
du -sh data/ecallisto/yolo*        # 12 个数据集目录
```

`yolo_phase1` / `yolo_phase1_debias` / `yolo_noisy` / `yolo_clean` / `yolo_both` / `yolo_wide` /
`yolo2_*` / `yolo3_clean` / `yolo4_clean` 都是历史实验产物,**结论已记录在文档里,数据集本身
可以删**(随时能用 `build_yolo_dataset.py` 重建,只需几分钟)。保留 `yolo5_*` 和 `yolo6_*` 即可。

`detection/runs/` 下 27 个目录同理,`.gitignore` 已经排除它们,但占磁盘。

---

## 六、当前资产快照(2026-08-13)

- **审核 1097 条**(其中 **721 条**有人工时间修正),全部 Arecibo
- 窗口数据 `data/ecallisto/windows/` 6609 个 15 分钟窗口(4.5 GB)
- **已知的三个人工核查结论**(都在 `PHASE1_RESULTS.md` 里有完整数据):
  - 人工标注自我一致性 **88%**(IoU≥0.5),但有水分(44% 是"两次都没动起点")
  - 目录漏记率:34 个"虚警"里 **32 个(94%)** 是真 burst
  - 负样本污染:16 个被标记的窗口里 **14 个(88%)** 确实含 burst
- **最好的模型**:`detection/runs/v5_single/weights/best.pt`
  (单类,570 训练框,AP@0.50 **0.361**、检出率 88%、时间中位偏差 **7.8s**)
- git:三个 commit 已推到 `origin/main`(`9876862` 为最新),审核数据已纳入版本控制

---

## 实际执行(2026-08-14 04:49 启动)

脚本 `detection/run_overnight.sh` 已创建并启动,日志 `detection/overnight.log`。
相对第二节的方案有**三处改动**,都是启动前的核对发现的,不是随手改的。

### 改动一(最重要):换数据集 `yolo6_single` → `yolo7_single`

`yolo6_single` 的 **val 被污染剔除改过了:581 → 565 张**。`--drop-contaminated-negatives`
当时同时作用在 train 和 val 上,而被剔掉的那 16 个 val 窗口**正是模型会开火的那些**
(它们本来就是靠"模型在负样本上有检出"选出来的)。拿它去和 v5_single 的 **0.361** 比,
测到的是"我们删掉了 16 个会产生虚警的窗口",不是"清理训练负样本有没有用" —— **两个变量同时动了**。

处置:`build_yolo_dataset.py` 新增 `--contaminated-negatives-scope {both,train}`
(默认 `both`,保持已建数据集可复现)。剔除改到 val 划分**之后**执行,并豁免 val 天,
理由和下面那段 de-bias 的注释是同一个:**val 是尺子,必须跨实验固定**。

新建 `yolo7_single`,并**逐字节验证**(不是只看数量):

```
val 图像列表   vs yolo5_single: IDENTICAL
val labels/    vs yolo5_single: IDENTICAL   (diff -r -q)
val images/    vs yolo5_single: IDENTICAL   (diff -r -q)
train 列表     vs yolo6_single: IDENTICAL
train 框数     570(与 yolo5 相同,只掉了 74 个负样本窗口)
```

所以 `o_clean_neg` vs `v5_single 0.361` 现在是**单变量**对比。
`yolo7_wide` 同法建(640×1280),labels 与 `yolo7_single` 逐字节相同,只有渲染尺寸不同。

### 改动二:**boxes.csv 故意没有刷新**

`review_status.csv`(08-13 03:32)比 `boxes.csv`(08-13 01:07)新,有约 90 条审核没进 boxes。
**没有跑 `refresh_boxes.py`** —— 刷新会改变 val 划分,那样今晚每一组的对照基线 0.361 就全部失效。
**这批结果读完之后再刷新**,不要在读之前刷。

### 改动三:C 组变成真正的学习率搜索

`train_yolo.py` 新增 `--lr0-stage1`(默认 1e-3)/ `--lr0-stage2`(默认 1e-4),
两个默认值与此前写死的值一致,**不影响任何已有结果的可复现性**。
已烟测确认参数真的传到 ultralytics(`args.yaml` 里 `lr0: 0.007` / `0.0005`),不是只加了个没接线的开关。
顺带修了 `--seed` 帮助串里一个 `%` 导致 `--help` 直接崩溃的老 bug(只影响 `--help`)。

C 组因此调整为:`o_lr_hi`(stage2 3e-4)、`o_lr_lo`(stage2 3e-5)、`o_nofreeze`(freeze 0 / 40ep)、
`o_ep60`(5+60 / patience 20)。原方案里 `o_lr3`/`o_lr4` 其实只在改冻结策略、并没有动学习率。

### 九组清单(顺序执行,按价值排序,前面的先跑完)

| 顺序 | 组 | 数据集 | 对照 | 回答什么 |
|---|---|---|---|---|
| 1 | `o_clean_neg` | yolo7_single | v5_single **0.361** | 剔除 74 个污染训练负样本有没有用 |
| 2-4 | `o_seed1/2/3` | yolo7_single | 彼此 + 上面 seed 0 | **噪声底 σ(n=4)** |
| 5 | `o_lr_hi` | yolo7_single | `o_clean_neg` | stage2 学习率 ×3 |
| 6 | `o_lr_lo` | yolo7_single | `o_clean_neg` | stage2 学习率 ÷3 |
| 7 | `o_nofreeze` | yolo7_single | `o_clean_neg` | 冻结那 5 轮值不值 |
| 8 | `o_ep60` | yolo7_single | `o_clean_neg` | 60 epoch 会不会更好 |
| 9 | `o_wide` | yolo7_wide | `o_clean_neg` | 时间分辨率翻倍(看 AP@0.75) |

预计 7-8 小时(v5 实测 5+30 epoch ≈ 38 分钟,早停会砍掉一部分)。

### 明早评测口径(注意数据集名变了)

```bash
grep -c "OVERNIGHT DONE" detection/overnight.log     # 1 = 全跑完
grep "finished rc=" detection/overnight.log          # 每组的退出码和耗时
ls -d detection/runs/o_*/ | grep -v _frozen | wc -l  # 应为 9

for n in o_clean_neg o_seed1 o_seed2 o_seed3 o_lr_hi o_lr_lo o_nofreeze o_ep60; do
  echo "=== $n ==="
  python3 detection/evaluate.py detection/runs/$n/weights/best.pt \
    data/ecallisto/yolo7_single/data.yaml --imgsz 640 --conf 0.05 --class-agnostic \
    2>&1 | grep -E "^burst|detected|false alarms"
done
python3 detection/evaluate.py detection/runs/o_wide/weights/best.pt \
  data/ecallisto/yolo7_wide/data.yaml --imgsz 1280 --conf 0.05 --class-agnostic
```

`rect` 仍然必须跨组一致(evaluate.py 没显式传就是 ultralytics 默认值,九组都一样即可)。
判读顺序不变:**先用 1-4 组算 σ,差异 < 2σ 一律当没差异**。

### 脚本里写死的三个防坑点

1. 启动前检查用 `pgrep -f "train_yolo.py"` + 显存余量,**没有**用"GPU 上有没有进程"——
   这台机器常驻 `dcvagent`(远程桌面,~300MiB)和 Xorg,那种检查在这里永远为真。
   第一次启动就是这么失败的,已修。
2. 顺序执行,**没有任何轮询/等待逻辑**,从根上避开上轮那个自锁 4 小时的坑。
3. 没有 `set -e`:某一组崩了不会带走后面几组,每组退出码单独打进日志。
