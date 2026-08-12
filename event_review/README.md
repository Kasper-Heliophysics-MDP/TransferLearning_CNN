# event_review

人工审核工具:对每个burst事件,并排看raw和清洗后的频谱图,分类为"直接可用/需要case-by-case处理/弃用"。

背景见仓库根目录`开发日志.md`(eCallisto部分)——这个session里反复发现同一套清洗参数不能可靠地适配所有站点/事件,数字指标也无法自动识别哪些case出了问题,唯一稳定的信号是肉眼判断。这个工具把"肉眼判断"这一步固化成可持续、可追溯的流程,而不是每次临时用matplotlib画图看。

## 启动

```bash
pip install -r event_review/requirements.txt
streamlit run event_review/app.py
```

浏览器会自动打开(默认 http://localhost:8501)。

## 用法

侧边栏选数据源(own-station / eCallisto任意`scrape.py`输出目录)、按站点/类型/审核状态筛选、上一个/下一个翻页。主区域看raw和cleaned两张图,红线是catalog标注的burst起止时间,底色阴影是实际保护窗口(含margin)——两者之间的缝隙就是真实信号可能被误伤的地方(具体案例见开发日志的SWISS-Landschlacht部分)。

三选一分类 + 可选备注,点"保存"或"保存并下一个"立即写入该数据源目录下的`review_status.csv`(逐条原子写入,不会因为中途关闭丢失已保存的审核)。

如果某个事件用默认参数处理得不理想,展开"调整清洗参数"面板现场调(最常用的是`known_burst_margin`),预览会实时更新;分类选"可用但需case by case处理"并保存时,当前面板里的参数会一并存进`override_params_json`列,供下游生成最终训练数据时读取使用。

## 重要:own-station的"raw"是重建出来的,不是直接存在的文件

`data/burst_data/rough_events/*.npy`本身已经是去噪之后的产物(`extract_events_own_station.py`在裁剪前就跑了清洗),不是原始数据。这个工具展示的own-station raw,是实时从`data/burst_data/csv/original/`里对应的原始CSV按时间片段重新读出来的(只读需要的行范围,不会整个文件加载进内存——这些CSV单个最大1.3GB)。定位到正确原始文件这一步做了模糊匹配(处理`PeachMountian`这类拼写不一致)+ 同日多录制session的时间范围二次确认,细节见`data_access.py`里`resolve_own_station_csv`的注释。

eCallisto这边没有这个问题,`.npy`本身就是真实raw。

## 下游怎么用审核结果

`review_status.csv`(`file_name`列关联`metadata.csv`):
- `status=discard` → 训练数据准备阶段整条排除
- `status=usable` → 用默认参数的清洗结果
- `status=case_by_case`且`override_params_json`非空 → 用该JSON里的参数重新跑一遍`clean()`生成最终数组,而不是用默认参数的结果

这一步(读取`review_status.csv`、按状态分流生成最终训练数组)目前还没有实现,是这个工具之后的下一步。
