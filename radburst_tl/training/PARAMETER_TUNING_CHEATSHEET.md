# 🎛️ Enhanced Loss Function Parameter Tuning Cheat Sheet

## 📊 核心可调参数一览表

| 参数组 | 参数名 | 默认值 | 调整范围 | 主要作用 |
|--------|--------|--------|----------|----------|
| **Loss Weights** | `focal` | 1.0 | 0.5-2.0 | 控制类别不平衡处理强度 |
|  | `iou` | 1.0 | 0.5-2.0 | 控制重叠质量重要性 |
|  | `boundary` | 0.2 | 0.1-1.0 | 控制边缘检测精度 |
| **Focal Params** | `alpha` | 0.75 | 0.25-0.95 | 正负样本重要性平衡 |
|  | `gamma` | 2.0 | 1.0-5.0 | 困难样本关注度 |

## 🎯 问题导向的调参策略

### 🚨 问题1: 漏检太多 (Low Recall, High Precision)
**症状**: 很多radio burst没被检测到，但检测到的都是对的

**调参方案**:
```python
loss_weights = {'focal': 1.5, 'iou': 1.5, 'boundary': 0.2}
focal_params = {'alpha': 0.85, 'gamma': 3.0}
```

**解释**: 
- ↑ `alpha` (0.75→0.85): 更关注正样本
- ↑ `gamma` (2.0→3.0): 更关注困难样本  
- ↑ `focal_weight` (1.0→1.5): 加强focal loss作用

---

### 🚨 问题2: 误检太多 (High Recall, Low Precision)  
**症状**: 检测到很多burst，但很多是噪声

**调参方案**:
```python
loss_weights = {'focal': 0.8, 'iou': 1.5, 'boundary': 0.4}
focal_params = {'alpha': 0.65, 'gamma': 2.0}
```

**解释**:
- ↓ `alpha` (0.75→0.65): 更关注负样本
- ↑ `boundary_weight` (0.2→0.4): 加强边缘质量
- ↑ `iou_weight` (1.0→1.5): 要求更好的重叠

---

### 🚨 问题3: 边界模糊 (Good Detection, Poor Edges)
**症状**: 能检测到burst，但边界不精确

**调参方案**:
```python
loss_weights = {'focal': 1.0, 'iou': 1.0, 'boundary': 0.6}
focal_params = {'alpha': 0.75, 'gamma': 2.0}  # 保持默认
```

**解释**:
- ↑ `boundary_weight` (0.2→0.6): 重点关注边缘
- 其他参数保持默认，专注边缘改进

---

### 🚨 问题4: 训练不稳定 (Loss Oscillation)
**症状**: Loss曲线震荡，训练不收敛

**调参方案**:
```python
loss_weights = {'focal': 0.8, 'iou': 0.8, 'boundary': 0.1}
focal_params = {'alpha': 0.6, 'gamma': 1.5}
```

**解释**:
- ↓ `gamma` (2.0→1.5): 减少激进关注
- ↓ 所有权重: 降低loss强度
- `alpha`趋向0.5: 更平衡的类别处理

---

### 🚨 问题5: 极度不平衡 (Severe Class Imbalance)
**症状**: 模型几乎只预测背景，radio burst太稀少

**调参方案**:
```python
loss_weights = {'focal': 2.0, 'iou': 1.5, 'boundary': 0.1}
focal_params = {'alpha': 0.9, 'gamma': 4.0}
```

**解释**:
- ↑↑ `alpha` (0.75→0.9): 极度关注正样本
- ↑↑ `gamma` (2.0→4.0): 极度关注困难样本
- ↑↑ `focal_weight` (1.0→2.0): focal loss主导

## 🔧 系统化调参流程

### Step 1: 问题诊断
```python
from loss_tuner import quick_tune

# 输入你的验证指标
val_metrics = {
    'precision': 0.85,  # 你的precision
    'recall': 0.45,     # 你的recall  
    'f1': 0.59,         # 你的f1
    'iou': 0.42         # 你的iou
}

# 获得建议参数
suggested = quick_tune(val_metrics)
```

### Step 2: 应用建议参数
```python
# 在训练循环中使用
loss = combined_loss(
    y_true, y_pred,
    loss_weights=suggested['loss_weights'],
    focal_params=suggested['focal_params']
)
```

### Step 3: 迭代优化
- 训练10-20个epoch
- 观察验证指标变化趋势
- 如果改善，继续同方向调整
- 如果没改善，尝试其他策略

## 🎯 快速参考表

| 你的问题 | 主要调整 | 次要调整 |
|----------|----------|----------|
| **漏检严重** | `alpha↑`, `gamma↑` | `focal_weight↑` |
| **误检严重** | `alpha↓`, `boundary_weight↑` | `iou_weight↑` |
| **边界模糊** | `boundary_weight↑` | 保持其他默认 |
| **训练不稳** | `gamma↓`, 所有权重↓ | `alpha`趋向0.5 |
| **极度不平衡** | `alpha↑↑`, `gamma↑↑` | `focal_weight↑↑` |

## 💡 实用技巧

### 🔥 针对Solar Radio Burst的最佳起点:
```python
# 推荐的起始配置
loss_weights = {'focal': 1.2, 'iou': 1.5, 'boundary': 0.2}
focal_params = {'alpha': 0.8, 'gamma': 2.5}
```

### ⚡ 快速验证方法:
1. 用建议参数训练5-10 epochs
2. 如果val_loss下降趋势良好→继续
3. 如果val_loss震荡或上升→尝试其他配置

### 📊 监控重点:
- **Focal loss值**: 应该随训练下降
- **Boundary loss值**: 边界任务的改善指标
- **Validation F1**: 主要优化目标
- **Validation IoU**: 定位质量指标

### 🎛️ 微调技巧:
- **小步调整**: 每次改变±10-20%
- **单变量法**: 一次只调一个参数组
- **对比验证**: 保留原loss做baseline对比
- **耐心等待**: 给每个配置足够的训练时间

### ⚠️ 注意事项:
- 不要在validation set上过度调参
- 保留test set做最终验证
- 记录每次实验的参数和结果
- 考虑数据质量对参数选择的影响

## 🚀 立即开始

最简单的开始方式：

```python
# 1. 诊断当前问题
from loss_tuner import LossTuner
tuner = LossTuner()
problem, strategy = tuner.diagnose_problem(your_val_metrics)

# 2. 获得建议参数  
suggested = tuner.suggest_parameters(your_val_metrics)

# 3. 应用到训练
loss = combined_loss(y_true, y_pred, **suggested)

# 4. 训练并观察改善
```

记住：**调参是迭代过程，从建议配置开始，根据结果逐步优化！**
