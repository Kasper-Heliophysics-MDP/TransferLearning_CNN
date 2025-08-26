# 🎯 Enhanced Loss Function Integration Guide

## ✅ What's Been Implemented

The enhanced loss functions have been successfully added to `train_utils.py`:

### 🧩 New Loss Components:

1. **Focal Loss** - Addresses class imbalance (radio bursts are rare)
2. **Boundary Loss** - Improves edge detection accuracy  
3. **Adaptive IoU Loss** - Better overlap quality with power weighting
4. **Combined Loss** - Integrates all components with configurable weights

### 📁 New Files Created:

- `train_utils.py` - Enhanced with new loss functions
- `loss_config_example.py` - Configuration examples and scenarios
- `ENHANCED_LOSS_INTEGRATION.md` - This integration guide

## 🔄 How to Integrate

### Step 1: Update Your Training Script

Replace the loss computation in your training loop:

```python
# OLD (in train_one_epoch function):
loss = combined_loss(y, preds)

# NEW:
from loss_config_example import get_loss_config_for_scenario

config = get_loss_config_for_scenario("imbalanced")  # Recommended for radio bursts
loss = combined_loss(
    y, preds,
    loss_weights=config['loss_weights'],
    focal_params=config['focal_params']
)
```

### Step 2: Choose Your Configuration

Based on your data characteristics:

```python
# For typical radio burst detection (sparse signals)
config = get_loss_config_for_scenario("imbalanced")

# For high-precision boundary detection
config = get_loss_config_for_scenario("boundary_critical")

# For noisy data
config = get_loss_config_for_scenario("noisy")

# For backward compatibility (original loss)
config = get_loss_config_for_scenario("original")
```

### Step 3: Monitor Loss Components

Track individual components during training:

```python
# In your validation loop:
with torch.no_grad():
    focal_val = focal_loss(preds, y_true, alpha=0.75, gamma=2.0)
    iou_val = adaptive_iou_loss(preds, y_true, power=1.5)
    boundary_val = boundary_loss(preds, y_true)
    
    print(f"Focal: {focal_val:.4f}, IoU: {iou_val:.4f}, Boundary: {boundary_val:.4f}")
```

## 🎯 Recommended Settings for Radio Burst Detection

### For Typical Solar Radio Burst Data:

```python
loss_weights = {'focal': 1.2, 'iou': 1.5, 'boundary': 0.2}
focal_params = {'alpha': 0.8, 'gamma': 3.0}
```

**Rationale:**
- High focal weight (1.2) + gamma (3.0) to handle sparse radio bursts
- High IoU weight (1.5) for accurate signal localization
- Lower boundary weight (0.2) since radio bursts can have fuzzy edges

### For High-Precision Applications:

```python
loss_weights = {'focal': 0.6, 'iou': 1.0, 'boundary': 0.8}
focal_params = {'alpha': 0.75, 'gamma': 2.0}
```

**Rationale:**
- High boundary weight (0.8) for precise edge detection
- Moderate focal loss for balanced training

## 🚀 Quick Start

### Minimal Changes to Existing Code:

1. In `train_utils.py` - ✅ Already done!

2. In your training script, replace:
```python
# train_one_epoch function, line ~206:
loss = combined_loss(y, preds)
```

With:
```python
# Enhanced loss with imbalanced data configuration
loss = combined_loss(
    y, preds,
    loss_weights={'focal': 1.2, 'iou': 1.5, 'boundary': 0.2},
    focal_params={'alpha': 0.8, 'gamma': 3.0}
)
```

### For A/B Testing:

Keep both versions and compare:

```python
# Original loss
original_loss = simple_combined_loss(y, preds)

# Enhanced loss  
enhanced_loss = combined_loss(y, preds, ...)

# Use enhanced_loss for training, original_loss for comparison
loss = enhanced_loss
```

## 📊 Expected Improvements

### What You Should See:

1. **Better handling of sparse radio bursts** - Focal loss focuses on hard examples
2. **Improved edge quality** - Boundary loss enhances burst boundaries
3. **More stable training** - Adaptive IoU loss handles class imbalance better
4. **Configurable emphasis** - Tune weights based on your specific needs

### Validation Metrics to Watch:

- **IoU/F1 scores** should improve, especially for small bursts
- **Edge quality** should be visually better
- **False positive rate** should decrease (focal loss effect)
- **Training stability** should improve

## 🔧 Troubleshooting

### If Loss is Too High:
- Reduce `gamma` in focal_params (try 1.5-2.0)
- Lower `boundary` weight (try 0.1-0.2)

### If Missing Small Bursts:
- Increase `gamma` in focal_params (try 3.0-4.0)
- Increase `alpha` in focal_params (try 0.85-0.9)

### If Edges are Poor:
- Increase `boundary` weight (try 0.5-0.8)
- Check that your ground truth has clean edges

### For Backward Compatibility:
```python
# Use original loss anytime
loss = simple_combined_loss(y, preds)
```

## 🎯 Next Steps

1. **Start with "imbalanced" configuration** - Best for typical radio burst data
2. **Monitor individual loss components** - Understand what's happening
3. **Experiment with weights** - Fine-tune based on validation performance
4. **Compare with original** - Use simple_combined_loss for baseline comparison
5. **Adjust based on results** - Each dataset may need slight tuning

The enhanced loss functions are designed to be drop-in replacements that should improve performance on solar radio burst detection tasks, especially with imbalanced and noisy data!
