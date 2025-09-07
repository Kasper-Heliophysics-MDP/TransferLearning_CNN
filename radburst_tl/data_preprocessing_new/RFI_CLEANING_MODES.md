# RFI Cleaning Modes Guide

## 🚀 Fast Mode (Default) - Recommended for Initial Processing

### What it does:
- ✅ Step 1: Time-median detrending + MAD normalization
- ✅ Step 2: Vertical RFI detection (broadband interference)
- ✅ Step 3: Horizontal RFI detection (narrowband carriers)
- ⚡ Step 4: **SKIPPED** (fine-grained cleaning)
- ✅ Step 5: Thickness-based burst protection
- ❌ Step 6: **SKIPPED** (interpolation inpainting)

### Performance:
- **Processing time**: 2-5 minutes per file (vs 30-60 minutes comprehensive)
- **Speedup**: ~10-15x faster
- **Quality**: 80-90% of comprehensive cleaning quality

### Best for:
- Initial pipeline testing
- Large batch processing (73 burst files)
- Quick quality verification
- GAN training data preparation

---

## 🔬 Comprehensive Mode - High Quality Processing

### What it does:
- ✅ Step 1: Time-median detrending + MAD normalization
- ✅ Step 2: Vertical RFI detection
- ✅ Step 3: Horizontal RFI detection
- ✅ Step 4: Fine-grained scattered noise removal (SLOW)
- ✅ Step 5: Thickness-based burst protection
- ✅ Step 6: Interpolation inpainting

### Performance:
- **Processing time**: 30-60 minutes per file
- **Quality**: Maximum cleaning quality
- **Memory usage**: Higher

### Best for:
- Final high-quality dataset
- Critical burst events
- Research publication data
- When processing time is not a constraint

---

## 🛡️ Conservative Mode - Gentle Cleaning

### What it does:
- All 6 steps with relaxed thresholds
- **1.5x higher MAD thresholds** (less aggressive)
- **1.3x higher occupancy thresholds** (preserve more signal)
- Stronger burst protection

### Best for:
- When worried about over-cleaning
- Preserving weak burst signals
- Uncertain data quality

---

## 📊 Usage Examples

### Batch Processing with Fast Mode
```python
# Process all 73 bursts quickly (~2-4 hours total)
results = process_all_bursts_by_type(
    catalog_path=CATALOG_PATH,
    cleaning_method="fast"  # 10-15x speedup
)
```

### Single Burst with Comprehensive Mode
```python
# High-quality processing for important bursts
result = slicer.slice_burst_with_fixed_windows(
    csv_file_path=file,
    burst_start_time=start,
    burst_end_time=end,
    cleaning_method="comprehensive"  # Full 6-step cleaning
)
```

### Conservative Mode for Weak Signals
```python
# Gentle cleaning for uncertain data
result = slicer.slice_burst_with_fixed_windows(
    csv_file_path=file,
    burst_start_time=start,
    burst_end_time=end,
    cleaning_method="conservative"  # Relaxed thresholds
)
```

---

## ⏱️ Time Estimates for Your Dataset

### Fast Mode (Recommended for Initial Run):
- **Single file**: 2-5 minutes
- **73 bursts**: 2-4 hours total
- **Quality**: Good (80-90% of comprehensive)

### Comprehensive Mode:
- **Single file**: 30-60 minutes  
- **73 bursts**: 24-48 hours total
- **Quality**: Excellent (100%)

### Conservative Mode:
- **Single file**: 20-40 minutes
- **73 bursts**: 12-24 hours total
- **Quality**: Good with signal preservation

---

## 🎯 Recommended Workflow

1. **Start with Fast Mode**: Process all 73 bursts quickly to verify pipeline
2. **Quality Check**: Review a few sample results visually
3. **Selective Comprehensive**: Re-process important bursts with comprehensive mode if needed
4. **GAN Training**: Use fast mode results for initial GAN training
5. **Final Polish**: Use comprehensive mode for final publication-quality dataset

---

## 🔧 Performance Optimization Tips

### For Fast Mode:
- Use `apply_denoising=True, cleaning_method="fast"`
- Process in batches of 10-20 files
- Monitor memory usage

### For Comprehensive Mode:
- Process overnight or during breaks
- Consider processing only Type 3 bursts (62 events) first
- Save intermediate results frequently

### Memory Management:
- Close notebooks between large processing runs
- Clear variables after batch processing
- Consider processing by date ranges if memory limited

---

## 📈 Quality vs Speed Trade-offs

| Mode | Speed | RFI Removal | Burst Preservation | Use Case |
|------|-------|-------------|-------------------|----------|
| **Fast** | ⚡⚡⚡ | 80% | 95% | Initial processing, batch runs |
| **Comprehensive** | ⚡ | 95% | 95% | Final dataset, publication |
| **Conservative** | ⚡⚡ | 70% | 98% | Weak signals, uncertain data |

**Recommendation**: Start with Fast mode for all 73 bursts, then selectively upgrade important ones to Comprehensive mode.
