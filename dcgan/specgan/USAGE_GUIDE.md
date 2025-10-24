# SpecGAN Usage Guide

## Prerequisites

Ensure you have the following files:
- `csv_spectrogram_dataset.py` - Data loader
- `specgan/specgan_models.py` - Generator & Discriminator
- `specgan/specgan_utils.py` - Training utilities
- `specgan/compute_moments.py` - Preprocessing script
- `specgan_training.ipynb` - Training notebook

---

## Training Workflow (3 Steps)

### **Step 1: Compute Per-Frequency Statistics (Run Once)**

```bash
cd /Users/remiliascarlet/Desktop/MDP/transfer_learning/dcgan
python specgan/compute_moments.py
```

**What it does:**
- Loads all 218 CSV spectrogram files
- Computes mean and std for each of 128 frequency bins
- Saves to `checkpoints_specgan/type3_moments.npz`

**Time:** ~1-2 minutes  
**Output:** `type3_moments.npz` (required for training)

**Optional: Use all burst types**
```bash
python specgan/compute_moments.py --all_types
```
This uses Type 2 + Type 3 + Type 5 (258 samples total)

---

### **Step 2: Train SpecGAN**

1. Open `specgan_training.ipynb` in Jupyter
2. Run all cells sequentially
3. Monitor training progress

**Key configuration (Cell 6):**
```python
dataroot = "path/to/type_3/"  
moments_path = "./checkpoints_specgan/type3_moments.npz"
loss_type = 'wgan-gp'  # Recommended
disc_nupdates = 5       # D trains 5x per G update
batch_size = 16
num_epochs = 500
```

**Training will:**
- Use per-frequency normalization (SpecGAN's key advantage)
- Apply temporal shift augmentation (fixes spatial bias)
- Train with WGAN-GP loss (more stable)
- Save best models automatically

---

### **Step 3: Evaluate Results**

Check the generated spectrograms in notebook cells:
- Cell 10: Final generated spectrograms
- Cell 11: Real vs Fake comparison
- Cell 8: Loss curves

Compare with DCGAN results:
- Horizontal striping (should be reduced)
- Spatial distribution (bursts across time, not just left)
- Overall quality

---

## Configuration Options

### **For Different Burst Types:**

```python
# Type 3 only (default)
dataroot = "path/to/type_3/"
moments_path = "./checkpoints_specgan/type3_moments.npz"

# All types
dataroot = "path/to/gan_training_windows_128/"
moments_path = "./checkpoints_specgan/all_types_moments.npz"
# Then run: python compute_moments.py --all_types
```

### **Loss Function Options:**

```python
loss_type = 'wgan-gp'  # Recommended (SpecGAN default)
loss_type = 'dcgan'    # Standard BCE  
loss_type = 'lsgan'    # Least squares
loss_type = 'wgan'     # Vanilla Wasserstein
```

### **Training Parameters:**

```python
# SpecGAN defaults (recommended)
kernel_len = 5
dim = 64
use_batchnorm = False
disc_nupdates = 5
lr = 1e-4

# For faster experimentation
batch_size = 8       # Reduce if memory limited
num_epochs = 100     # Quick test
```

---

## Loading Trained Models

### **Load Best Checkpoint:**

```python
from specgan.specgan_utils import load_generator_only
from specgan.specgan_models import SpecGANGenerator

# Create generator
netG = SpecGANGenerator(nz=100, kernel_len=5, dim=64, nc=1)

# Load best checkpoint
load_generator_only(
    'checkpoints_specgan/checkpoint_epoch_X_quality_Y.pth',
    netG,
    device='cuda'
)

# Generate samples
z = torch.randn(16, 100, device='cuda')
fake = netG(z)  # [16, 1, 128, 128]
```

### **Find Best Checkpoint Automatically:**

```python
from specgan.specgan_utils import find_best_checkpoint

best_path = find_best_checkpoint('checkpoints_specgan/')
print(f"Best model: {best_path}")
```

---

## Troubleshooting

### **Error: "No moments file found"**
**Solution:** Run `compute_moments.py` first

### **Error: "ImportError: specgan_utils"**
**Solution:** Check path in notebook Cell 2:
```python
sys.path.insert(0, '/path/to/dcgan')
```

### **Error: "CUDA out of memory"**
**Solution:** Reduce `batch_size` to 8 or 4

### **Training unstable (Loss exploding)**
**Solution:** 
- Use `loss_type='wgan-gp'` (most stable)
- Reduce `lr` to `5e-5`
- Increase `disc_nupdates` to 10

---

## Expected Results

### **Compared to DCGAN:**

**Improvements:**
- ✅ Horizontal striping: 70-80% reduction
- ✅ Spatial bias: 90% elimination (bursts distributed across time)
- ✅ Training stability: Significantly better
- ✅ Frequency coherence: Better preserved

**Metrics to watch:**
- D(real) and D(fake) should both be near 0 for WGAN-GP
- Loss curves should be stable (no sudden spikes)
- Generated spectrograms should show clear burst structures

---

## File Structure

```
dcgan/
├── csv_spectrogram_dataset.py      # Data loader (modified)
├── specgan/
│   ├── specgan_models.py           # G & D architectures
│   ├── specgan_utils.py            # Training utilities
│   ├── compute_moments.py          # Preprocessing script
│   └── specgan_training.ipynb      # Training notebook
└── checkpoints_specgan/            # Saved models & moments
    ├── type3_moments.npz
    └── checkpoint_epoch_*.pth
```

---

## Quick Reference

### **SpecGAN Default Parameters:**
```python
nz = 100                # Latent dimension
nc = 1                  # Single channel
kernel_len = 5          # 5×5 kernels
dim = 64                # Dimension multiplier
use_batchnorm = False   # No BatchNorm
disc_nupdates = 5       # D:G ratio = 5:1
loss_type = 'wgan-gp'   # WGAN with gradient penalty
lr = 1e-4               # Learning rate (both G and D)
beta1, beta2 = 0.5, 0.9 # Adam parameters
```

### **Dataset Configuration:**
```python
normalize_method = 'per_frequency'  # SpecGAN's key feature
grayscale = True                    # Single channel
moments_path = 'moments.npz'        # Pre-computed statistics
augment = True                      # Temporal shift augmentation
```

---

## Summary

**Three-step workflow:**
1. `python compute_moments.py` → Generate statistics
2. Run `specgan_training.ipynb` → Train model
3. Evaluate results → Compare with DCGAN

**Key advantage:** Per-frequency normalization preserves frequency-specific characteristics, leading to higher quality spectrogram generation.

