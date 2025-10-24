# SpecGAN Architecture Guide

## Generator Architecture

### **Overview**
- **Input:** 100-dimensional noise vector `z ~ Uniform[-1, 1]`
- **Output:** 128×128 single-channel spectrogram (range: [-1, 1])
- **Layers:** 5 transpose convolutions with progressive upsampling
- **Kernels:** 5×5 (larger receptive field than DCGAN's 4×4)

### **Layer Structure**

```
Input: [N, 100]
  ↓
Dense + Reshape: [N, 1024, 4, 4]
  ↓
Conv2DTranspose (5×5, stride=2): [N, 512, 8, 8]
  ↓
Conv2DTranspose (5×5, stride=2): [N, 256, 16, 16]
  ↓
Conv2DTranspose (5×5, stride=2): [N, 128, 32, 32]
  ↓
Conv2DTranspose (5×5, stride=2): [N, 64, 64, 64]
  ↓
Conv2DTranspose (5×5, stride=2): [N, 1, 128, 128]
  ↓
Tanh activation: Range [-1, 1]
  ↓
Output: [N, 1, 128, 128]
```

### **Key Design Choices**

**Activation Functions:**
- Hidden layers: ReLU
- Output layer: Tanh (ensures [-1, 1] range)

**BatchNorm:**
- Default: False (SpecGAN convention)
- Optional: Can be enabled via `use_batchnorm=True`

**Dimension Progression:**
- 1024 → 512 → 256 → 128 → 64 → 1 channels
- Each layer halves the channel count, doubles spatial size

---

## Discriminator Architecture

### **Overview**
- **Input:** 128×128 single-channel spectrogram
- **Output:** Single logit value (real/fake score, NOT probability)
- **Layers:** 5 convolutions with progressive downsampling
- **Kernels:** 5×5 (matches Generator)

### **Layer Structure**

```
Input: [N, 1, 128, 128]
  ↓
Conv2D (5×5, stride=2): [N, 64, 64, 64]
  ↓
Conv2D (5×5, stride=2): [N, 128, 32, 32]
  ↓
Conv2D (5×5, stride=2): [N, 256, 16, 16]
  ↓
Conv2D (5×5, stride=2): [N, 512, 8, 8]
  ↓
Conv2D (5×5, stride=2): [N, 1024, 4, 4]
  ↓
Flatten: [N, 16384]
  ↓
Dense: [N, 1]
  ↓
Output: [N] (logits, no sigmoid)
```

### **Key Design Choices**

**Activation Function:**
- All layers: LeakyReLU(0.2)

**BatchNorm:**
- First layer: NO BatchNorm (SpecGAN convention)
- Other layers: Optional (default: False)

**Output Type:**
- Returns logits, NOT probabilities
- Sigmoid applied in loss function (e.g., BCEWithLogitsLoss)
- For WGAN-GP, raw scores used directly

**Dimension Progression:**
- 1 → 64 → 128 → 256 → 512 → 1024 channels
- Each layer doubles channel count, halves spatial size

---

## Architecture Principles

### **Why 5×5 Kernels?**
- Larger receptive field captures more context
- Better for spectrograms with extended temporal/frequency patterns
- Original SpecGAN uses 5×5 for audio spectrograms

### **Why Single Channel?**
- Spectrograms are naturally grayscale (intensity values)
- RGB channels (3) are artificial for spectrogram data
- Reduces parameters and potential artifacts

### **Why No BatchNorm on D's First Layer?**
- SpecGAN convention (matches original implementation)
- Prevents discriminator from being overconfident early
- Common practice in GAN architectures

### **Why Logits Instead of Probabilities?**
- Modern GAN losses expect logits (numerically more stable)
- WGAN-GP uses raw scores (no sigmoid)
- PyTorch's `BCEWithLogitsLoss` combines sigmoid + BCE internally

---

## Comparison with DCGAN

### **SpecGAN vs DCGAN Architectures**

| Aspect | DCGAN | SpecGAN | Advantage |
|--------|-------|---------|-----------|
| **Kernel Size** | 4×4 | 5×5 | Larger receptive field |
| **Channels** | 3 (RGB) | 1 (Grayscale) | Matches data nature |
| **Layers (128×128)** | 6 layers | 5 layers | More efficient |
| **BatchNorm Default** | Enabled | Disabled | Simpler, faster |
| **Normalization** | Global | **Per-frequency** | **Key advantage** |
| **Augmentation** | None | **Temporal shift** | **Breaks spatial bias** |
| **Loss** | BCE | **WGAN-GP** | **More stable** |
| **D:G Ratio** | 1:1 | **5:1** | **Better balance** |

**Most Critical Difference:** Per-frequency normalization (implemented in data pipeline, not model architecture)

---

## Model Parameters

### **Default Configuration (from SpecGAN paper)**

```python
# Architecture
nz = 100              # Latent vector dimension
nc = 1                # Number of channels (single-channel)
kernel_len = 5        # Kernel size (5×5)
dim = 64              # Dimension multiplier
use_batchnorm = False # BatchNorm disabled

# Training
disc_nupdates = 5     # D updates per G update
loss_type = 'wgan-gp' # Loss function
lr = 1e-4             # Learning rate (both G and D)
beta1 = 0.5           # Adam beta1
beta2 = 0.9           # Adam beta2
```

### **Parameter Count**

- **Generator:** ~11M parameters
- **Discriminator:** ~11M parameters
- **Total:** ~22M parameters

(Fewer than 3-channel DCGAN due to nc=1)

---

## Usage Examples

### **Basic Usage:**

```python
from specgan.specgan_models import SpecGANGenerator, SpecGANDiscriminator

# Create models with defaults
netG = SpecGANGenerator()
netD = SpecGANDiscriminator()

# Generate spectrograms
z = torch.randn(16, 100)
fake = netG(z)  # [16, 1, 128, 128]
```

### **Custom Configuration:**

```python
# Smaller model (faster training)
netG = SpecGANGenerator(dim=32)  # Fewer channels

# Larger model (potentially better quality)
netG = SpecGANGenerator(dim=128)  # More channels

# Enable BatchNorm
netG = SpecGANGenerator(use_batchnorm=True)
netD = SpecGANDiscriminator(use_batchnorm=True)
```

### **Load Trained Model:**

```python
from specgan.specgan_utils import load_generator_only

netG = SpecGANGenerator()
load_generator_only('checkpoints_specgan/best_model.pth', netG, device='cuda')

# Generate new samples
with torch.no_grad():
    z = torch.randn(100, 100, device='cuda')
    samples = netG(z)  # [100, 1, 128, 128]
```

---

## Training Strategy

### **SpecGAN's Approach (from train_specgan.py)**

**Discriminator Update (5 times):**
```python
for d_iter in range(5):  # Train D multiple times
    D_real = netD(real_data)
    D_fake = netD(fake_data)
    G_loss, D_loss = GANLoss.wgan_gp_loss(...)
    D_loss.backward()
    optimizerD.step()
```

**Generator Update (once):**
```python
# Train G once after D is updated
noise = torch.randn(batch_size, nz)
fake = netG(noise)
D_fake = netD(fake)
G_loss = -torch.mean(D_fake)  # WGAN loss
G_loss.backward()
optimizerG.step()
```

**Why 5:1 ratio?**
- Gives D time to provide meaningful gradients
- Prevents G from overwhelming D too quickly
- Empirically proven effective for spectrogram generation

---

## Expected Training Behavior

### **Healthy Training (WGAN-GP):**

```
Epoch 0:  D_loss: ~50-100,  G_loss: ~-50,    D(real): ~5,   D(fake): ~-5
Epoch 10: D_loss: ~10-20,   G_loss: ~-10,    D(real): ~2,   D(fake): ~-2
Epoch 50: D_loss: ~5-15,    G_loss: ~-5,     D(real): ~1,   D(fake): ~-1
Epoch 100: D_loss: ~2-10,   G_loss: ~-2,     D(real): ~0.5, D(fake): ~-0.5
```

**Target:** Both D(real) and D(fake) converge toward 0 (Nash equilibrium)

### **Warning Signs:**

- D(real) and D(fake) both stuck at large values → Increase learning rate
- Loss curves oscillating wildly → Reduce learning rate
- All generated samples identical → Mode collapse (reduce disc_nupdates)

---

## Implementation Source

**Ported from:** Chris Donahue's SpecGAN  
**Repository:** https://github.com/chrisdonahue/wavegan  
**Paper:** "Adversarial Audio Synthesis" (Donahue et al., ICLR 2019)

**Files mapped:**
- `specgan.py` → `specgan_models.py`
- `train_specgan.py` (moments, losses, training) → `specgan_utils.py`
- `train_specgan.py` (training loop) → `specgan_training.ipynb`

All line numbers referenced in code comments for verification.

