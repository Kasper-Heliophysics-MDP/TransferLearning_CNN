import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
import segmentation_models_pytorch as smp

def build_unet(input_shape=(256, 256, 1), num_classes=1, encoder_weights='imagenet'):
    """
    Constructs a UNet model with an option to load pre-trained encoder weights for transfer learning.

    Parameters:
        input_shape (tuple): Specifies the dimensions of the input images.
                             For example, (256, 256, 1) represents 256x256 grayscale images.
        num_classes (int): The number of output classes (typically 1 for binary segmentation or 2 for multi-class segmentation).
        encoder_weights (str or None): If set to 'imagenet', pre-trained weights for the encoder will be loaded.
                                       If None, the encoder's weights will be randomly initialized.

    Returns:
        model: A UNet model instance constructed using the segmentation_models library.
    """
    # Use the Unet model from the segmentation_models library as an example.
    model = smp.Unet(
        encoder_name='resnet34',  # select backbone architecture
        encoder_weights=encoder_weights,  # 'imagenet' to load pre-trained weights
        in_channels=input_shape[-1],  # number of input channels
        classes=num_classes,          # number of output classes
        activation='sigmoid'          # final activation function
    )
    return model

def build_deeplabv3(input_shape=(256, 256, 1), num_classes=1, encoder_weights=None, encoder_name='resnet34'):
    """
    Constructs a DeepLabV3+ model optimized for radio burst detection.
    
    DeepLabV3+ advantages for radio burst detection:
    - Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale temporal patterns
    - Better handling of objects at different scales (short vs long bursts)
    - Improved boundary detection compared to UNet
    - More robust feature extraction for sparse signals
    
    Parameters:
        input_shape (tuple): Specifies the dimensions of the input images.
                             For example, (256, 256, 1) represents 256x256 grayscale spectrograms.
        num_classes (int): The number of output classes (typically 1 for binary segmentation).
        encoder_weights (str or None): If None, encoder weights will be randomly initialized.
                                       Recommended to use None for spectrogram data.
        encoder_name (str): Backbone architecture. Options: 'resnet34', 'resnet18', 'efficientnet-b0'
    
    Returns:
        model: A DeepLabV3+ model instance optimized for radio burst segmentation.
    """
    print(f"🚀 Building DeepLabV3+ with {encoder_name} backbone")
    print(f"   Input: {input_shape}, Classes: {num_classes}")
    print(f"   Encoder weights: {'None (from scratch)' if encoder_weights is None else encoder_weights}")
    
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,           # backbone architecture
        encoder_weights=encoder_weights,      # None for spectrogram-specific training
        in_channels=input_shape[-1],         # number of input channels (1 for grayscale spectrogram)
        classes=num_classes,                 # number of output classes
        activation='sigmoid',                # final activation function for binary segmentation
        encoder_depth=5,                     # depth of feature extraction
        decoder_channels=256,                # decoder feature channels
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model

def build_lightweight_deeplabv3(input_shape=(256, 256, 1), num_classes=1):
    """
    Constructs a lightweight DeepLabV3+ model for faster training and inference.
    
    Parameters:
        input_shape (tuple): Input dimensions
        num_classes (int): Number of output classes
    
    Returns:
        model: Lightweight DeepLabV3+ model
    """
    print("🚀 Building Lightweight DeepLabV3+ with EfficientNet-B0 backbone")
    
    model = smp.DeepLabV3Plus(
        encoder_name='efficientnet-b0',      # lightweight backbone
        encoder_weights=None,                # train from scratch
        in_channels=input_shape[-1],
        classes=num_classes,
        activation='sigmoid',
        decoder_channels=128,                # reduced decoder channels for speed
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Lightweight model created! Parameters: {total_params:,}")
    
    return model

def build_comparison_models(input_shape=(256, 256, 1), num_classes=1):
    """
    Build multiple models for comparison testing.
    
    Returns:
        dict: Dictionary containing different model architectures
    """
    models = {}
    
    print("🏗️ Building model comparison suite...")
    
    # 1. Original UNet with ImageNet (baseline)
    print("\n📍 1. UNet + ImageNet (Original)")
    models['unet_imagenet'] = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=input_shape[-1],
        classes=num_classes,
        activation='sigmoid'
    )
    
    # 2. UNet without ImageNet
    print("📍 2. UNet from scratch")
    models['unet_scratch'] = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=input_shape[-1],
        classes=num_classes,
        activation='sigmoid'
    )
    
    # 3. DeepLabV3+ from scratch
    print("📍 3. DeepLabV3+ from scratch")
    models['deeplabv3_scratch'] = build_deeplabv3(input_shape, num_classes, encoder_weights=None)
    
    # 4. Lightweight DeepLabV3+
    print("📍 4. Lightweight DeepLabV3+")
    models['deeplabv3_light'] = build_lightweight_deeplabv3(input_shape, num_classes)
    
    # Print comparison summary
    print("\n📊 Model Comparison Summary:")
    print("-" * 70)
    print(f"{'Model':<25} {'Parameters':<15} {'Backbone':<15} {'Pretrained':<10}")
    print("-" * 70)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        backbone = model.encoder.name if hasattr(model.encoder, 'name') else 'N/A'
        pretrained = 'Yes' if 'imagenet' in name else 'No'
        print(f"{name:<25} {params:<15,} {backbone:<15} {pretrained:<10}")
    
    return models