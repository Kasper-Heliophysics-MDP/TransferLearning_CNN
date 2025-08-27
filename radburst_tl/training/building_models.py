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


def build_deeplabv3_rgb_to_mono(input_shape=(256, 256, 1), num_classes=1, 
                               conversion_method='average', encoder_name='resnet34'):
    """
    Constructs a DeepLabV3+ model with RGB-to-mono weight conversion for radio burst detection.
    
    This function performs transfer learning by:
    1. Loading ImageNet pretrained RGB DeepLabV3+ model
    2. Converting RGB first layer weights to single channel
    3. Transferring all other weights to single channel model
    
    Conversion methods:
    - 'average': Simple average of RGB channels (W_mono = (W_R + W_G + W_B) / 3)
    - 'luminance': Weighted average based on human perception (0.299*R + 0.587*G + 0.114*B)
    
    Parameters:
        input_shape (tuple): Input dimensions for single channel spectrograms
        num_classes (int): Number of output classes (typically 1 for binary segmentation)
        conversion_method (str): Method for RGB to mono conversion ('average' or 'luminance')
        encoder_name (str): Backbone architecture name
    
    Returns:
        model: DeepLabV3+ model with converted weights from ImageNet RGB pretraining
    """
    import torch
    import torch.nn.functional as F
    
    print(f"🚀 Building DeepLabV3+ with RGB-to-mono conversion")
    print(f"   Conversion method: {conversion_method}")
    print(f"   Encoder: {encoder_name}")
    
    # Step 1: Build RGB model with ImageNet pretraining
    print("📥 Loading ImageNet pretrained RGB model...")
    model_rgb = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights='imagenet',     # Load ImageNet pretrained weights
        in_channels=3,                  # RGB input channels
        classes=num_classes,
        activation='sigmoid',
        encoder_depth=5,
        decoder_channels=256
    )
    
    # Step 2: Build target single channel model
    print("🎯 Creating target single channel model...")
    model_mono = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,           # No pretraining, will load converted weights
        in_channels=input_shape[-1],    # Single channel input
        classes=num_classes,
        activation='sigmoid',
        encoder_depth=5,
        decoder_channels=256
    )
    
    # Step 3: Convert RGB weights to single channel
    print(f"🔄 Converting RGB first layer weights using '{conversion_method}' method...")
    
    # Get RGB first layer weights: shape [64, 3, 7, 7]
    rgb_conv1_weight = model_rgb.encoder.conv1.weight.data
    print(f"   RGB weights shape: {rgb_conv1_weight.shape}")
    
    if conversion_method == 'average':
        # Simple average across RGB channels
        mono_conv1_weight = rgb_conv1_weight.mean(dim=1, keepdim=True)
        print("   Applied simple average conversion")
        
    elif conversion_method == 'luminance':
        # Weighted average based on luminance perception
        # Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601 standard)
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], 
                                 dtype=rgb_conv1_weight.dtype,
                                 device=rgb_conv1_weight.device)
        
        # Apply weighted combination: [64, 3, 7, 7] -> [64, 1, 7, 7]
        mono_conv1_weight = (rgb_conv1_weight * rgb_weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        print("   Applied luminance-weighted conversion")
        
    else:
        raise ValueError(f"Unsupported conversion method: {conversion_method}. Use 'average' or 'luminance'")
    
    print(f"   Converted weights shape: {mono_conv1_weight.shape}")
    
    # Step 4: Transfer weights from RGB model to mono model
    print("🔗 Transferring weights...")
    
    rgb_state_dict = model_rgb.state_dict()
    mono_state_dict = model_mono.state_dict()
    
    transferred_count = 0
    converted_count = 0
    skipped_count = 0
    
    for key in mono_state_dict.keys():
        if key == 'encoder.conv1.weight':
            # Special handling for first convolution layer
            mono_state_dict[key] = mono_conv1_weight
            converted_count += 1
            print(f"   ✅ Converted: {key}")
            
        elif key in rgb_state_dict and rgb_state_dict[key].shape == mono_state_dict[key].shape:
            # Direct transfer for matching shapes
            mono_state_dict[key] = rgb_state_dict[key]
            transferred_count += 1
            
        else:
            # Skip incompatible weights
            skipped_count += 1
            if skipped_count <= 3:  # Only print first few skipped items
                print(f"   ⚠️ Skipped: {key} (shape mismatch)")
    
    # Load converted state dict
    model_mono.load_state_dict(mono_state_dict)
    
    print(f"📊 Weight transfer summary:")
    print(f"   ✅ Converted layers: {converted_count}")
    print(f"   ✅ Transferred layers: {transferred_count}")
    print(f"   ⚠️ Skipped layers: {skipped_count}")
    
    # Step 5: Validation
    print("🔍 Validating converted model...")
    
    # Test forward pass
    test_input = torch.randn(1, input_shape[-1], *input_shape[:2])
    model_mono.eval()
    
    with torch.no_grad():
        test_output = model_mono(test_input)
    
    total_params = sum(p.numel() for p in model_mono.parameters())
    
    print(f"✅ Validation successful:")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    print(f"   Total parameters: {total_params:,}")
    
    print("🎉 RGB-to-mono conversion completed successfully!")
    print("   Model ready for training on radio burst data")
    
    return model_mono