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