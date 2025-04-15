import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2  # Optional: for visualization
from torch.utils.data import DataLoader, TensorDataset

def reconstruct_mask(tiles, positions, original_shape, tile_size=256):
    """
    Reconstructs the full mask from predicted tiles.
    
    This function creates an empty full mask and fills in the predictions at the appropriate positions.
    For overlapping regions, it averages the predictions.
    
    Parameters:
        tiles (numpy array): Predicted binary masks for each tile (shape: [N, 1, tile_size, tile_size]).
        positions (list): A list of (row, col) positions for the top-left corner of each tile.
        original_shape (tuple): The original shape of the spectrogram (H, W).
        tile_size (int): Size of each tile.
        
    Returns:
        numpy array: The reconstructed full binary mask.
    """
    H, W = original_shape
    full_mask = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    
    for tile, (i, j) in zip(tiles, positions):
        # Remove channel dimension (tile shape: (1, tile_size, tile_size))
        tile = tile[0]
        full_mask[i:i+tile_size, j:j+tile_size] += tile
        count[i:i+tile_size, j:j+tile_size] += 1
        
    # For overlapping tiles, average the predictions
    count[count == 0] = 1
    full_mask = full_mask / count
    # Convert to binary mask using 0.5 threshold
    full_mask = (full_mask > 0.5).astype(np.uint8)
    return full_mask