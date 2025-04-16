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

# 归一化图像块 - 与训练过程保持一致
def normalize_tiles(tiles):
    """
    根据训练过程中的归一化方法处理图像块
    
    Parameters:
        tiles: 图像块列表或numpy数组
        
    Returns:
        归一化后的numpy数组，格式适用于模型输入
    """
    # 转换为numpy数组（如果还不是）
    if not isinstance(tiles, np.ndarray):
        tiles = np.array(tiles)
    
    # 确保数据类型为float32
    tiles = tiles.astype(np.float32)
    
    # 打印原始形状以便调试
    print(f"原始tiles形状: {tiles.shape}")
    
    # 按照训练过程中相同的方式归一化
    max_val = np.max(tiles)
    tiles_normalized = tiles / max_val if max_val != 0 else tiles
    
    # 打印最终形状以便调试
    print(f"归一化后tiles形状: {tiles_normalized.shape}")
    
    return tiles_normalized

