import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

class SpectrogramSlicer:
    def __init__(self, target_size=(256, 256), overlap_ratio=0.25, random_offset=True):
        """
        Initialize the spectrogram slicer
        
        Args:
            target_size: Target slice size, default (256, 256)
            overlap_ratio: Overlap ratio, default 0.25 (25%)
            random_offset: Whether to use random starting points, default True
        """
        self.target_height, self.target_width = target_size
        self.overlap_ratio = overlap_ratio
        self.random_offset = random_offset
        
    def slice_spectrogram_time_range(self, spectrogram, time_array, start_time_str, end_time_str, 
                                     mask=None, time_padding=0.2, is_training=True):
        """
        Slice the spectrogram within a specified time range
        
        Args:
            spectrogram: Input spectrogram, shape (height, width) or (height, width, channels)
            time_array: Array of timestamps corresponding to the columns of the spectrogram
            start_time_str: Start time in 'HH:MM:SS' format
            end_time_str: End time in 'HH:MM:SS' format
            mask: Corresponding mask, if any, shape should be (height, width)
            time_padding: Additional padding around the time range as a fraction of range length
            is_training: Whether in training mode, training mode will use random starting points
            
        Returns:
            slices: List of spectrogram slices
            mask_slices: List of mask slices (if mask is provided)
            positions: Position of each slice in the original image [(y, x), ...]
        """
        height, width = spectrogram.shape[:2]
        
        # Convert time strings to column indices
        from datetime import datetime
        
        # Convert time strings to datetime objects
        def time_to_datetime(time_str):
            # Try multiple possible time formats
            formats = ["%H:%M:%S", "%H:%M:%S.%f", "%M:%S.%f", "%M:%S"]
            
            for fmt in formats:
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, raise an error with more specific information
            raise ValueError(f"Unsupported time format: '{time_str}'。supported formats: HH:MM:SS, HH:MM:SS.sss")
        
        # Helper function to find closest index in time_array
        def find_closest_time_index(target_time):
            target_dt = time_to_datetime(target_time)
            if hasattr(time_array[0], 'strftime'):  # If time_array contains datetime objects
                time_diffs = [abs((t - target_dt).total_seconds()) for t in time_array]
            else:  # If time_array contains strings
                time_diffs = [abs((time_to_datetime(t) - target_dt).total_seconds()) 
                             for t in time_array]
            return time_diffs.index(min(time_diffs))
        
        # Find start and end column indices
        start_idx = find_closest_time_index(start_time_str)
        end_idx = find_closest_time_index(end_time_str)
        
        # 从起始时间向前256个时间点
        lookback_range = 256
        min_start_x = max(0, start_idx - lookback_range)
        
        # 在起始时间前256个时间点之间随机选择一个起始点
        if is_training:
            start_x = random.randint(min_start_x, start_idx)
        else:
            # 非训练模式使用确定性起点
            start_x = min_start_x
        
        # 计算从起始点到结束点的距离
        time_range = end_idx - start_x
        
        # 计算能容纳的切片数量（向上取整）
        stride_width = int(self.target_width * (1 - self.overlap_ratio))
        n_time_slices = int(np.ceil(time_range / stride_width))
        
        # 计算频率维度上可以放置多少个切片
        stride_height = int(self.target_height * (1 - self.overlap_ratio))
        n_freq_slices = max(1, int(np.floor(height / stride_height)))
        
        # 如果高度不足以放置两个切片，则只在顶部和底部各放一个
        if height < 2 * self.target_height:
            freq_positions = [0, max(0, height - self.target_height)]
            # 如果两个位置相同，则只保留一个
            freq_positions = list(set(freq_positions))
        else:
            # 否则，创建顶部和底部的切片位置
            top_positions = list(range(0, height // 2, stride_height))
            bottom_positions = list(range(height - self.target_height, height // 2 - self.target_height, -stride_height))
            freq_positions = top_positions + bottom_positions
        
        slices = []
        mask_slices = [] if mask is not None else None
        positions = []
        
        # 遍历时间切片和频率切片位置
        for i in range(n_time_slices):
            # 计算当前时间切片位置
            x = min(start_x + i * stride_width, width - self.target_width)
            
            # 对每个时间位置，创建所有频率位置的切片
            for y in freq_positions:
                # 确保y不超出边界
                y = min(y, height - self.target_height)
                
                # 提取切片
                if len(spectrogram.shape) == 3:  # With channels
                    s = spectrogram[y:y+self.target_height, x:x+self.target_width, :]
                else:  # Without channels
                    s = spectrogram[y:y+self.target_height, x:x+self.target_width]
                
                # 确保切片大小正确
                if s.shape[:2] != (self.target_height, self.target_width):
                    continue
                    
                slices.append(s)
                positions.append((y, x))
                
                if mask is not None:
                    m = mask[y:y+self.target_height, x:x+self.target_width]
                    mask_slices.append(m)
        
        return slices, mask_slices, positions
    
    def reconstruct_from_slices(self, slices, positions, original_shape, weights=None):
        """
        Reconstruct the complete spectrogram from slices
        
        Args:
            slices: List of spectrogram slices
            positions: Position of each slice in the original image [(y, x), ...]
            original_shape: Original spectrogram shape (height, width) or (height, width, channels)
            weights: Weight map for weighted averaging, if None, create Gaussian weights
            
        Returns:
            reconstructed: Reconstructed spectrogram
        """
        # Determine output shape and number of channels
        height, width = original_shape[:2]
        channels = 1
        if len(original_shape) > 2:
            channels = original_shape[2]
            
        # Create output array and count array
        reconstructed = np.zeros((height, width, channels), dtype=np.float32)
        counts = np.zeros((height, width, 1), dtype=np.float32)
        
        # Create weight map (if not provided)
        if weights is None:
            # Create Gaussian weight map, high weight in center, low weight at edges
            y, x = np.mgrid[0:self.target_height, 0:self.target_width]
            center_y, center_x = self.target_height // 2, self.target_width // 2
            # Gaussian weights, sigma can be adjusted
            sigma = min(self.target_height, self.target_width) / 6
            weights = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
            weights = weights[:, :, np.newaxis]
        
        # Merge all slices
        for i, (slice_img, (y, x)) in enumerate(zip(slices, positions)):
            # Ensure slice is 3D
            if len(slice_img.shape) == 2:
                slice_img = slice_img[:, :, np.newaxis]
                
            # Apply weights
            weighted_slice = slice_img * weights
            
            # Add to reconstructed image
            reconstructed[y:y+self.target_height, x:x+self.target_width] += weighted_slice
            counts[y:y+self.target_height, x:x+self.target_width] += weights
            
        # Avoid division by zero
        counts[counts == 0] = 1
        
        # Calculate weighted average
        reconstructed = reconstructed / counts
        
        # If original image is 2D, remove extra dimension
        if len(original_shape) == 2:
            reconstructed = reconstructed[:, :, 0]
            
        return reconstructed
    
    def smooth_boundaries(self, reconstructed, kernel_size=5):
        """
        Apply Gaussian smoothing to reduce discontinuities at stitching boundaries
        
        Args:
            reconstructed: Reconstructed spectrogram
            kernel_size: Gaussian kernel size
            
        Returns:
            smoothed: Smoothed spectrogram
        """
        if len(reconstructed.shape) == 2:
            return cv2.GaussianBlur(reconstructed, (kernel_size, kernel_size), 0)
        else:
            # Apply Gaussian smoothing to each channel
            smoothed = np.zeros_like(reconstructed)
            for c in range(reconstructed.shape[2]):
                smoothed[:, :, c] = cv2.GaussianBlur(reconstructed[:, :, c], 
                                                     (kernel_size, kernel_size), 0)
            return smoothed

    def save_slices_to_csv(self, slices, mask_slices=None, save_dir='./saved_slices', 
                           naming_format='slice_{slice_index}_y{y}_x{x}', 
                           positions=None, metadata=None):
        """
        Save spectrogram slices and masks to CSV files
        
        Args:
            slices: List of spectrogram slices
            mask_slices: List of corresponding mask slices, can be None
            save_dir: Directory to save the files
            naming_format: File naming format, can use the following variables:
                          {slice_index}: Index of the slice
                          {y}, {x}: Coordinates of the slice in the original image (if positions is provided)
                          {meta_*}: Values from metadata dictionary (if metadata is provided)
            positions: List of slice positions in the original image, format [(y, x), ...]
            metadata: Dictionary containing additional information for file naming
        
        Returns:
            List of saved file paths
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        # Iterate through all slices
        for i, slice_data in enumerate(slices):
            # Prepare file naming variables
            naming_vars = {'slice_index': i}
            
            # Add position information (if available)
            if positions and i < len(positions):
                y, x = positions[i]
                naming_vars['y'] = y
                naming_vars['x'] = x
            
            # Add metadata (if available)
            if metadata:
                for key, value in metadata.items():
                    naming_vars[f'meta_{key}'] = value
            
            # Generate filename
            try:
                filename = naming_format.format(**naming_vars)
            except KeyError as e:
                # If naming format contains unavailable variables
                print(f"Warning: Naming format error - {e}")
                filename = f"slice_{i}"
            
            # Save spectrogram slice
            slice_path = os.path.join(save_dir, f"{filename}.csv")
            pd.DataFrame(slice_data).to_csv(slice_path, index=False)
            saved_files.append(slice_path)
            
            # Save corresponding mask (if available)
            if mask_slices and i < len(mask_slices):
                mask_path = os.path.join(save_dir, f"{filename}_mask.csv")
                pd.DataFrame(mask_slices[i]).to_csv(mask_path, index=False)
                saved_files.append(mask_path)
        
        print(f"Saved {len(slices)} spectrogram slices to {save_dir}")
        return saved_files

# Data generator (for training)
class SpectrogramDataGenerator:
    def __init__(self, slicer, batch_size=16, augment=True):
        """
        Initialize data generator
        
        Args:
            slicer: SpectrogramSlicer instance
            batch_size: Batch size
            augment: Whether to perform data augmentation
        """
        self.slicer = slicer
        self.batch_size = batch_size
        self.augment = augment
        
    def flow_from_spectrograms(self, spectrograms, masks=None, shuffle=True):
        """
        Create a data generator from a list of spectrograms
        
        Args:
            spectrograms: List of spectrograms
            masks: List of corresponding masks, can be None
            shuffle: Whether to shuffle the data
            
        Returns:
            Generator, each time producing (batch_x, batch_y) or just batch_x
        """
        while True:
            # All slices and corresponding masks
            all_slices = []
            all_mask_slices = [] if masks is not None else None
            
            # Slice each spectrogram
            for i, spec in enumerate(spectrograms):
                mask = masks[i] if masks is not None else None
                slices, mask_slices, _ = self.slicer.slice_spectrogram_time_range(spec, mask, True)
                
                all_slices.extend(slices)
                if masks is not None:
                    all_mask_slices.extend(mask_slices)
            
            # Convert to arrays
            all_slices = np.array(all_slices)
            if masks is not None:
                all_mask_slices = np.array(all_mask_slices)
            
            # Create indices and shuffle
            indices = np.arange(len(all_slices))
            if shuffle:
                np.random.shuffle(indices)
            
            # Generate batches
            for start_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch_x = all_slices[batch_indices]
                
                if masks is not None:
                    batch_y = all_mask_slices[batch_indices]
                    yield batch_x, batch_y
                else:
                    yield batch_x
    
    def generate_and_save_slices(self, spectrograms, masks=None, save_dir='./generated_slices', 
                                naming_format='gen_slice_{slice_index}', metadata=None):
        """
        Generate slices from spectrograms and save them to CSV files
        
        Args:
            spectrograms: List of spectrograms
            masks: List of corresponding masks, can be None
            save_dir: Directory to save the files
            naming_format: File naming format, can use the following variables:
                          {slice_index}: Index of the slice
                          {spec_index}: Index of the source spectrogram
                          {meta_*}: Values from metadata dictionary (if metadata is provided)
            metadata: Dictionary containing additional information for file naming
        
        Returns:
            List of saved file paths
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        slice_count = 0
        
        # Process each spectrogram
        for spec_idx, spec in enumerate(spectrograms):
            mask = masks[spec_idx] if masks is not None else None
            
            # Generate slices
            slices, mask_slices, positions = self.slicer.slice_spectrogram_time_range(
                spec, mask=mask, is_training=False)
            
            # Prepare naming variables base
            base_naming_vars = {'spec_index': spec_idx}
            
            # Add metadata (if available)
            if metadata:
                for key, value in metadata.items():
                    base_naming_vars[f'meta_{key}'] = value
            
            # Save each slice
            for i, slice_data in enumerate(slices):
                # Complete naming variables
                naming_vars = dict(base_naming_vars)
                naming_vars['slice_index'] = slice_count
                
                # Generate filename
                try:
                    filename = naming_format.format(**naming_vars)
                except KeyError as e:
                    print(f"Warning: Naming format error - {e}")
                    filename = f"gen_slice_{spec_idx}_{i}"
                
                # Save spectrogram slice
                slice_path = os.path.join(save_dir, f"{filename}.csv")
                pd.DataFrame(slice_data).to_csv(slice_path, index=False)
                saved_files.append(slice_path)
                
                # Save corresponding mask (if available)
                if mask_slices:
                    mask_path = os.path.join(save_dir, f"{filename}_mask.csv")
                    pd.DataFrame(mask_slices[i]).to_csv(mask_path, index=False)
                    saved_files.append(mask_path)
                
                slice_count += 1
        
        print(f"Saved {slice_count} generated slices to {save_dir}")
        return saved_files
