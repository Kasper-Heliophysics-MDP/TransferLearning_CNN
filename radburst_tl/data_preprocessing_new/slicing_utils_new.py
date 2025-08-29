"""
Fixed Window Slicing Utilities for GAN Training

This module implements fixed 4-minute window slicing with 50% overlap
for solar radio burst GAN training data preparation.

Based on existing data_slicing.py but designed specifically for:
- Fixed duration windows (4 minutes)
- 50% overlap strategy
- Burst-centered slicing with buffer zones
- Direct scaling to 128x128 for GAN training
"""

import numpy as np
import pandas as pd
import cv2
import os
from datetime import datetime
from tqdm import tqdm

# Import existing utilities from parent directories
import sys
sys.path.append('../data_preprocessing')
from data_label import time_to_column_indices
from data_denoise import remove_horizontal_noise, remove_vertical_noise


class BurstFixedWindowSlicer:
    """
    Fixed window slicer for GAN training data preparation
    
    Creates 4-minute windows with 50% overlap around burst events
    Handles edge cases and ensures burst completeness
    """
    
    def __init__(self, window_duration=4*60, overlap_ratio=0.5, target_size=(128, 128)):
        """
        Initialize the fixed window slicer
        
        Args:
            window_duration (int): Window duration in seconds (default: 4 minutes)
            overlap_ratio (float): Overlap ratio for sliding windows (default: 0.5)
            target_size (tuple): Target size for resized windows (default: (128, 128))
        """
        self.window_duration = window_duration  # 4 minutes = 240 seconds
        self.overlap_ratio = overlap_ratio      # 50% overlap
        self.target_size = target_size          # (128, 128) for GAN
        self.sampling_interval = 0.1            # 100ms sampling interval
        
        # Calculate derived parameters
        self.window_samples = int(window_duration / self.sampling_interval)  # 2400 samples
        self.step_samples = int(self.window_samples * (1 - overlap_ratio))   # 1200 samples (2 min)
        self.buffer_samples = int(2*60 / self.sampling_interval)             # 1200 samples (2 min buffer)
        
        print(f"BurstFixedWindowSlicer initialized:")
        print(f"  Window duration: {window_duration}s ({self.window_samples} samples)")
        print(f"  Overlap: {overlap_ratio*100}% (step: {self.step_samples} samples)")
        print(f"  Buffer zone: 2min ({self.buffer_samples} samples)")
        print(f"  Target size: {target_size}")
    
    def load_and_preprocess_csv(self, csv_file_path, apply_denoising=True):
        """
        Load CSV file and apply preprocessing
        
        Args:
            csv_file_path (str): Path to the raw CSV file
            apply_denoising (bool): Whether to apply noise removal
            
        Returns:
            tuple: (processed_data, times, raw_data)
        """
        print(f"Loading CSV file: {os.path.basename(csv_file_path)}")
        
        # Load raw data
        raw_data = pd.read_csv(csv_file_path, on_bad_lines='skip')
        
        # Extract components
        dates = raw_data['Date']
        times = raw_data['Time']
        spectral_data = raw_data.iloc[:, 2:]  # Skip Date, Time columns
        
        print(f"  Data shape: {spectral_data.shape}")
        print(f"  Time range: {times.iloc[0]} to {times.iloc[-1]}")
        print(f"  Frequency channels: {len(spectral_data.columns)}")
        
        # Apply denoising if requested
        if apply_denoising:
            print("  Applying noise removal...")
            denoised_data = remove_horizontal_noise(spectral_data, num_std=15)
            processed_data = remove_vertical_noise(denoised_data)
        else:
            processed_data = spectral_data
        
        return processed_data, times, raw_data
    
    def transpose_data(self, spectral_data):
        """
        Apply the critical transpose operation: data.T[::-1]
        
        Args:
            spectral_data: Spectral data of shape (time_points, frequency_channels)
            
        Returns:
            numpy.ndarray: Transposed data of shape (frequency_channels, time_points)
        """
        # CRITICAL: Apply the transpose operation as required
        # Original CSV: (time_points, frequency_channels) -> (frequency_channels, time_points)
        transposed_data = spectral_data.T[::-1]
        
        print(f"  Data transposed: {spectral_data.shape} -> {transposed_data.shape}")
        return transposed_data
    
    def calculate_slicing_range(self, burst_start_idx, burst_end_idx, total_length):
        """
        Calculate slicing range based on the three scenarios:
        1. Normal case: burst_start - 2min to burst_end + 2min
        2. Start boundary: from burst_start if too close to beginning
        3. End boundary: backtrack from burst_end if too close to end
        
        Args:
            burst_start_idx (int): Burst start index
            burst_end_idx (int): Burst end index  
            total_length (int): Total data length
            
        Returns:
            tuple: (actual_start_idx, actual_end_idx)
        """
        print(f"  Calculating slicing range...")
        print(f"    Burst: [{burst_start_idx}, {burst_end_idx}] (duration: {burst_end_idx - burst_start_idx} samples)")
        print(f"    Total length: {total_length} samples")
        
        # Scenario 1: Normal case
        ideal_start = burst_start_idx - self.buffer_samples
        ideal_end = burst_end_idx + self.buffer_samples
        
        print(f"    Ideal range: [{ideal_start}, {ideal_end}]")
        
        # Scenario 2: Start boundary handling
        if ideal_start < 0:
            print(f"    Start boundary detected: ideal_start ({ideal_start}) < 0")
            actual_start = max(0, burst_start_idx)  # Start from burst_start or 0
        else:
            actual_start = ideal_start
        
        # Scenario 3: End boundary handling
        if ideal_end + self.window_samples > total_length:
            print(f"    End boundary detected: need space for windows beyond {ideal_end}")
            # Ensure we can fit windows that cover burst_end + buffer
            required_end = burst_end_idx + self.buffer_samples
            # Find the latest start position that still covers the required end
            latest_start_for_coverage = required_end - self.window_samples
            actual_start = min(actual_start, max(0, latest_start_for_coverage))
            actual_end = min(total_length - self.window_samples, required_end)
        else:
            actual_end = ideal_end
        
        print(f"    Final range: [{actual_start}, {actual_end}]")
        print(f"    Coverage: {actual_end - actual_start} samples ({(actual_end - actual_start) * self.sampling_interval / 60:.1f} minutes)")
        
        return actual_start, actual_end
    
    def extract_fixed_windows(self, transposed_data, start_idx, end_idx):
        """
        Extract fixed 4-minute windows with 50% overlap
        
        Args:
            transposed_data: Transposed spectral data (freq_channels, time_points)
            start_idx (int): Start index for slicing
            end_idx (int): End index for slicing
            
        Returns:
            tuple: (windows, positions)
                windows: List of resized windows (128, 128)
                positions: List of x positions for each window
        """
        windows = []
        positions = []
        
        current_pos = start_idx
        window_count = 0
        
        print(f"  Extracting fixed windows...")
        print(f"    Window size: {self.window_samples} samples ({self.window_duration}s)")
        print(f"    Step size: {self.step_samples} samples ({self.step_samples * self.sampling_interval}s)")
        
        while current_pos + self.window_samples <= min(end_idx + self.window_samples, transposed_data.shape[1]):
            # Extract 4-minute window: (freq_channels, window_samples)
            window = transposed_data[:, current_pos:current_pos + self.window_samples]
            
            # Resize to target size (128, 128)
            resized_window = self.resize_window(window)
            
            windows.append(resized_window)
            positions.append(current_pos)
            window_count += 1
            
            # Move to next position (50% overlap = 2-minute step)
            current_pos += self.step_samples
        
        print(f"    Extracted {window_count} windows")
        print(f"    Positions: {positions}")
        
        return windows, positions
    
    def resize_window(self, window_data):
        """
        Resize window from (freq_channels, time_samples) to target size
        
        Args:
            window_data: Window data of shape (freq_channels, time_samples)
            
        Returns:
            numpy.ndarray: Resized window of target_size
        """
        # Convert to float32 for OpenCV
        window_float = window_data.astype(np.float32)
        
        # Resize: (411, 2400) -> (128, 128)
        # Time dimension: 2400 -> 128 (compression ratio: 18.75)
        # Freq dimension: 411 -> 128 (compression ratio: 3.2)
        resized = cv2.resize(window_float, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def generate_filenames(self, positions, burst_start_time, burst_end_time, source_file):
        """
        Generate filenames following the existing naming convention
        
        Args:
            positions: List of x positions
            burst_start_time (str): Burst start time string
            burst_end_time (str): Burst end time string
            source_file (str): Source CSV file path
            
        Returns:
            list: List of generated filenames
        """
        # Extract base name from source file
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        
        # Clean time strings for filename (remove colons and dots)
        clean_start = burst_start_time.replace(':', '').replace('.', '')
        clean_end = burst_end_time.replace(':', '').replace('.', '')
        
        filenames = []
        for i, x_pos in enumerate(positions):
            # Format: window_YYYYMMDD_x{position}_{location}_burst_{start}to{end}
            filename = f"window_{base_name}_x{x_pos}_burst_{clean_start}to{clean_end}"
            filenames.append(filename)
        
        return filenames
    
    def save_windows_to_csv(self, windows, filenames, save_dir):
        """
        Save windows to CSV files
        
        Args:
            windows: List of window arrays
            filenames: List of filenames
            save_dir (str): Directory to save files
            
        Returns:
            list: List of saved file paths
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        print(f"  Saving {len(windows)} windows to {save_dir}")
        
        for window, filename in tqdm(zip(windows, filenames), total=len(windows), desc="Saving windows"):
            # Save window as CSV
            file_path = os.path.join(save_dir, f"{filename}.csv")
            pd.DataFrame(window).to_csv(file_path, index=False, header=False)
            saved_files.append(file_path)
        
        print(f"  Saved {len(saved_files)} files")
        return saved_files
    
    def slice_burst_with_fixed_windows(self, csv_file_path, burst_start_time, burst_end_time, 
                                     save_dir=None, apply_denoising=True):
        """
        Complete pipeline: load data, slice burst region with fixed windows
        
        Args:
            csv_file_path (str): Path to raw CSV file
            burst_start_time (str): Burst start time in 'HH:MM:SS' format
            burst_end_time (str): Burst end time in 'HH:MM:SS' format
            save_dir (str): Directory to save windows (optional)
            apply_denoising (bool): Whether to apply noise removal
            
        Returns:
            dict: Results containing windows, positions, filenames, and metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing burst: {burst_start_time} to {burst_end_time}")
        print(f"Source: {os.path.basename(csv_file_path)}")
        print(f"{'='*60}")
        
        # Step 1: Load and preprocess data
        processed_data, times, raw_data = self.load_and_preprocess_csv(csv_file_path, apply_denoising)
        
        # Step 2: Apply transpose operation
        transposed_data = self.transpose_data(processed_data)
        
        # Step 3: Convert time strings to indices
        print(f"  Converting time strings to indices...")
        start_idx, end_idx = time_to_column_indices(times, burst_start_time, burst_end_time)
        print(f"    Burst indices: [{start_idx}, {end_idx}]")
        
        # Step 4: Calculate slicing range
        actual_start, actual_end = self.calculate_slicing_range(start_idx, end_idx, transposed_data.shape[1])
        
        # Step 5: Extract fixed windows
        windows, positions = self.extract_fixed_windows(transposed_data, actual_start, actual_end)
        
        # Step 6: Generate filenames
        filenames = self.generate_filenames(positions, burst_start_time, burst_end_time, csv_file_path)
        
        # Step 7: Save windows if save_dir provided
        saved_files = []
        if save_dir:
            saved_files = self.save_windows_to_csv(windows, filenames, save_dir)
        
        # Return comprehensive results
        results = {
            'windows': windows,
            'positions': positions, 
            'filenames': filenames,
            'saved_files': saved_files,
            'metadata': {
                'source_file': csv_file_path,
                'burst_start_time': burst_start_time,
                'burst_end_time': burst_end_time,
                'burst_start_idx': start_idx,
                'burst_end_idx': end_idx,
                'slicing_range': (actual_start, actual_end),
                'num_windows': len(windows),
                'window_duration': self.window_duration,
                'overlap_ratio': self.overlap_ratio,
                'target_size': self.target_size
            }
        }
        
        print(f"\n✅ Processing completed!")
        print(f"   Generated {len(windows)} windows of size {self.target_size}")
        if save_dir:
            print(f"   Saved to: {save_dir}")
        
        return results


def process_multiple_bursts(burst_list, save_dir, apply_denoising=True):
    """
    Process multiple bursts from different files
    
    Args:
        burst_list: List of dictionaries with keys:
                   'csv_file': path to CSV file
                   'start_time': burst start time
                   'end_time': burst end time
        save_dir (str): Directory to save all windows
        apply_denoising (bool): Whether to apply noise removal
        
    Returns:
        list: List of results for each burst
    """
    slicer = BurstFixedWindowSlicer()
    all_results = []
    
    print(f"\n🚀 Processing {len(burst_list)} bursts...")
    
    for i, burst_info in enumerate(burst_list):
        print(f"\n--- Processing burst {i+1}/{len(burst_list)} ---")
        
        try:
            result = slicer.slice_burst_with_fixed_windows(
                csv_file_path=burst_info['csv_file'],
                burst_start_time=burst_info['start_time'],
                burst_end_time=burst_info['end_time'],
                save_dir=save_dir,
                apply_denoising=apply_denoising
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing burst {i+1}: {e}")
            continue
    
    # Summary
    total_windows = sum(len(r['windows']) for r in all_results)
    print(f"\n🎉 Batch processing completed!")
    print(f"   Processed: {len(all_results)}/{len(burst_list)} bursts")
    print(f"   Total windows: {total_windows}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    csv_file = "/Users/remiliascarlet/Desktop/MDP/transfer_learning/burst_data/csv/original/240725113837-Skyline High School.csv"
    
    # Create slicer
    slicer = BurstFixedWindowSlicer(window_duration=4*60, overlap_ratio=0.5)
    
    # Process a burst
    result = slicer.slice_burst_with_fixed_windows(
        csv_file_path=csv_file,
        burst_start_time="15:33:52",
        burst_end_time="15:35:47",
        save_dir="./test_windows",
        apply_denoising=True
    )
    
    print(f"Generated {len(result['windows'])} windows")
