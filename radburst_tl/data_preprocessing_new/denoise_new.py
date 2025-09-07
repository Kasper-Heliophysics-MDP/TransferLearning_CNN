"""
Advanced RFI Cleaning for Solar Radio Burst Detection

This module implements sophisticated RFI (Radio Frequency Interference) removal
based on the 6-step approach:
1. Preprocessing with time-median detrending
2. Vertical RFI detection (instantaneous broadband interference)
3. Horizontal RFI detection (narrowband continuous carriers)
4. Fine-grained scattered noise removal
5. Burst protection (thickness-based filtering)
6. Interpolation inpainting

Key features:
- Burst-aware cleaning (reduced sensitivity in known burst regions)
- Type-specific parameter adaptation
- Thickness-based burst protection
- MAD-based robust anomaly detection
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import median_abs_deviation
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk, rectangle
import warnings
warnings.filterwarnings('ignore')


class AdvancedRFICleaner:
    """
    Advanced RFI cleaning with burst protection
    """
    
    def __init__(self, burst_type=None):
        """
        Initialize with type-specific parameters
        
        Args:
            burst_type (int): Type of burst (2, 3, 5) for parameter adaptation
        """
        self.burst_type = burst_type
        self.params = self._get_type_specific_params(burst_type)
    
    def _get_type_specific_params(self, burst_type):
        """
        Get parameters adapted for specific burst types
        """
        TYPE_PARAMS = {
            2: {  # Type 2 - longer events, more conservative cleaning
                "mad_threshold": 3.5,
                "occupancy_threshold": 0.35,
                "protection_factor": 0.3,  # Strong protection
                "min_thickness": 4,
                "coverage_threshold": 0.2,
                "fine_threshold": 2.5
            },
            3: {  # Type 3 - short events, moderate cleaning
                "mad_threshold": 3.0,
                "occupancy_threshold": 0.3,
                "protection_factor": 0.5,
                "min_thickness": 2,
                "coverage_threshold": 0.1,
                "fine_threshold": 3.0
            },
            5: {  # Type 5 - adjust based on actual data
                "mad_threshold": 3.2,
                "occupancy_threshold": 0.32,
                "protection_factor": 0.4,
                "min_thickness": 3,
                "coverage_threshold": 0.18,
                "fine_threshold": 2.8
            }
        }
        
        # Default parameters if type not specified
        default_params = TYPE_PARAMS.get(3)  # Use Type 3 as default
        return TYPE_PARAMS.get(burst_type, default_params)
    
    def step1_preprocess_spectrum(self, S):
        """
        Step 1: Preprocessing with time-median detrending and normalization
        
        Args:
            S: Spectral data of shape (time_points, frequency_channels)
            
        Returns:
            S_tilde: Detrended and normalized spectrum
        """
        print(f"  Step 1: Preprocessing spectrum...")
        
        # Convert to numpy array if DataFrame
        if isinstance(S, pd.DataFrame):
            S = S.values
        
        S = S.astype(np.float64)  # Ensure float precision
        
        # 1. Time-median detrending: S_tilde(f,t) = S(f,t) - median_t(S(f,t))
        print(f"    Applying time-median detrending...")
        S_detrended = np.zeros_like(S)
        
        for f in range(S.shape[1]):  # For each frequency channel
            freq_channel = S[:, f]
            median_baseline = np.median(freq_channel)
            S_detrended[:, f] = freq_channel - median_baseline
        
        # 2. Robust normalization using MAD
        print(f"    Applying robust normalization...")
        S_normalized = np.zeros_like(S_detrended)
        
        for f in range(S_detrended.shape[1]):
            channel = S_detrended[:, f]
            median_val = np.median(channel)
            mad_val = median_abs_deviation(channel)
            
            if mad_val > 0:
                S_normalized[:, f] = (channel - median_val) / mad_val
            else:
                S_normalized[:, f] = channel - median_val
        
        print(f"    Detrending completed. Shape: {S_normalized.shape}")
        return S_normalized
    
    def step2_detect_vertical_rfi(self, S_tilde, protection_mask=None):
        """
        Step 2: Detect vertical RFI (instantaneous broadband interference)
        
        Args:
            S_tilde: Preprocessed spectrum
            protection_mask: Burst protection mask
            
        Returns:
            vertical_rfi_mask: Boolean mask for vertical RFI
        """
        print(f"  Step 2: Detecting vertical RFI...")
        
        time_points, freq_channels = S_tilde.shape
        vertical_rfi_mask = np.zeros(time_points, dtype=bool)
        
        mad_threshold = self.params['mad_threshold']
        coverage_threshold = self.params['coverage_threshold']
        protection_factor = self.params['protection_factor']
        
        for t in range(time_points):
            time_column = S_tilde[t, :]  # Current time, all frequencies
            
            # Calculate MAD-based anomaly score
            median_val = np.median(time_column)
            mad_val = median_abs_deviation(time_column)
            
            if mad_val > 0:
                # Apply burst protection: relax threshold in protected regions
                current_threshold = mad_threshold
                if protection_mask is not None and np.any(protection_mask[t, :]):
                    current_threshold = mad_threshold * (1 + protection_factor)
                
                # Detect anomalous pixels
                anomaly_threshold = median_val + current_threshold * mad_val
                anomalous_pixels = np.sum(time_column > anomaly_threshold)
                
                # If coverage exceeds threshold → vertical RFI
                if anomalous_pixels / freq_channels > coverage_threshold:
                    vertical_rfi_mask[t] = True
        
        # Morphological dilation to handle edges
        kernel = np.ones(3)  # 1D dilation kernel
        vertical_rfi_mask = binary_dilation(vertical_rfi_mask, kernel)
        
        detected_columns = np.sum(vertical_rfi_mask)
        print(f"    Detected {detected_columns} vertical RFI time columns")
        
        return vertical_rfi_mask
    
    def step3_detect_horizontal_rfi(self, S_tilde, protection_mask=None):
        """
        Step 3: Detect horizontal RFI (narrowband continuous carriers)
        
        Args:
            S_tilde: Preprocessed spectrum
            protection_mask: Burst protection mask
            
        Returns:
            horizontal_rfi_mask: Boolean mask for horizontal RFI
        """
        print(f"  Step 3: Detecting horizontal RFI...")
        
        time_points, freq_channels = S_tilde.shape
        horizontal_rfi_mask = np.zeros(freq_channels, dtype=bool)
        
        occupancy_threshold = self.params['occupancy_threshold']
        protection_factor = self.params['protection_factor']
        
        for f in range(freq_channels):
            freq_row = S_tilde[:, f]  # Current frequency, all times
            
            # Calculate occupancy ratio (long-term brightness)
            median_val = np.median(freq_row)
            mad_val = median_abs_deviation(freq_row)
            
            if mad_val > 0:
                # Apply burst protection
                current_threshold = 2.0  # Base threshold for narrowband detection
                if protection_mask is not None and np.any(protection_mask[:, f]):
                    current_threshold = current_threshold * (1 + protection_factor)
                
                threshold = median_val + current_threshold * mad_val
                bright_ratio = np.sum(freq_row > threshold) / time_points
                
                # Apply burst protection to occupancy threshold
                current_occupancy_threshold = occupancy_threshold
                if protection_mask is not None and np.any(protection_mask[:, f]):
                    current_occupancy_threshold = occupancy_threshold * (1 + protection_factor)
                
                # If long-term bright → narrowband carrier RFI
                if bright_ratio > current_occupancy_threshold:
                    horizontal_rfi_mask[f] = True
        
        detected_rows = np.sum(horizontal_rfi_mask)
        print(f"    Detected {detected_rows} horizontal RFI frequency channels")
        
        return horizontal_rfi_mask
    
    def step4_fine_grained_cleaning(self, S_tilde, vertical_mask, horizontal_mask, protection_mask=None):
        """
        Step 4: Fine-grained scattered noise removal
        
        Args:
            S_tilde: Preprocessed spectrum
            vertical_mask: Vertical RFI mask
            horizontal_mask: Horizontal RFI mask
            protection_mask: Burst protection mask
            
        Returns:
            fine_rfi_mask: Boolean mask for scattered RFI
        """
        print(f"  Step 4: Fine-grained scattered noise removal...")
        
        fine_threshold = self.params['fine_threshold']
        protection_factor = self.params['protection_factor']
        
        # Create mask for regions not already identified as RFI
        existing_rfi = np.zeros_like(S_tilde, dtype=bool)
        for t in range(len(vertical_mask)):
            if vertical_mask[t]:
                existing_rfi[t, :] = True
        for f in range(len(horizontal_mask)):
            if horizontal_mask[f]:
                existing_rfi[:, f] = True
        
        # Fine cleaning only in non-RFI regions
        fine_rfi_mask = np.zeros_like(S_tilde, dtype=bool)
        
        # Adaptive thresholding with local statistics
        window_size = 10  # Local window for adaptive threshold
        
        for t in range(S_tilde.shape[0]):
            for f in range(S_tilde.shape[1]):
                if existing_rfi[t, f]:
                    continue  # Skip already identified RFI regions
                
                # Local neighborhood statistics
                t_start = max(0, t - window_size//2)
                t_end = min(S_tilde.shape[0], t + window_size//2)
                f_start = max(0, f - window_size//2)  
                f_end = min(S_tilde.shape[1], f + window_size//2)
                
                local_region = S_tilde[t_start:t_end, f_start:f_end]
                local_median = np.median(local_region)
                local_mad = median_abs_deviation(local_region.flatten())
                
                if local_mad > 0:
                    # Apply burst protection
                    current_threshold = fine_threshold
                    if protection_mask is not None and protection_mask[t, f]:
                        current_threshold = fine_threshold * (1 + protection_factor)
                    
                    threshold = local_median + current_threshold * local_mad
                    
                    if S_tilde[t, f] > threshold:
                        fine_rfi_mask[t, f] = True
        
        detected_pixels = np.sum(fine_rfi_mask)
        print(f"    Detected {detected_pixels} scattered RFI pixels")
        
        return fine_rfi_mask
    
    def step5_thickness_filtering(self, S_tilde, vertical_mask, horizontal_mask, fine_mask):
        """
        Step 5: Thickness-based burst protection filtering
        
        Args:
            S_tilde: Preprocessed spectrum
            vertical_mask, horizontal_mask, fine_mask: RFI masks
            
        Returns:
            filtered_vertical_mask, filtered_horizontal_mask: Protected RFI masks
        """
        print(f"  Step 5: Thickness-based burst protection...")
        
        min_thickness = self.params['min_thickness']
        
        # ✅ FIX: Vertical RFI doesn't need thickness filtering
        # Single time columns are inherently "thin" and should always be cleaned
        print(f"    Vertical RFI: Keeping all {np.sum(vertical_mask)} detected columns (no thickness filter)")
        filtered_vertical = vertical_mask.copy()  # Keep all detected vertical RFI
        
        # Apply thickness filtering only to horizontal RFI and fine RFI
        combined_non_vertical_rfi = np.zeros_like(S_tilde, dtype=bool)
        
        # Apply horizontal RFI
        for f in range(len(horizontal_mask)):
            if horizontal_mask[f]:
                combined_non_vertical_rfi[:, f] = True
        
        # Apply fine RFI  
        combined_non_vertical_rfi = combined_non_vertical_rfi | fine_mask
        
        # Connected component analysis only for non-vertical RFI
        if np.any(combined_non_vertical_rfi):
            labeled_regions = label(combined_non_vertical_rfi)
            regions = regionprops(labeled_regions)
            
            protected_non_vertical_mask = np.zeros_like(combined_non_vertical_rfi)
            burst_preserved = 0
            rfi_confirmed = 0
            
            for region in regions:
                # Check region thickness
                bbox = region.bbox  # (min_row, min_col, max_row, max_col)
                width = bbox[3] - bbox[1]   # Time direction width
                height = bbox[2] - bbox[0]  # Frequency direction height
                
                # Determine if this is thin RFI or thick burst
                is_thin_horizontal = height <= 2 and width >= min_thickness  # Horizontal RFI line
                is_small_scattered = width <= 3 and height <= 3             # Small scattered noise
                
                if is_thin_horizontal or is_small_scattered:
                    # This is likely RFI, keep the cleaning
                    coords = region.coords
                    for coord in coords:
                        protected_non_vertical_mask[coord[0], coord[1]] = True
                    rfi_confirmed += 1
                else:
                    # This might be a thick burst, preserve it
                    burst_preserved += 1
            
            print(f"    Protected {burst_preserved} potential burst regions")
            print(f"    Confirmed {rfi_confirmed} non-vertical RFI regions")
        else:
            protected_non_vertical_mask = np.zeros_like(combined_non_vertical_rfi)
        
        # Reconstruct horizontal mask from thickness filtering
        filtered_horizontal = np.zeros(len(horizontal_mask), dtype=bool)
        for f in range(S_tilde.shape[1]):
            if np.any(protected_non_vertical_mask[:, f]):
                filtered_horizontal[f] = True
        
        # Combine final RFI mask: vertical (unfiltered) + horizontal/fine (filtered)
        final_combined_rfi = np.zeros_like(S_tilde, dtype=bool)
        
        # Add all vertical RFI (no filtering)
        for t in range(len(filtered_vertical)):
            if filtered_vertical[t]:
                final_combined_rfi[t, :] = True
        
        # Add filtered horizontal/fine RFI
        final_combined_rfi = final_combined_rfi | protected_non_vertical_mask
        
        print(f"    Final RFI pixels: {np.sum(final_combined_rfi):,} (including {np.sum(filtered_vertical)} vertical columns)")
        
        return filtered_vertical, filtered_horizontal, final_combined_rfi
    
    def step6_interpolation_inpainting(self, S_original, rfi_mask):
        """
        Step 6: Interpolation inpainting for RFI regions
        
        Args:
            S_original: Original spectrum data
            rfi_mask: Combined RFI mask
            
        Returns:
            S_cleaned: Interpolation-cleaned spectrum
        """
        print(f"  Step 6: Interpolation inpainting...")
        
        S_cleaned = S_original.copy().astype(np.float64)
        total_rfi_pixels = np.sum(rfi_mask)
        
        if total_rfi_pixels == 0:
            print(f"    No RFI pixels to interpolate")
            return S_cleaned
        
        print(f"    Interpolating {total_rfi_pixels} RFI pixels...")
        
        # Find RFI pixels
        rfi_coords = np.where(rfi_mask)
        
        for i, (t, f) in enumerate(zip(rfi_coords[0], rfi_coords[1])):
            # Time-direction interpolation for vertical RFI
            left_val = self._find_good_neighbor(S_cleaned, rfi_mask, t, f, direction='left')
            right_val = self._find_good_neighbor(S_cleaned, rfi_mask, t, f, direction='right')
            
            if left_val is not None and right_val is not None:
                time_interp = (left_val + right_val) / 2
            elif left_val is not None:
                time_interp = left_val
            elif right_val is not None:
                time_interp = right_val
            else:
                time_interp = 0
            
            # Frequency-direction interpolation for horizontal RFI
            upper_val = self._find_good_neighbor(S_cleaned, rfi_mask, t, f, direction='up')
            lower_val = self._find_good_neighbor(S_cleaned, rfi_mask, t, f, direction='down')
            
            if upper_val is not None and lower_val is not None:
                freq_interp = (upper_val + lower_val) / 2
            elif upper_val is not None:
                freq_interp = upper_val
            elif lower_val is not None:
                freq_interp = lower_val
            else:
                freq_interp = 0
            
            # Combine time and frequency interpolations
            if left_val is not None or right_val is not None:
                S_cleaned[t, f] = time_interp
            elif upper_val is not None or lower_val is not None:
                S_cleaned[t, f] = freq_interp
            else:
                # Use local median as fallback
                t_start = max(0, t-5)
                t_end = min(S_cleaned.shape[0], t+5)
                f_start = max(0, f-5)
                f_end = min(S_cleaned.shape[1], f+5)
                local_clean = S_cleaned[t_start:t_end, f_start:f_end][~rfi_mask[t_start:t_end, f_start:f_end]]
                if len(local_clean) > 0:
                    S_cleaned[t, f] = np.median(local_clean)
        
        print(f"    Interpolation completed")
        return S_cleaned
    
    def _find_good_neighbor(self, S, rfi_mask, t, f, direction='left', max_search=10):
        """
        Find nearest non-RFI neighbor for interpolation
        """
        if direction == 'left':
            for dt in range(1, min(max_search, t+1)):
                if not rfi_mask[t-dt, f]:
                    return S[t-dt, f]
        elif direction == 'right':
            for dt in range(1, min(max_search, S.shape[0]-t)):
                if not rfi_mask[t+dt, f]:
                    return S[t+dt, f]
        elif direction == 'up':
            for df in range(1, min(max_search, f+1)):
                if not rfi_mask[t, f-df]:
                    return S[t, f-df]
        elif direction == 'down':
            for df in range(1, min(max_search, S.shape[1]-f)):
                if not rfi_mask[t, f+df]:
                    return S[t, f+df]
        
        return None
    
    def create_burst_protection_mask(self, S_shape, burst_start_idx, burst_end_idx, 
                                   protection_margin=50):
        """
        Create protection mask for known burst regions
        
        Args:
            S_shape: Shape of spectrum data (time_points, frequency_channels)
            burst_start_idx: Burst start index
            burst_end_idx: Burst end index  
            protection_margin: Additional margin around burst (in samples)
            
        Returns:
            protection_mask: Boolean mask for protected regions
        """
        time_points, freq_channels = S_shape
        protection_mask = np.zeros((time_points, freq_channels), dtype=bool)
        
        if burst_start_idx is not None and burst_end_idx is not None:
            # Expand protection range
            protected_start = max(0, burst_start_idx - protection_margin)
            protected_end = min(time_points, burst_end_idx + protection_margin)
            
            # Mark protection region
            protection_mask[protected_start:protected_end, :] = True
            
            print(f"    Burst protection: [{protected_start}, {protected_end}] "
                  f"(margin: {protection_margin} samples)")
        
        return protection_mask
    
    def advanced_rfi_cleaning(self, spectral_data, burst_start_idx=None, burst_end_idx=None, 
                            apply_interpolation=True, verbose=True, fast_mode=False):
        """
        Main function: Complete 6-step advanced RFI cleaning
        
        Args:
            spectral_data: Input spectrum (time_points, frequency_channels)
            burst_start_idx: Known burst start index for protection
            burst_end_idx: Known burst end index for protection
            apply_interpolation: Whether to apply step 6 (interpolation)
            verbose: Print detailed progress
            fast_mode: If True, skip Step 4 (fine-grained cleaning) for speed
            
        Returns:
            cleaned_data: RFI-cleaned spectrum data
            cleaning_masks: Dictionary with all intermediate masks for analysis
        """
        if verbose:
            mode_str = "Fast Mode" if fast_mode else "Comprehensive Mode"
            print(f"\n🧹 Advanced RFI Cleaning (Type {self.burst_type}) - {mode_str}")
            print(f"   Input shape: {spectral_data.shape}")
            if burst_start_idx is not None:
                print(f"   Burst protection: [{burst_start_idx}, {burst_end_idx}]")
            if fast_mode:
                print(f"   ⚡ Fast mode: Skipping Step 4 (fine-grained cleaning) for speed")
        
        # Step 1: Preprocessing
        S_tilde = self.step1_preprocess_spectrum(spectral_data)
        
        # Create burst protection mask
        protection_mask = None
        if burst_start_idx is not None and burst_end_idx is not None:
            protection_mask = self.create_burst_protection_mask(
                spectral_data.shape, burst_start_idx, burst_end_idx
            )
        
        # Step 2: Vertical RFI detection
        vertical_mask = self.step2_detect_vertical_rfi(S_tilde, protection_mask)
        
        # Step 3: Horizontal RFI detection  
        horizontal_mask = self.step3_detect_horizontal_rfi(S_tilde, protection_mask)
        
        # Step 4: Fine-grained cleaning (skip in fast mode)
        if fast_mode:
            if verbose:
                print(f"  Step 4: Skipped (fast mode)")
            fine_mask = np.zeros_like(S_tilde, dtype=bool)  # Empty mask
        else:
            fine_mask = self.step4_fine_grained_cleaning(S_tilde, vertical_mask, horizontal_mask, protection_mask)
        
        # Step 5: Combine masks with thickness filtering
        filtered_vertical, filtered_horizontal, combined_rfi_mask = self.step5_thickness_filtering(
            S_tilde, vertical_mask, horizontal_mask, fine_mask
        )
        
        # Convert to numpy array for processing if needed
        if isinstance(spectral_data, pd.DataFrame):
            spectral_array = spectral_data.values
        else:
            spectral_array = spectral_data
        
        # Step 6: Interpolation inpainting (optional)
        if apply_interpolation:
            cleaned_data = self.step6_interpolation_inpainting(spectral_array, combined_rfi_mask)
        else:
            # Simple masking without interpolation
            cleaned_data = spectral_array.copy()
            cleaned_data[combined_rfi_mask] = np.median(spectral_array)
        
        # Collect masks for analysis
        cleaning_masks = {
            'vertical_rfi': vertical_mask,
            'horizontal_rfi': horizontal_mask,
            'fine_rfi': fine_mask,
            'combined_rfi': combined_rfi_mask,
            'protection_mask': protection_mask,
            'preprocessed': S_tilde
        }
        
        total_cleaned = np.sum(combined_rfi_mask)
        clean_ratio = total_cleaned / combined_rfi_mask.size * 100
        
        if verbose:
            print(f"✅ Advanced RFI cleaning completed!")
            print(f"   Cleaned pixels: {total_cleaned} ({clean_ratio:.2f}%)")
            print(f"   Vertical RFI columns: {np.sum(filtered_vertical)}")
            print(f"   Horizontal RFI rows: {np.sum(filtered_horizontal)}")
        
        return cleaned_data, cleaning_masks


def advanced_rfi_cleaning_wrapper(spectral_data, burst_start_idx=None, burst_end_idx=None, 
                                burst_type=None, method="comprehensive"):
    """
    Wrapper function to replace the old denoising code
    
    This function provides the same interface as the old denoising functions
    but uses the advanced RFI cleaning approach.
    
    Args:
        spectral_data: Input spectrum data
        burst_start_idx: Known burst start index (optional)
        burst_end_idx: Known burst end index (optional)
        burst_type: Type of burst (2, 3, 5) for parameter adaptation
        method: Cleaning method ('comprehensive', 'fast', 'conservative')
        
    Returns:
        cleaned_data: RFI-cleaned spectrum data
    """
    # Initialize cleaner with type-specific parameters
    cleaner = AdvancedRFICleaner(burst_type=burst_type)
    
    # Adjust parameters based on method
    if method == "fast":
        # Skip Step 4 (fine-grained cleaning) and interpolation for speed
        cleaned_data, masks = cleaner.advanced_rfi_cleaning(
            spectral_data, burst_start_idx, burst_end_idx, 
            apply_interpolation=False, verbose=False, fast_mode=True
        )
    elif method == "conservative":
        # Use more relaxed thresholds
        cleaner.params['mad_threshold'] *= 1.5
        cleaner.params['occupancy_threshold'] *= 1.3
        cleaned_data, masks = cleaner.advanced_rfi_cleaning(
            spectral_data, burst_start_idx, burst_end_idx,
            apply_interpolation=True, verbose=True
        )
    else:  # comprehensive (default)
        cleaned_data, masks = cleaner.advanced_rfi_cleaning(
            spectral_data, burst_start_idx, burst_end_idx,
            apply_interpolation=True, verbose=True
        )
    
    # Ensure return numpy array (for compatibility with transpose operations)
    if isinstance(cleaned_data, pd.DataFrame):
        return cleaned_data.values
    else:
        return cleaned_data


# Backward compatibility functions
def remove_horizontal_noise_advanced(spectral_data, burst_info=None, **kwargs):
    """
    Advanced replacement for remove_horizontal_noise
    """
    burst_type = kwargs.get('burst_type', 3)
    return advanced_rfi_cleaning_wrapper(spectral_data, burst_type=burst_type, method="comprehensive")


def remove_vertical_noise_advanced(spectral_data, burst_info=None, **kwargs):
    """
    Advanced replacement for remove_vertical_noise
    This is now handled together with horizontal noise in the comprehensive method
    """
    # For backward compatibility, just return the input
    # The advanced method handles both vertical and horizontal RFI together
    return spectral_data


if __name__ == "__main__":
    # Example usage
    print("Testing Advanced RFI Cleaning...")
    
    # Create sample data for testing
    test_data = np.random.randn(1000, 100) * 0.1
    
    # Add simulated burst (wide structure)
    test_data[300:500, 40:60] += np.random.exponential(2.0, (200, 20))
    
    # Add simulated RFI
    test_data[:, 25] += 5.0  # Horizontal RFI (narrowband carrier)
    test_data[600, :] += 3.0  # Vertical RFI (broadband pulse)
    
    # Test cleaning
    cleaner = AdvancedRFICleaner(burst_type=3)
    cleaned, masks = cleaner.advanced_rfi_cleaning(
        test_data, 
        burst_start_idx=300, 
        burst_end_idx=500,
        apply_interpolation=True
    )
    
    print(f"Original range: [{test_data.min():.3f}, {test_data.max():.3f}]")
    print(f"Cleaned range: [{cleaned.min():.3f}, {cleaned.max():.3f}]")
    print(f"RFI pixels cleaned: {np.sum(masks['combined_rfi'])}")
