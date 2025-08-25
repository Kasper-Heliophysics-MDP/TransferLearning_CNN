"""
Progressive Labeling Strategy for Solar Radio Burst Detection

This module implements a multi-stage progressive labeling approach to create 
high-quality ground truth masks for solar radio burst detection. The strategy
addresses the common trade-off between noise inclusion and boundary preservation
by using a three-stage approach:

1. Stage 1: Conservative core detection with high confidence
2. Stage 2: Region growing from cores with relaxed criteria  
3. Stage 3: Gradient-based boundary refinement
"""

import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening, binary_closing, binary_dilation, binary_erosion
from skimage.morphology import disk, rectangle
from skimage import measure
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
import os


class ProgressiveLabelingStrategy:
    """
    Multi-stage progressive labeling for solar radio burst detection.
    
    This class implements a sophisticated labeling strategy that starts with
    conservative detection of burst cores and progressively expands to include
    likely burst regions, finishing with boundary refinement.
    """
    
    def __init__(self, 
                 conservative_percentile=75,
                 expansion_percentile=40, 
                 min_core_area=150,
                 expansion_radius=3):
        """
        Initialize the progressive labeling strategy.
        
        Args:
            conservative_percentile (float): Percentile threshold for stage 1 core detection.
                                           Higher values = more conservative (less noise, may miss edges)
            expansion_percentile (float): Percentile threshold for stage 2 expansion.
                                        Lower values = more aggressive expansion
            min_core_area (int): Minimum area (in pixels) for valid core regions
            expansion_radius (int): Maximum distance for region growing search
        """
        self.conservative_percentile = conservative_percentile
        self.expansion_percentile = expansion_percentile
        self.min_core_area = min_core_area
        self.expansion_radius = expansion_radius
        
    def create_progressive_mask(self, data, start_index, end_index, 
                              start_time_str=None, end_time_str=None):
        """
        Main function to create progressive labeling mask.
        
        This function orchestrates the three-stage progressive labeling process:
        1. Conservative core detection
        2. Region growing from cores
        3. Boundary refinement
        
        Args:
            data (DataFrame or numpy.ndarray): Spectrogram data
            start_index (int): Start time index of the burst
            end_index (int): End time index of the burst
            start_time_str (str, optional): Start time string for validation
            end_time_str (str, optional): End time string for validation
            
        Returns:
            tuple: (final_mask, stage_results)
                - final_mask: Final binary mask after all stages
                - stage_results: Dictionary containing intermediate results and statistics
        """
        
        # Extract burst region data
        if isinstance(data, pd.DataFrame):
            burst_data = data.iloc[start_index:end_index, :].values
        else:
            burst_data = data[start_index:end_index, :]
        
        print(f"Progressive labeling for burst region: {burst_data.shape}")
        
        # Stage 1: Conservative core detection
        print("Stage 1: Conservative core detection...")
        core_mask, core_stats = self.stage1_conservative_detection(burst_data)
        
        # Stage 2: Region growing from cores
        print("Stage 2: Region growing from cores...")
        expanded_mask, expansion_stats = self.stage2_region_growing(
            burst_data, core_mask)
        
        # Stage 3: Boundary refinement
        print("Stage 3: Boundary refinement...")
        refined_mask, refinement_stats = self.stage3_boundary_refinement(
            burst_data, expanded_mask)
        
        # Collect stage results for analysis
        stage_results = {
            'stage1_core': core_mask,
            'stage2_expanded': expanded_mask, 
            'stage3_refined': refined_mask,
            'stats': {
                'core': core_stats,
                'expansion': expansion_stats,
                'refinement': refinement_stats
            }
        }
        
        print(f"Progressive labeling completed. Final coverage: "
              f"{np.sum(refined_mask)/refined_mask.size*100:.1f}%")
        
        return refined_mask, stage_results
    
    def stage1_conservative_detection(self, burst_data):
        """
        Stage 1: Conservative detection of burst cores.
        
        This stage uses high thresholds and strong morphological operations
        to detect only the most confident burst regions. The goal is to avoid
        false positives even if some true burst edges are missed.
        
        Args:
            burst_data (numpy.ndarray): 2D array of burst region intensities
            
        Returns:
            tuple: (core_mask, core_stats)
                - core_mask: Binary mask of detected cores
                - core_stats: Dictionary with detection statistics
        """
        
        # 1.1 Calculate conservative threshold (high percentile)
        conservative_threshold = np.percentile(burst_data, self.conservative_percentile)
        
        # 1.2 Create initial core mask
        core_mask = burst_data > conservative_threshold
        
        # 1.3 Strong noise removal using morphological operations
        # Use rectangular structuring element to match time-frequency characteristics
        # of radio bursts (longer in time dimension)
        core_mask = binary_opening(core_mask, rectangle(2, 6))  # (height, width)
        
        # 1.4 Filter out regions that are too small to be real bursts
        core_mask = self._filter_small_regions(core_mask, self.min_core_area)
        
        # 1.5 Fill small holes to ensure connectivity
        core_mask = binary_closing(core_mask, disk(2))
        
        # 1.6 Analyze core detection results
        core_stats = self._analyze_core_regions(core_mask, burst_data)
        
        print(f"  - Core regions found: {core_stats['num_regions']}")
        print(f"  - Total core coverage: {core_stats['coverage_percentage']:.1f}%")
        
        return core_mask, core_stats
    
    def stage2_region_growing(self, burst_data, core_mask):
        """
        Stage 2: Region growing from confident cores.
        
        This stage expands from the confident core regions using more relaxed
        criteria. It searches in the neighborhood of cores for pixels that
        likely belong to the burst but were missed in the conservative detection.
        
        Args:
            burst_data (numpy.ndarray): 2D array of burst region intensities
            core_mask (numpy.ndarray): Binary mask of core regions from stage 1
            
        Returns:
            tuple: (expanded_mask, expansion_stats)
                - expanded_mask: Binary mask after region growing
                - expansion_stats: Dictionary with expansion statistics
        """
        
        if not np.any(core_mask):
            print("  - No core regions found, skipping expansion")
            return core_mask, {'expanded_pixels': 0}
        
        # 2.1 Initialize expansion mask with cores
        expanded_mask = core_mask.copy()
        
        # 2.2 Calculate expansion threshold (more relaxed than core threshold)
        expansion_threshold = np.percentile(burst_data, self.expansion_percentile)
        
        # 2.3 Create candidate pixels for expansion
        expansion_candidates = burst_data > expansion_threshold
        
        # 2.4 Iterative region growing from boundary pixels
        grown_pixels = 0
        max_iterations = 10
        
        for iteration in range(max_iterations):
            new_pixels = 0
            
            # Get current boundary of the expanded mask
            current_boundary = self._get_region_boundary(expanded_mask)
            
            # Check neighborhood around each boundary pixel
            for y, x in current_boundary:
                neighbors = self._get_valid_neighbors(
                    y, x, burst_data.shape, self.expansion_radius
                )
                
                # Evaluate each neighbor for inclusion
                for ny, nx in neighbors:
                    if (not expanded_mask[ny, nx] and 
                        expansion_candidates[ny, nx] and
                        self._should_include_neighbor(burst_data, (ny, nx), expanded_mask)):
                        
                        expanded_mask[ny, nx] = True
                        new_pixels += 1
            
            grown_pixels += new_pixels
            print(f"    Iteration {iteration+1}: Added {new_pixels} pixels")
            
            # Stop if no new pixels were added
            if new_pixels == 0:
                break
        
        # 2.5 Light morphological cleanup
        expanded_mask = binary_opening(expanded_mask, disk(1))
        
        expansion_stats = {
            'expanded_pixels': grown_pixels,
            'total_coverage': np.sum(expanded_mask) / expanded_mask.size * 100,
            'expansion_iterations': iteration + 1
        }
        
        print(f"  - Expanded by {grown_pixels} pixels in {iteration+1} iterations")
        print(f"  - Total coverage after expansion: {expansion_stats['total_coverage']:.1f}%")
        
        return expanded_mask, expansion_stats
    
    def stage3_boundary_refinement(self, burst_data, expanded_mask):
        """
        Stage 3: Gradient-based boundary refinement.
        
        This stage uses gradient information to refine the boundaries detected
        in previous stages. It removes weak boundary pixels that are likely
        noise and ensures the final boundaries follow intensity gradients.
        
        Args:
            burst_data (numpy.ndarray): 2D array of burst region intensities
            expanded_mask (numpy.ndarray): Binary mask from stage 2
            
        Returns:
            tuple: (refined_mask, refinement_stats)
                - refined_mask: Final refined binary mask
                - refinement_stats: Dictionary with refinement statistics
        """
        
        if not np.any(expanded_mask):
            return expanded_mask, {'refinement_changes': 0}
        
        # 3.1 Compute intensity gradient magnitude
        gradient_magnitude = self._compute_gradient_magnitude(burst_data)
        
        # 3.2 Refine boundaries based on gradient information
        refined_mask = self._gradient_based_refinement(
            expanded_mask, burst_data, gradient_magnitude
        )
        
        # 3.3 Final morphological cleanup
        refined_mask = self._final_morphological_cleanup(refined_mask)
        
        # 3.4 Calculate refinement statistics
        refinement_stats = {
            'refinement_changes': np.sum(expanded_mask != refined_mask),
            'final_coverage': np.sum(refined_mask) / refined_mask.size * 100,
            'boundary_smoothness': self._calculate_boundary_smoothness(refined_mask)
        }
        
        print(f"  - Boundary refinement changed {refinement_stats['refinement_changes']} pixels")
        print(f"  - Final coverage: {refinement_stats['final_coverage']:.1f}%")
        
        return refined_mask, refinement_stats
    
    def _filter_small_regions(self, mask, min_area):
        """
        Filter out connected regions smaller than minimum area.
        
        Args:
            mask (numpy.ndarray): Binary mask
            min_area (int): Minimum area in pixels
            
        Returns:
            numpy.ndarray: Filtered binary mask
        """
        
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)
        
        filtered_mask = np.zeros_like(mask, dtype=bool)
        
        for prop in props:
            if prop.area >= min_area:
                filtered_mask[labeled == prop.label] = True
        
        return filtered_mask
    
    def _analyze_core_regions(self, core_mask, burst_data):
        """
        Analyze statistical properties of detected core regions.
        
        Args:
            core_mask (numpy.ndarray): Binary mask of core regions
            burst_data (numpy.ndarray): Original intensity data
            
        Returns:
            dict: Statistics about core regions
        """
        
        labeled = measure.label(core_mask)
        props = measure.regionprops(labeled)
        
        stats = {
            'num_regions': len(props),
            'coverage_percentage': np.sum(core_mask) / core_mask.size * 100,
            'avg_intensity': np.mean(burst_data[core_mask]) if np.any(core_mask) else 0,
            'region_sizes': [prop.area for prop in props]
        }
        
        return stats
    
    def _get_region_boundary(self, mask):
        """
        Get boundary pixels of regions in the mask.
        
        Args:
            mask (numpy.ndarray): Binary mask
            
        Returns:
            numpy.ndarray: Array of (y, x) coordinates of boundary pixels
        """
        
        # Use morphological operation to find boundary
        eroded = binary_erosion(mask, disk(1))
        boundary = mask & ~eroded
        
        # Return boundary pixel coordinates
        boundary_coords = np.column_stack(np.where(boundary))
        
        return boundary_coords
    
    def _get_valid_neighbors(self, y, x, shape, radius):
        """
        Get valid neighbor coordinates within specified radius.
        
        Args:
            y, x (int): Center pixel coordinates
            shape (tuple): Shape of the image (height, width)
            radius (int): Search radius
            
        Returns:
            list: List of (y, x) tuples for valid neighbors
        """
        
        neighbors = []
        height, width = shape
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                    
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < height and 0 <= nx < width:
                    neighbors.append((ny, nx))
        
        return neighbors
    
    def _should_include_neighbor(self, burst_data, point, current_mask):
        """
        Determine if a neighbor pixel should be included in the expansion.
        
        This function applies several criteria to decide if a pixel should
        be added to the growing region:
        1. Local continuity check
        2. Connectivity requirement
        
        Args:
            burst_data (numpy.ndarray): Original intensity data
            point (tuple): (y, x) coordinates of the candidate pixel
            current_mask (numpy.ndarray): Current state of the mask
            
        Returns:
            bool: True if pixel should be included
        """
        
        y, x = point
        
        # 1. Check local continuity (intensity gradient)
        if not self._check_local_continuity(burst_data, point, current_mask):
            return False
        
        # 2. Require at least one masked neighbor to avoid isolated pixels
        neighbor_count = self._count_mask_neighbors(point, current_mask)
        if neighbor_count < 1:
            return False
        
        return True
    
    def _check_local_continuity(self, data, point, mask):
        """
        Check if pixel intensity is consistent with local neighborhood.
        
        Args:
            data (numpy.ndarray): Intensity data
            point (tuple): (y, x) coordinates
            mask (numpy.ndarray): Current mask
            
        Returns:
            bool: True if locally consistent
        """
        
        y, x = point
        height, width = data.shape
        
        # Collect 3x3 neighborhood values
        neighborhood_values = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighborhood_values.append(data[ny, nx])
        
        if len(neighborhood_values) < 3:
            return False
        
        # Check if current pixel value is consistent with neighborhood
        neighborhood_std = np.std(neighborhood_values)
        current_value = data[y, x]
        neighborhood_mean = np.mean(neighborhood_values)
        
        # Accept if within 2 standard deviations of neighborhood mean
        return abs(current_value - neighborhood_mean) <= 2 * neighborhood_std
    
    def _count_mask_neighbors(self, point, mask):
        """
        Count how many neighbors of a pixel are already in the mask.
        
        Args:
            point (tuple): (y, x) coordinates
            mask (numpy.ndarray): Binary mask
            
        Returns:
            int: Number of masked neighbors
        """
        
        y, x = point
        height, width = mask.shape
        count = 0
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and mask[ny, nx]):
                    count += 1
        
        return count
    
    def _compute_gradient_magnitude(self, data):
        """
        Compute intensity gradient magnitude using Sobel operator.
        
        Args:
            data (numpy.ndarray): 2D intensity data
            
        Returns:
            numpy.ndarray: Gradient magnitude
        """
        
        # Use Sobel operator to compute gradients
        grad_x = cv2.Sobel(data.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(data.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_magnitude
    
    def _gradient_based_refinement(self, mask, data, gradient_magnitude):
        """
        Refine mask boundaries based on gradient information.
        
        Strong gradients indicate true boundaries, while weak gradients
        may indicate noise that should be removed.
        
        Args:
            mask (numpy.ndarray): Input binary mask
            data (numpy.ndarray): Original intensity data
            gradient_magnitude (numpy.ndarray): Gradient magnitude
            
        Returns:
            numpy.ndarray: Refined binary mask
        """
        
        refined_mask = mask.copy()
        
        # Find current boundary pixels
        boundary = self._get_region_boundary(mask)
        
        # Calculate gradient threshold (pixels with weak gradients may be noise)
        gradient_threshold = np.percentile(gradient_magnitude, 60)
        
        for y, x in boundary:
            # Check if at strong gradient location (true boundary)
            if gradient_magnitude[y, x] > gradient_threshold:
                # Strong gradient: keep boundary
                continue
            else:
                # Weak gradient: check if should be removed
                local_intensity = data[y, x]
                local_mean = self._get_local_mean(data, (y, x), radius=2)
                
                # If current pixel intensity is much lower than local average,
                # it might be a noise boundary
                if local_intensity < local_mean * 0.8:
                    refined_mask[y, x] = False
        
        return refined_mask
    
    def _get_local_mean(self, data, point, radius=2):
        """
        Calculate local mean intensity around a point.
        
        Args:
            data (numpy.ndarray): Intensity data
            point (tuple): (y, x) coordinates
            radius (int): Radius for local region
            
        Returns:
            float: Local mean intensity
        """
        
        y, x = point
        height, width = data.shape
        
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)
        
        local_region = data[y_min:y_max, x_min:x_max]
        
        return np.mean(local_region)
    
    def _final_morphological_cleanup(self, mask):
        """
        Apply final morphological operations to clean up the mask.
        
        Args:
            mask (numpy.ndarray): Input binary mask
            
        Returns:
            numpy.ndarray: Cleaned binary mask
        """
        
        # Remove isolated small dots
        cleaned_mask = binary_opening(mask, disk(1))
        
        # Fill small holes
        cleaned_mask = binary_closing(cleaned_mask, disk(2))
        
        # Final small region filtering
        cleaned_mask = self._filter_small_regions(cleaned_mask, min_area=50)
        
        return cleaned_mask
    
    def _calculate_boundary_smoothness(self, mask):
        """
        Calculate a simple boundary smoothness metric.
        
        Args:
            mask (numpy.ndarray): Binary mask
            
        Returns:
            float: Smoothness metric (ratio of convex hull perimeter to boundary length)
        """
        
        boundary = self._get_region_boundary(mask)
        
        if len(boundary) < 10:
            return 0
        
        # Calculate ratio of convex hull perimeter to actual boundary length
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(boundary)
            smoothness = hull.volume / len(boundary)  # volume is perimeter in 2D
        except:
            smoothness = 1.0
        
        return smoothness


def create_srb_mask_progressive(data, start_index, end_index, 
                               use_progressive=True, **kwargs):
    """
    Create SRB mask with support for both traditional and progressive methods.
    
    This function provides a unified interface for creating solar radio burst
    masks using either the traditional single-stage approach or the new
    progressive multi-stage approach.
    
    Args:
        data (DataFrame or numpy.ndarray): Spectrogram data
        start_index (int): Start time index of the burst
        end_index (int): End time index of the burst
        use_progressive (bool): Whether to use progressive labeling
        **kwargs: Additional parameters
            - For progressive method:
                - conservative_percentile (float): Stage 1 threshold (default: 75)
                - expansion_percentile (float): Stage 2 threshold (default: 40)
                - min_core_area (int): Minimum core area (default: 150)
                - save_intermediate (bool): Save intermediate results (default: False)
                - save_dir (str): Directory for intermediate results (default: './')
            - For traditional method:
                - pct_threshold (float): Percentile threshold (default: 37)
                - erosion_radius (int): Erosion radius (default: 20)
                - dilation_radius (int): Dilation radius (default: 25)
                - window_size (int): Rolling median window size (default: 5)
    
    Returns:
        tuple: (final_mask, stage_results)
            - final_mask: Final binary mask
            - stage_results: Dictionary with stage information (None for traditional method)
    """
    
    if use_progressive:
        # Use new progressive method
        progressive_labeler = ProgressiveLabelingStrategy(
            conservative_percentile=kwargs.get('conservative_percentile', 75),
            expansion_percentile=kwargs.get('expansion_percentile', 40),
            min_core_area=kwargs.get('min_core_area', 150),
            expansion_radius=kwargs.get('expansion_radius', 3)
        )
        
        final_mask, stage_results = progressive_labeler.create_progressive_mask(
            data, start_index, end_index
        )
        
        # Optionally save intermediate results for analysis
        if kwargs.get('save_intermediate', False):
            save_intermediate_results(stage_results, kwargs.get('save_dir', './'))
        
        return final_mask, stage_results
    
    else:
        # Use traditional method - import from existing data_label.py
        from .data_label import create_srb_mask, apply_morphological_operations, apply_rolling_median_filter
        
        initial_mask = create_srb_mask(data, start_index, end_index, 
                                     kwargs.get('pct_threshold', 37))
        morph_mask = apply_morphological_operations(
            initial_mask, 
            erosion_radius=kwargs.get('erosion_radius', 20),
            dilation_radius=kwargs.get('dilation_radius', 25)
        )
        final_mask = apply_rolling_median_filter(morph_mask, 
                                               window_size=kwargs.get('window_size', 5))
        
        return final_mask, None


def save_intermediate_results(stage_results, save_dir):
    """
    Save intermediate results from progressive labeling for analysis and debugging.
    
    This function creates visualizations and saves statistics from each stage
    of the progressive labeling process.
    
    Args:
        stage_results (dict): Results from progressive labeling stages
        save_dir (str): Directory to save results
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create visualization of all stages
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each stage
    stages = ['stage1_core', 'stage2_expanded', 'stage3_refined']
    titles = ['Stage 1: Core Detection', 'Stage 2: Region Growing', 'Stage 3: Boundary Refinement']
    
    for i, (stage, title) in enumerate(zip(stages, titles)):
        axes[i].imshow(stage_results[stage], aspect='auto', cmap='gray', origin='lower')
        axes[i].set_title(title)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'progressive_labeling_stages.png'), dpi=150)
    plt.close()
    
    # Save statistics to text file
    stats_file = os.path.join(save_dir, 'labeling_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Progressive Labeling Statistics\n")
        f.write("=" * 40 + "\n")
        for stage, stats in stage_results['stats'].items():
            f.write(f"\n{stage.upper()} STAGE:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"Intermediate results saved to {save_dir}")


def compare_labeling_methods(data, start_index, end_index, save_dir='./comparison/'):
    """
    Compare traditional and progressive labeling methods side by side.
    
    This function runs both labeling approaches on the same data and creates
    a comparison visualization and statistics.
    
    Args:
        data: Spectrogram data
        start_index (int): Start time index of the burst
        end_index (int): End time index of the burst
        save_dir (str): Directory to save comparison results
    
    Returns:
        dict: Comparison results and statistics
    """
    
    print("Comparing traditional and progressive labeling methods...")
    
    # Run traditional method
    print("\nRunning traditional method...")
    traditional_mask, _ = create_srb_mask_progressive(
        data, start_index, end_index, use_progressive=False
    )
    
    # Run progressive method
    print("\nRunning progressive method...")
    progressive_mask, stage_results = create_srb_mask_progressive(
        data, start_index, end_index, use_progressive=True
    )
    
    # Calculate comparison statistics
    traditional_coverage = np.sum(traditional_mask) / traditional_mask.size * 100
    progressive_coverage = np.sum(progressive_mask) / progressive_mask.size * 100
    
    # Calculate overlap and differences
    overlap = np.sum(traditional_mask & progressive_mask) / np.sum(traditional_mask | progressive_mask) * 100
    traditional_only = np.sum(traditional_mask & ~progressive_mask)
    progressive_only = np.sum(progressive_mask & ~traditional_mask)
    
    comparison_stats = {
        'traditional_coverage': traditional_coverage,
        'progressive_coverage': progressive_coverage,
        'overlap_percentage': overlap,
        'traditional_only_pixels': traditional_only,
        'progressive_only_pixels': progressive_only
    }
    
    # Create comparison visualization
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    if isinstance(data, pd.DataFrame):
        plot_data = data.iloc[start_index:end_index, :].values
    else:
        plot_data = data[start_index:end_index, :]
    
    axes[0, 0].imshow(plot_data, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 0].set_title('Original Data')
    
    # Traditional method
    axes[0, 1].imshow(traditional_mask, aspect='auto', cmap='gray', origin='lower')
    axes[0, 1].set_title(f'Traditional Method ({traditional_coverage:.1f}% coverage)')
    
    # Progressive method
    axes[1, 0].imshow(progressive_mask, aspect='auto', cmap='gray', origin='lower')
    axes[1, 0].set_title(f'Progressive Method ({progressive_coverage:.1f}% coverage)')
    
    # Difference map
    diff_map = np.zeros_like(traditional_mask, dtype=int)
    diff_map[traditional_mask & progressive_mask] = 1  # Both methods
    diff_map[traditional_mask & ~progressive_mask] = 2  # Traditional only
    diff_map[progressive_mask & ~traditional_mask] = 3  # Progressive only
    
    im = axes[1, 1].imshow(diff_map, aspect='auto', cmap='Set1', origin='lower')
    axes[1, 1].set_title(f'Difference Map (Overlap: {overlap:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=150)
    plt.close()
    
    # Save comparison statistics
    stats_file = os.path.join(save_dir, 'comparison_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Labeling Method Comparison\n")
        f.write("=" * 30 + "\n")
        f.write(f"Traditional coverage: {traditional_coverage:.2f}%\n")
        f.write(f"Progressive coverage: {progressive_coverage:.2f}%\n")
        f.write(f"Overlap percentage: {overlap:.2f}%\n")
        f.write(f"Traditional only pixels: {traditional_only}\n")
        f.write(f"Progressive only pixels: {progressive_only}\n")
    
    print(f"Comparison results saved to {save_dir}")
    
    return comparison_stats


# Example usage functions
def example_basic_usage():
    """
    Example of basic usage of progressive labeling.
    """
    # This is a template - replace with your actual data
    # data = your_spectrogram_data
    # start_index = your_burst_start
    # end_index = your_burst_end
    
    # Basic progressive labeling
    # progressive_mask, stage_info = create_srb_mask_progressive(
    #     data=data,
    #     start_index=start_index, 
    #     end_index=end_index,
    #     use_progressive=True
    # )
    
    pass


def example_custom_parameters():
    """
    Example of using progressive labeling with custom parameters.
    """
    # This is a template - replace with your actual data
    # data = your_spectrogram_data
    # start_index = your_burst_start
    # end_index = your_burst_end
    
    # Progressive labeling with custom parameters
    # progressive_mask, stage_info = create_srb_mask_progressive(
    #     data=data,
    #     start_index=start_index,
    #     end_index=end_index, 
    #     use_progressive=True,
    #     conservative_percentile=80,  # More conservative core detection
    #     expansion_percentile=35,     # More aggressive expansion
    #     min_core_area=100,          # Smaller minimum core area
    #     save_intermediate=True,      # Save intermediate results
    #     save_dir='./progressive_results/'
    # )
    
    pass


def example_method_comparison():
    """
    Example of comparing traditional and progressive methods.
    """
    # This is a template - replace with your actual data
    # data = your_spectrogram_data
    # start_index = your_burst_start
    # end_index = your_burst_end
    
    # Compare both methods
    # comparison_results = compare_labeling_methods(
    #     data, start_index, end_index, save_dir='./method_comparison/'
    # )
    # 
    # print("Comparison completed:")
    # print(f"Traditional coverage: {comparison_results['traditional_coverage']:.1f}%")
    # print(f"Progressive coverage: {comparison_results['progressive_coverage']:.1f}%")
    # print(f"Overlap: {comparison_results['overlap_percentage']:.1f}%")
    
    pass
