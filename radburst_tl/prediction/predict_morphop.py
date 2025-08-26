"""
Smart Post-Processing for Solar Radio Burst Prediction

This module implements intelligent post-processing techniques for improving
prediction results from deep learning models. It focuses on adaptive
morphological operations and connected component analysis to refine
predicted masks while preserving important signal characteristics.

Key Features:
1. Adaptive morphological operations based on prediction confidence
2. Connected component analysis and filtering  
3. Multi-level confidence-aware processing
4. Physics-informed constraints for radio burst characteristics
"""

import numpy as np
import cv2
from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
from scipy.ndimage import label, find_objects
from skimage.morphology import disk, rectangle, remove_small_objects, remove_small_holes
from skimage import measure
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any


class AdaptiveMorphologyProcessor:
    """
    Adaptive morphological operations processor that adjusts parameters
    based on prediction confidence and local signal characteristics.
    
    Confidence Level Determination:
    - High confidence (>0.8): Model is very certain, preserve boundaries
    - Medium confidence (0.5-0.8): Model is moderately certain, balanced processing
    - Low confidence (<0.5): Model is uncertain, aggressive noise removal
    
    These thresholds should be calibrated based on your specific model's
    confidence distribution and validation performance.
    """
    
    def __init__(self,
                 high_conf_threshold: float = 0.8,
                 med_conf_threshold: float = 0.5,
                 min_object_size: int = 50,
                 max_hole_size: int = 20):
        """
        Initialize the adaptive morphology processor.
        
        Args:
            high_conf_threshold: Threshold for high confidence regions
            med_conf_threshold: Threshold for medium confidence regions  
            min_object_size: Minimum size for objects to keep
            max_hole_size: Maximum size of holes to fill
        """
        self.high_conf_threshold = high_conf_threshold
        self.med_conf_threshold = med_conf_threshold
        self.min_object_size = min_object_size
        self.max_hole_size = max_hole_size
        
        # Define structuring elements for different confidence levels
        self.high_conf_kernel = disk(1)      # Small kernel for high confidence
        self.med_conf_kernel = disk(2)       # Medium kernel for medium confidence  
        self.low_conf_kernel = disk(3)       # Large kernel for low confidence
        
        # Radio burst specific kernels (elongated in time dimension)
        self.burst_kernel_small = rectangle(1, 3)   # Height=1, Width=3
        self.burst_kernel_medium = rectangle(2, 5)  # Height=2, Width=5
        self.burst_kernel_large = rectangle(3, 7)   # Height=3, Width=7
        
    def adaptive_morphological_operations(self, 
                                        predicted_mask: np.ndarray,
                                        confidence_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply adaptive morphological operations based on confidence levels.
        
        Args:
            predicted_mask: Binary or probability mask from model prediction
            confidence_map: Optional confidence map (if None, uses predicted_mask as confidence)
            
        Returns:
            Refined binary mask after adaptive morphological operations
        """
        
        # Convert to binary if needed
        if predicted_mask.dtype != bool:
            binary_mask = predicted_mask > 0.5
        else:
            binary_mask = predicted_mask.copy()
            
        # Use predicted mask as confidence if not provided
        if confidence_map is None:
            confidence_map = predicted_mask.astype(float)
            
        # Initialize output mask
        refined_mask = np.zeros_like(binary_mask, dtype=bool)
        
        # 1. High confidence regions - preserve details with minimal processing
        high_conf_regions = confidence_map >= self.high_conf_threshold
        high_conf_mask = binary_mask & high_conf_regions
        
        if np.any(high_conf_mask):
            # Light cleaning only - preserve boundaries
            processed_high = binary_opening(high_conf_mask, self.high_conf_kernel)
            processed_high = binary_closing(processed_high, self.burst_kernel_small)
            refined_mask |= processed_high
            
        # 2. Medium confidence regions - moderate processing
        med_conf_regions = (confidence_map >= self.med_conf_threshold) & (confidence_map < self.high_conf_threshold)
        med_conf_mask = binary_mask & med_conf_regions
        
        if np.any(med_conf_mask):
            # Moderate morphological operations
            processed_med = binary_opening(med_conf_mask, self.med_conf_kernel)
            processed_med = binary_closing(processed_med, self.burst_kernel_medium)
            # Remove very small isolated regions
            processed_med = remove_small_objects(processed_med, min_size=self.min_object_size//2)
            refined_mask |= processed_med
            
        # 3. Low confidence regions - aggressive noise removal
        low_conf_regions = confidence_map < self.med_conf_threshold
        low_conf_mask = binary_mask & low_conf_regions
        
        if np.any(low_conf_mask):
            # Aggressive cleaning for noisy low-confidence regions
            processed_low = binary_opening(low_conf_mask, self.low_conf_kernel)
            processed_low = binary_closing(processed_low, self.burst_kernel_large)
            # Remove small objects more aggressively
            processed_low = remove_small_objects(processed_low, min_size=self.min_object_size)
            refined_mask |= processed_low
            
        # 4. Final cleanup - fill small holes and remove tiny objects
        refined_mask = remove_small_holes(refined_mask, area_threshold=self.max_hole_size)
        refined_mask = remove_small_objects(refined_mask, min_size=self.min_object_size//4)
        
        return refined_mask
    
    def visualize_adaptive_processing(self,
                                    original_mask: np.ndarray,
                                    confidence_map: np.ndarray,
                                    refined_mask: np.ndarray,
                                    save_path: Optional[str] = None):
        """
        Visualize the adaptive morphological processing results.
        
        Args:
            original_mask: Original predicted mask
            confidence_map: Confidence map
            refined_mask: Processed mask
            save_path: Optional path to save the visualization
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original mask
        axes[0, 0].imshow(original_mask.T[::-1], aspect='auto', cmap='gray', origin='lower')
        axes[0, 0].set_title('Original Predicted Mask')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        
        # Confidence map
        im1 = axes[0, 1].imshow(confidence_map.T[::-1], aspect='auto', cmap='viridis', origin='lower')
        axes[0, 1].set_title('Confidence Map')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Refined mask
        axes[0, 2].imshow(refined_mask.T[::-1], aspect='auto', cmap='gray', origin='lower')
        axes[0, 2].set_title('Adaptive Morphology Result')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Frequency')
        
        # Confidence regions
        high_conf = confidence_map >= self.high_conf_threshold
        med_conf = (confidence_map >= self.med_conf_threshold) & (confidence_map < self.high_conf_threshold)
        low_conf = confidence_map < self.med_conf_threshold
        
        confidence_regions = np.zeros_like(confidence_map)
        confidence_regions[low_conf] = 1
        confidence_regions[med_conf] = 2  
        confidence_regions[high_conf] = 3
        
        axes[1, 0].imshow(confidence_regions.T[::-1], aspect='auto', cmap='RdYlGn', origin='lower')
        axes[1, 0].set_title('Confidence Regions\n(Red=Low, Yellow=Med, Green=High)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Frequency')
        
        # Difference map
        difference = refined_mask.astype(int) - original_mask.astype(int)
        axes[1, 1].imshow(difference.T[::-1], aspect='auto', cmap='RdBu', origin='lower', vmin=-1, vmax=1)
        axes[1, 1].set_title('Changes\n(Blue=Removed, Red=Added)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Frequency')
        
        # Statistics
        original_pixels = np.sum(original_mask)
        refined_pixels = np.sum(refined_mask)
        added_pixels = np.sum(refined_mask & ~original_mask)
        removed_pixels = np.sum(original_mask & ~refined_mask)
        
        stats_text = f"""Adaptive Morphology Statistics:
        
Original pixels: {original_pixels:,}
Refined pixels: {refined_pixels:,}
Added pixels: {added_pixels:,}
Removed pixels: {removed_pixels:,}
Net change: {refined_pixels - original_pixels:+,}
        
Confidence Distribution:
High conf: {np.sum(high_conf):,} pixels
Med conf: {np.sum(med_conf):,} pixels  
Low conf: {np.sum(low_conf):,} pixels"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Adaptive morphology visualization saved to {save_path}")
            
        plt.show()


class ConnectedComponentAnalyzer:
    """
    Connected component analyzer for filtering and validating prediction regions
    based on size, shape, and radio burst characteristics.
    """
    
    def __init__(self,
                 min_component_size: int = 100,
                 max_component_size: int = 50000,
                 min_aspect_ratio: float = 0.1,
                 max_aspect_ratio: float = 10.0,
                 min_solidity: float = 0.3):
        """
        Initialize the connected component analyzer.
        
        Args:
            min_component_size: Minimum component size to keep
            max_component_size: Maximum component size to keep  
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            min_solidity: Minimum solidity (area/convex_hull_area)
        """
        self.min_component_size = min_component_size
        self.max_component_size = max_component_size
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        
    def analyze_connected_components(self, 
                                   binary_mask: np.ndarray,
                                   confidence_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyze and filter connected components based on radio burst characteristics.
        
        Args:
            binary_mask: Binary mask to analyze
            confidence_map: Optional confidence map for weighted analysis
            
        Returns:
            Tuple of (filtered_mask, analysis_stats)
        """
        
        # Label connected components
        labeled_mask, num_components = label(binary_mask)
        
        # Initialize filtered mask
        filtered_mask = np.zeros_like(binary_mask, dtype=bool)
        
        # Get region properties
        regions = measure.regionprops(labeled_mask, intensity_image=confidence_map)
        
        analysis_stats = {
            'original_components': num_components,
            'kept_components': 0,
            'removed_components': 0,
            'removal_reasons': {
                'too_small': 0,
                'too_large': 0,
                'bad_aspect_ratio': 0,
                'low_solidity': 0,
                'low_confidence': 0,
                'suspicious_noise': 0
            },
            'component_info': []
        }
        
        for region in regions:
            keep_component = True
            removal_reason = None
            
            # Extract component properties
            area = region.area
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            aspect_ratio = width / max(height, 1)
            solidity = region.solidity
            
            # Average confidence if available
            avg_confidence = region.mean_intensity if confidence_map is not None else 1.0
            
            # Component info for analysis
            component_info = {
                'label': region.label,
                'area': area,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'avg_confidence': avg_confidence,
                'centroid': region.centroid
            }
            
            # Apply filters
            if area < self.min_component_size:
                keep_component = False
                removal_reason = 'too_small'
            elif area > self.max_component_size:
                keep_component = False
                removal_reason = 'too_large'
            elif aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                keep_component = False
                removal_reason = 'bad_aspect_ratio'
            elif solidity < self.min_solidity:
                keep_component = False
                removal_reason = 'low_solidity'
            elif confidence_map is not None and avg_confidence < 0.3:
                keep_component = False
                removal_reason = 'low_confidence'
            elif confidence_map is not None and self._is_suspicious_high_confidence_noise(region, confidence_map, area):
                keep_component = False
                removal_reason = 'suspicious_noise'
                
            # Update statistics and mask
            if keep_component:
                filtered_mask[labeled_mask == region.label] = True
                analysis_stats['kept_components'] += 1
                component_info['kept'] = True
            else:
                analysis_stats['removed_components'] += 1
                analysis_stats['removal_reasons'][removal_reason] += 1
                component_info['kept'] = False
                component_info['removal_reason'] = removal_reason
                
            analysis_stats['component_info'].append(component_info)
            
        return filtered_mask, analysis_stats
    
    def merge_nearby_components(self, 
                              binary_mask: np.ndarray,
                              max_distance: int = 10,
                              min_confidence: float = 0.4) -> np.ndarray:
        """
        Merge nearby components that likely belong to the same burst.
        
        Args:
            binary_mask: Binary mask with components to merge
            max_distance: Maximum distance for merging
            min_confidence: Minimum confidence for merging decision
            
        Returns:
            Mask with merged components
        """
        
        # Label components
        labeled_mask, num_components = label(binary_mask)
        
        if num_components <= 1:
            return binary_mask
            
        # Get component centroids
        regions = measure.regionprops(labeled_mask)
        centroids = [region.centroid for region in regions]
        labels = [region.label for region in regions]
        
        # Create distance matrix
        distances = np.zeros((num_components, num_components))
        for i in range(num_components):
            for j in range(i+1, num_components):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                             (centroids[i][1] - centroids[j][1])**2)
                distances[i, j] = distances[j, i] = dist
        
        # Merge nearby components
        merged_mask = binary_mask.copy()
        
        for i in range(num_components):
            for j in range(i+1, num_components):
                if distances[i, j] <= max_distance:
                    # Create bridge between components
                    y1, x1 = map(int, centroids[i])
                    y2, x2 = map(int, centroids[j])
                    
                    # Draw line between centroids
                    rr, cc = self._draw_line(y1, x1, y2, x2, merged_mask.shape)
                    
                    # Apply morphological dilation to create bridge
                    bridge_mask = np.zeros_like(merged_mask)
                    bridge_mask[rr, cc] = True
                    bridge_mask = binary_dilation(bridge_mask, disk(2))
                    
                    merged_mask |= bridge_mask
        
        return merged_mask
    
    def _draw_line(self, y1: int, x1: int, y2: int, x2: int, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw a line between two points using Bresenham's algorithm.
        
        Args:
            y1, x1: Start point
            y2, x2: End point  
            shape: Shape of the image
            
        Returns:
            Arrays of row and column indices
        """
        
        # Simple line drawing - can be replaced with skimage.draw.line
        y1, x1, y2, x2 = map(int, [y1, x1, y2, x2])
        
        points_y = []
        points_x = []
        
        # Handle vertical and horizontal lines
        if x1 == x2:  # Vertical line
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= y < shape[0] and 0 <= x1 < shape[1]:
                    points_y.append(y)
                    points_x.append(x1)
        elif y1 == y2:  # Horizontal line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= y1 < shape[0] and 0 <= x < shape[1]:
                    points_y.append(y1)
                    points_x.append(x)
        else:
            # Diagonal line - simple interpolation
            steps = max(abs(x2 - x1), abs(y2 - y1))
            for i in range(steps + 1):
                y = int(y1 + (y2 - y1) * i / steps)
                x = int(x1 + (x2 - x1) * i / steps)
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    points_y.append(y)
                    points_x.append(x)
        
        return np.array(points_y), np.array(points_x)
    
    def visualize_component_analysis(self,
                                   original_mask: np.ndarray,
                                   filtered_mask: np.ndarray,
                                   analysis_stats: Dict[str, Any],
                                   save_path: Optional[str] = None):
        """
        Visualize connected component analysis results.
        
        Args:
            original_mask: Original binary mask
            filtered_mask: Filtered mask after component analysis
            analysis_stats: Analysis statistics from analyze_connected_components
            save_path: Optional path to save visualization
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original mask
        axes[0, 0].imshow(original_mask.T[::-1], aspect='auto', cmap='gray', origin='lower')
        axes[0, 0].set_title('Original Mask')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        
        # Filtered mask
        axes[0, 1].imshow(filtered_mask.T[::-1], aspect='auto', cmap='gray', origin='lower')
        axes[0, 1].set_title('Filtered Mask')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')
        
        # Component labels (colored)
        original_labeled, _ = label(original_mask)
        axes[1, 0].imshow(original_labeled.T[::-1], aspect='auto', cmap='tab20', origin='lower')
        axes[1, 0].set_title('Original Components (Colored)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Frequency')
        
        # Statistics
        stats_text = f"""Connected Component Analysis:

Original components: {analysis_stats['original_components']}
Kept components: {analysis_stats['kept_components']}
Removed components: {analysis_stats['removed_components']}

Removal Reasons:
• Too small: {analysis_stats['removal_reasons']['too_small']}
• Too large: {analysis_stats['removal_reasons']['too_large']}  
• Bad aspect ratio: {analysis_stats['removal_reasons']['bad_aspect_ratio']}
• Low solidity: {analysis_stats['removal_reasons']['low_solidity']}
• Low confidence: {analysis_stats['removal_reasons']['low_confidence']}

Filtering efficiency: {analysis_stats['kept_components']}/{analysis_stats['original_components']} = {analysis_stats['kept_components']/max(analysis_stats['original_components'],1)*100:.1f}%"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Component analysis visualization saved to {save_path}")
            
        plt.show()
    
    def _is_suspicious_high_confidence_noise(self, region, confidence_map, area):
        """
        Detect suspicious high-confidence noise regions.
        
        This method identifies small, isolated regions with unusually high confidence
        that are likely to be noise rather than real radio bursts.
        
        Args:
            region: Region properties from skimage.measure.regionprops
            confidence_map: Full confidence map
            area: Area of the region
            
        Returns:
            bool: True if the region is suspected to be high-confidence noise
        """
        
        avg_confidence = region.mean_intensity if hasattr(region, 'mean_intensity') else 0
        
        # Criteria for suspicious high-confidence noise:
        
        # 1. Small but high confidence (classic noise pattern)
        if area < 200 and avg_confidence > 0.7:  # Lowered threshold since we're more aggressive
            return True
            
        # 2. Very small with any high confidence  
        if area < 50 and avg_confidence > 0.5:   # Lowered threshold for tiny regions
            return True
            
        # 3. Check isolation (surrounded mostly by low confidence)
        if avg_confidence > 0.6 and area < 1000:
            # Get bounding box and check neighborhood
            bbox = region.bbox
            y_min, x_min, y_max, x_max = bbox
            
            # Expand bounding box to check surroundings
            pad = 20
            y_min_exp = max(0, y_min - pad)
            x_min_exp = max(0, x_min - pad)
            y_max_exp = min(confidence_map.shape[0], y_max + pad)
            x_max_exp = min(confidence_map.shape[1], x_max + pad)
            
            # Get neighborhood confidence
            neighborhood = confidence_map[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
            neighborhood_mean = np.mean(neighborhood)
            
            # If this region is much higher confidence than neighborhood, it's suspicious
            if avg_confidence > neighborhood_mean + 0.4:  # 40% higher than surroundings
                return True
        
        # 4. Check for unrealistic aspect ratios combined with high confidence
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = width / max(height, 1)
        
        # Very thin or very wide regions with high confidence are often noise
        if avg_confidence > 0.6 and (aspect_ratio > 15 or aspect_ratio < 0.07):
            return True
            
        # 5. Check solidity - noisy regions often have irregular shapes
        if avg_confidence > 0.7 and region.solidity < 0.4 and area < 500:
            return True
            
        return False


class SmartPostProcessor:
    """
    Main class that combines adaptive morphology and connected component analysis
    for comprehensive smart post-processing of prediction results.
    """
    
    def __init__(self,
                 adaptive_morphology_params: Optional[Dict[str, Any]] = None,
                 component_analysis_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the smart post-processor.
        
        Args:
            adaptive_morphology_params: Parameters for adaptive morphology processor
            component_analysis_params: Parameters for connected component analyzer
        """
        
        # Initialize adaptive morphology processor
        if adaptive_morphology_params is None:
            adaptive_morphology_params = {}
        self.morphology_processor = AdaptiveMorphologyProcessor(**adaptive_morphology_params)
        
        # Initialize connected component analyzer
        if component_analysis_params is None:
            component_analysis_params = {}
        self.component_analyzer = ConnectedComponentAnalyzer(**component_analysis_params)
        
    def process(self,
               predicted_mask: np.ndarray,
               confidence_map: Optional[np.ndarray] = None,
               enable_morphology: bool = True,
               enable_component_analysis: bool = True,
               enable_merging: bool = False,
               verbose: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply complete smart post-processing pipeline.
        
        Args:
            predicted_mask: Binary or probability mask from model prediction
            confidence_map: Optional confidence map
            enable_morphology: Whether to apply adaptive morphology
            enable_component_analysis: Whether to apply component analysis
            enable_merging: Whether to merge nearby components
            verbose: Whether to print processing steps
            
        Returns:
            Tuple of (processed_mask, processing_stats)
        """
        
        processing_stats = {
            'original_pixels': int(np.sum(predicted_mask > 0.5)),
            'steps_applied': [],
            'morphology_stats': None,
            'component_stats': None,
            'final_pixels': None,
            'processing_efficiency': None
        }
        
        # Convert to binary mask
        current_mask = predicted_mask > 0.5 if predicted_mask.dtype != bool else predicted_mask.copy()
        
        if verbose:
            print("🔄 Starting Smart Post-Processing Pipeline...")
            print(f"   Original mask pixels: {processing_stats['original_pixels']:,}")
        
        # Step 1: Adaptive Morphological Operations
        if enable_morphology:
            if verbose:
                print("   Step 1: Adaptive morphological operations...")
            current_mask = self.morphology_processor.adaptive_morphological_operations(
                current_mask, confidence_map
            )
            processing_stats['steps_applied'].append('adaptive_morphology')
            processing_stats['morphology_stats'] = {
                'pixels_after_morphology': int(np.sum(current_mask))
            }
            if verbose:
                print(f"      → Pixels after morphology: {processing_stats['morphology_stats']['pixels_after_morphology']:,}")
        
        # Step 2: Connected Component Analysis
        if enable_component_analysis:
            if verbose:
                print("   Step 2: Connected component analysis...")
            current_mask, component_stats = self.component_analyzer.analyze_connected_components(
                current_mask, confidence_map
            )
            processing_stats['steps_applied'].append('component_analysis')
            processing_stats['component_stats'] = component_stats
            if verbose:
                print(f"      → Kept {component_stats['kept_components']}/{component_stats['original_components']} components")
        
        # Step 3: Component Merging (optional)
        if enable_merging:
            if verbose:
                print("   Step 3: Merging nearby components...")
            current_mask = self.component_analyzer.merge_nearby_components(current_mask)
            processing_stats['steps_applied'].append('component_merging')
            if verbose:
                print(f"      → Components merged")
        
        # Final statistics
        processing_stats['final_pixels'] = int(np.sum(current_mask))
        processing_stats['processing_efficiency'] = (
            processing_stats['final_pixels'] / max(processing_stats['original_pixels'], 1)
        )
        
        if verbose:
            print(f"   Final mask pixels: {processing_stats['final_pixels']:,}")
            print(f"   Processing efficiency: {processing_stats['processing_efficiency']:.2%}")
            print("✅ Smart Post-Processing Complete!")
        
        return current_mask, processing_stats
    
    def visualize_complete_pipeline(self,
                                  original_mask: np.ndarray,
                                  confidence_map: np.ndarray,
                                  processed_mask: np.ndarray,
                                  processing_stats: Dict[str, Any],
                                  save_path: Optional[str] = None):
        """
        Visualize the complete smart post-processing pipeline results.
        
        Args:
            original_mask: Original predicted mask
            confidence_map: Confidence map
            processed_mask: Final processed mask
            processing_stats: Processing statistics
            save_path: Optional path to save visualization
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original mask
        axes[0, 0].imshow(original_mask.T[::-1], aspect='auto', cmap='gray', origin='lower')
        axes[0, 0].set_title('Original Predicted Mask')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Frequency')
        
        # Confidence map
        im1 = axes[0, 1].imshow(confidence_map.T[::-1], aspect='auto', cmap='viridis', origin='lower')
        axes[0, 1].set_title('Confidence Map')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Final processed mask
        axes[0, 2].imshow(processed_mask.T[::-1], aspect='auto', cmap='gray', origin='lower')
        axes[0, 2].set_title('Smart Post-Processed Mask')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Frequency')
        
        # Difference visualization
        difference = processed_mask.astype(int) - (original_mask > 0.5).astype(int)
        axes[1, 0].imshow(difference.T[::-1], aspect='auto', cmap='RdBu', origin='lower', vmin=-1, vmax=1)
        axes[1, 0].set_title('Changes Made\n(Blue=Removed, Red=Added)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Frequency')
        
        # Overlay comparison
        overlay = np.zeros((*original_mask.shape, 3))
        overlay[original_mask > 0.5] = [1, 0, 0]      # Original in red
        overlay[processed_mask] += [0, 1, 0]          # Processed in green
        # Overlap will be yellow (red + green)
        
        axes[1, 1].imshow(overlay.transpose(1, 0, 2)[::-1], aspect='auto', origin='lower')
        axes[1, 1].set_title('Overlay Comparison\n(Red=Original, Green=Processed, Yellow=Both)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Frequency')
        
        # Statistics
        stats_text = f"""Smart Post-Processing Results:

Pipeline Steps: {' → '.join(processing_stats['steps_applied'])}

Pixel Changes:
• Original: {processing_stats['original_pixels']:,} pixels
• Final: {processing_stats['final_pixels']:,} pixels  
• Net change: {processing_stats['final_pixels'] - processing_stats['original_pixels']:+,}
• Efficiency: {processing_stats['processing_efficiency']:.2%}

Component Analysis:"""
        
        if processing_stats['component_stats']:
            comp_stats = processing_stats['component_stats']
            stats_text += f"""
• Original components: {comp_stats['original_components']}
• Kept components: {comp_stats['kept_components']}
• Removal rate: {(1 - comp_stats['kept_components']/max(comp_stats['original_components'],1))*100:.1f}%"""
            
        if processing_stats['morphology_stats']:
            morph_stats = processing_stats['morphology_stats']
            stats_text += f"""

Morphology Impact:
• Pixels after morphology: {morph_stats['pixels_after_morphology']:,}"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Complete pipeline visualization saved to {save_path}")
            
        plt.show()


# Example usage functions
def example_smart_post_processing():
    """
    Example demonstrating how to use the smart post-processing pipeline.
    """
    print("Smart Post-Processing Example")
    print("="*40)
    
    # This is a template - replace with your actual prediction data
    # predicted_mask = your_model_prediction  # Shape: (time, frequency)
    # confidence_map = your_confidence_scores  # Shape: (time, frequency), values 0-1
    
    # Initialize the smart post-processor
    post_processor = SmartPostProcessor(
        adaptive_morphology_params={
            'high_conf_threshold': 0.8,
            'med_conf_threshold': 0.5,
            'min_object_size': 100
        },
        component_analysis_params={
            'min_component_size': 50,
            'max_component_size': 10000,
            'min_aspect_ratio': 0.2,
            'max_aspect_ratio': 8.0
        }
    )
    
    # Apply smart post-processing
    # processed_mask, stats = post_processor.process(
    #     predicted_mask=predicted_mask,
    #     confidence_map=confidence_map,
    #     enable_morphology=True,
    #     enable_component_analysis=True,
    #     verbose=True
    # )
    
    # Visualize results
    # post_processor.visualize_complete_pipeline(
    #     original_mask=predicted_mask,
    #     confidence_map=confidence_map,
    #     processed_mask=processed_mask,
    #     processing_stats=stats,
    #     save_path='smart_post_processing_results.png'
    # )
    
    print("Replace the commented code with your actual data!")


# Confidence threshold calibration utilities
def calibrate_confidence_thresholds(confidence_maps: list, 
                                   ground_truth_masks: list,
                                   num_thresholds: int = 20) -> Dict[str, Any]:
    """
    Calibrate confidence thresholds based on validation data.
    
    This function analyzes the relationship between model confidence and
    actual performance to determine optimal thresholds for adaptive processing.
    
    Args:
        confidence_maps: List of confidence maps from validation data
        ground_truth_masks: List of corresponding ground truth masks
        num_thresholds: Number of threshold points to evaluate
        
    Returns:
        Dictionary with calibration results and recommended thresholds
    """
    
    print("🔍 Calibrating confidence thresholds...")
    
    # Collect all confidence values and their correctness
    all_confidences = []
    all_correct = []
    
    for conf_map, gt_mask in zip(confidence_maps, ground_truth_masks):
        conf_flat = conf_map.flatten()
        pred_flat = (conf_map > 0.5).flatten()
        gt_flat = gt_mask.flatten()
        
        # Check which predictions are correct
        correct = (pred_flat == gt_flat)
        
        all_confidences.extend(conf_flat)
        all_correct.extend(correct)
    
    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)
    
    # Evaluate different thresholds
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    accuracy_by_threshold = []
    precision_by_threshold = []
    recall_by_threshold = []
    volume_by_threshold = []
    
    for threshold in thresholds:
        high_conf_mask = all_confidences >= threshold
        
        if np.sum(high_conf_mask) > 0:
            accuracy = np.mean(all_correct[high_conf_mask])
            
            # For precision/recall, need to look at actual predictions
            high_conf_preds = all_confidences[high_conf_mask] > 0.5
            high_conf_gt = all_correct[high_conf_mask]  # This is not right, need actual GT
            
            volume = np.sum(high_conf_mask) / len(all_confidences)
        else:
            accuracy = 0
            volume = 0
            
        accuracy_by_threshold.append(accuracy)
        volume_by_threshold.append(volume)
    
    # Find optimal thresholds
    accuracy_by_threshold = np.array(accuracy_by_threshold)
    volume_by_threshold = np.array(volume_by_threshold)
    
    # High confidence threshold: where accuracy > 90% and reasonable volume
    high_conf_candidates = thresholds[accuracy_by_threshold > 0.9]
    if len(high_conf_candidates) > 0:
        # Choose lowest threshold that gives >90% accuracy
        recommended_high_threshold = high_conf_candidates[0]
    else:
        recommended_high_threshold = 0.8  # fallback
    
    # Medium confidence threshold: balanced accuracy and volume
    balance_scores = accuracy_by_threshold * volume_by_threshold
    recommended_med_threshold = thresholds[np.argmax(balance_scores)]
    
    calibration_results = {
        'thresholds': thresholds,
        'accuracy_by_threshold': accuracy_by_threshold,
        'volume_by_threshold': volume_by_threshold,
        'recommended_high_threshold': float(recommended_high_threshold),
        'recommended_med_threshold': float(recommended_med_threshold),
        'calibration_summary': {
            'total_samples': len(all_confidences),
            'overall_accuracy': float(np.mean(all_correct)),
            'confidence_distribution': {
                'mean': float(np.mean(all_confidences)),
                'std': float(np.std(all_confidences)),
                'median': float(np.median(all_confidences))
            }
        }
    }
    
    print(f"✅ Calibration complete!")
    print(f"   Recommended high confidence threshold: {recommended_high_threshold:.3f}")
    print(f"   Recommended medium confidence threshold: {recommended_med_threshold:.3f}")
    
    return calibration_results


def visualize_confidence_calibration(calibration_results: Dict[str, Any],
                                   save_path: Optional[str] = None):
    """
    Visualize confidence threshold calibration results.
    
    Args:
        calibration_results: Results from calibrate_confidence_thresholds
        save_path: Optional path to save the visualization
    """
    
    thresholds = calibration_results['thresholds']
    accuracy = calibration_results['accuracy_by_threshold']
    volume = calibration_results['volume_by_threshold']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy vs threshold
    axes[0, 0].plot(thresholds, accuracy, 'b-', linewidth=2, label='Accuracy')
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% target')
    axes[0, 0].axvline(x=calibration_results['recommended_high_threshold'], 
                      color='g', linestyle='-', alpha=0.7, label='Recommended high')
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Confidence Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volume vs threshold
    axes[0, 1].plot(thresholds, volume, 'orange', linewidth=2, label='Volume')
    axes[0, 1].axvline(x=calibration_results['recommended_med_threshold'],
                      color='purple', linestyle='-', alpha=0.7, label='Recommended med')
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('Volume (fraction of data)')
    axes[0, 1].set_title('Data Volume vs Confidence Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy-Volume tradeoff
    axes[1, 0].plot(volume, accuracy, 'purple', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_xlabel('Volume (fraction of data)')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy-Volume Tradeoff')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    summary = calibration_results['calibration_summary']
    conf_dist = summary['confidence_distribution']
    
    summary_text = f"""Confidence Calibration Summary

Total Samples: {summary['total_samples']:,}
Overall Accuracy: {summary['overall_accuracy']:.3f}

Confidence Distribution:
• Mean: {conf_dist['mean']:.3f}
• Std: {conf_dist['std']:.3f}  
• Median: {conf_dist['median']:.3f}

Recommended Thresholds:
• High confidence: {calibration_results['recommended_high_threshold']:.3f}
• Medium confidence: {calibration_results['recommended_med_threshold']:.3f}

Usage in SmartPostProcessor:
```python
post_processor = SmartPostProcessor(
    adaptive_morphology_params={{
        'high_conf_threshold': {calibration_results['recommended_high_threshold']:.3f},
        'med_conf_threshold': {calibration_results['recommended_med_threshold']:.3f}
    }}
)
```"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confidence calibration visualization saved to {save_path}")
        
    plt.show()


def analyze_model_confidence_distribution(confidence_maps: list,
                                        predicted_masks: list = None) -> Dict[str, Any]:
    """
    Analyze the distribution of confidence values from your model.
    
    This helps understand how your model's confidence correlates with
    actual prediction quality and determines appropriate thresholds.
    
    Args:
        confidence_maps: List of confidence maps from your model
        predicted_masks: Optional list of binary predictions
        
    Returns:
        Analysis results with recommended threshold ranges
    """
    
    print("📊 Analyzing model confidence distribution...")
    
    # Flatten all confidence maps
    all_confidences = np.concatenate([conf.flatten() for conf in confidence_maps])
    
    # Basic statistics
    stats = {
        'count': len(all_confidences),
        'mean': np.mean(all_confidences),
        'std': np.std(all_confidences),
        'median': np.median(all_confidences),
        'min': np.min(all_confidences),
        'max': np.max(all_confidences),
        'percentiles': {
            '10th': np.percentile(all_confidences, 10),
            '25th': np.percentile(all_confidences, 25),
            '75th': np.percentile(all_confidences, 75),
            '90th': np.percentile(all_confidences, 90),
            '95th': np.percentile(all_confidences, 95),
            '99th': np.percentile(all_confidences, 99)
        }
    }
    
    # Suggest thresholds based on distribution
    if stats['std'] > 0.3:  # High variance
        suggested_high = stats['percentiles']['90th']
        suggested_med = stats['median']
    elif stats['mean'] > 0.7:  # Generally high confidence
        suggested_high = stats['percentiles']['75th'] 
        suggested_med = stats['percentiles']['50th']
    else:  # Generally low confidence
        suggested_high = stats['percentiles']['95th']
        suggested_med = stats['percentiles']['75th']
    
    analysis_results = {
        'statistics': stats,
        'suggested_thresholds': {
            'high_confidence': float(suggested_high),
            'med_confidence': float(suggested_med),
            'reasoning': f"Based on distribution shape (std={stats['std']:.3f}, mean={stats['mean']:.3f})"
        },
        'distribution_type': 'high_variance' if stats['std'] > 0.3 else 'low_variance'
    }
    
    print(f"   Distribution analysis complete!")
    print(f"   Mean confidence: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"   Suggested high threshold: {suggested_high:.3f}")
    print(f"   Suggested medium threshold: {suggested_med:.3f}")
    
    return analysis_results


def example_confidence_calibration():
    """
    Example of how to calibrate confidence thresholds for your specific model.
    """
    print("Confidence Threshold Calibration Example")
    print("="*50)
    
    # This is a template - replace with your actual validation data
    print("Step 1: Load your validation data")
    print("   confidence_maps = [your_model_confidence_outputs]")
    print("   ground_truth_masks = [your_validation_ground_truths]")
    
    print("\nStep 2: Analyze confidence distribution")
    print("   analysis = analyze_model_confidence_distribution(confidence_maps)")
    
    print("\nStep 3: Calibrate thresholds (if you have ground truth)")
    print("   calibration = calibrate_confidence_thresholds(confidence_maps, ground_truth_masks)")
    
    print("\nStep 4: Visualize results")
    print("   visualize_confidence_calibration(calibration)")
    
    print("\nStep 5: Use calibrated thresholds")
    print("   post_processor = SmartPostProcessor(")
    print("       adaptive_morphology_params={")
    print("           'high_conf_threshold': calibration['recommended_high_threshold'],")
    print("           'med_conf_threshold': calibration['recommended_med_threshold']")
    print("       }")
    print("   )")


if __name__ == "__main__":
    example_smart_post_processing()
    print("\n" + "="*60)
    example_confidence_calibration()
