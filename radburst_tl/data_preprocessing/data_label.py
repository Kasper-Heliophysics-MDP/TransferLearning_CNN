from datetime import datetime
import cv2
import pandas as pd
import numpy as np
from scipy.ndimage import binary_opening
from skimage.morphology import binary_opening, rectangle
from skimage.morphology import binary_erosion, binary_dilation, disk

def cusum_slope(data, window_size=5):
    # calculate the moving average
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return moving_avg

def detect_srb(data, percentile=40):
    # calculate the CUSUM-slope
    slope = cusum_slope(data)
    
    # calculate the threshold
    threshold = np.percentile(slope, percentile)
    
    # generate the mask
    mask = slope > threshold
    
    # ensure the structuring element and data are both two-dimensional
    structuring_element = rectangle(1, 4)
    cleaned_mask = binary_opening(mask[np.newaxis, :], footprint=structuring_element)
    
    return cleaned_mask[0]

# //////////////////////////////////////////////////////////////// #

def time_to_column_indices(time_array, start_time_str, end_time_str):
    """
    Convert start and end times in hh:mm:ss format to column indices
    by matching times of day only (no dates).
    
    Args:
        time_array (pd.Series or array-like): Array of times in hh:mm:ss format.
        start_time_str (str): Start time in hh:mm:ss format.
        end_time_str (str): End time in hh:mm:ss format.
        
    Returns:
        tuple: The closest start and end column indices.
    """
    # 1. Parse the time_array as times (ignoring any date).
    #    If time_array is already a string like "18:03:00", we can do:
    time_series = pd.to_datetime(time_array, format="%H:%M:%S.%f").dt.time
    # print("time_series:", time_series)
    # 2. Parse start and end times as datetime.time objects
    start_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
    end_time = datetime.strptime(end_time_str, "%H:%M:%S").time()
    
    # Helper function: convert a datetime.time object to "seconds since midnight"
    def time_to_seconds(t):
        return t.hour * 3600 + t.minute * 60 + t.second
    
    # 3. Convert your series and start/end times to integer seconds
    time_in_seconds = np.array([time_to_seconds(t) for t in time_series])
    start_in_seconds = time_to_seconds(start_time)
    end_in_seconds = time_to_seconds(end_time)
    # print("start_in_seconds:", start_in_seconds, "end_in_seconds:", end_in_seconds)
    # print("time_in_seconds:", time_in_seconds)
    
    # 4. Find indices with minimum absolute difference
    start_index = np.abs(time_in_seconds - start_in_seconds).argmin()
    end_index = np.abs(time_in_seconds - end_in_seconds).argmin()
    
    return start_index, end_index

# IMPORTANT: CHANGE THE THRESHOLD HERE!!!
def create_srb_mask(data, start_index, end_index, pct_threshold=37):
    """
    Create a binary mask for SRB within a specified time range based on a percentile threshold.

    Args:
        data (np.ndarray or pd.DataFrame): Spectral data array.
        start_index (int): Start index for the time range.
        end_index (int): End index for the time range.
        pct_threshold (int): Percentile threshold for setting mask values to 1.

    Returns:
        np.ndarray or pd.DataFrame: Binary mask where values above threshold in the specified range are 1, others are 0.
    """
    # If data is a DataFrame, use .iloc for positional indexing
    if isinstance(data, pd.DataFrame):
        # Extract the relevant time range using positional indexing
        data_range = data.iloc[start_index:end_index, :]
        # Calculate the threshold based on the specified percentile
        threshold = np.percentile(data_range.values, pct_threshold)
        # Create the mask based on the threshold
        mask = data_range.values > threshold
        # Initialize a full mask with zeros with the same shape as data
        full_mask = np.zeros(data.shape, dtype=bool)
        # Place the calculated mask into the full mask at the specified time range
        full_mask[start_index:end_index, :] = mask
        # Optionally, return as a DataFrame with the same index and columns
        return pd.DataFrame(full_mask, index=data.index, columns=data.columns)
    
    else:
        # Assume data is a NumPy array
        data_range = data[start_index:end_index, :]
        threshold = np.percentile(data_range, pct_threshold)
        mask = data_range > threshold
        full_mask = np.zeros_like(data, dtype=bool)
        full_mask[start_index:end_index, :] = mask
        return full_mask


def apply_morphological_operations(mask, 
                                   erosion_radius=1, 
                                   dilation_radius=1, 
                                   operation_sequence=None):
    """
    Apply a sequence of morphological operations to a binary mask.

    Args:
        mask (np.ndarray): Input binary mask to be processed.
        erosion_radius (int): Radius of the structuring element for erosion.
        dilation_radius (int): Radius of the structuring element for dilation.
        operation_sequence (list of str): Sequence of operations to apply. 
                                          Options are 'dilate', 'erode', 'open', 'close'.
                                          Default is ['dilate', 'erode'].

    Returns:
        np.ndarray: Processed binary mask.
    """
    if operation_sequence is None:
        operation_sequence = ['dilate', 'erode']

    structuring_element_erosion = disk(erosion_radius)
    structuring_element_dilation = disk(dilation_radius)

    processed_mask = mask.copy()

    for operation in operation_sequence:
        if operation == 'dilate':
            processed_mask = binary_dilation(processed_mask, structuring_element_dilation)
        elif operation == 'erode':
            processed_mask = binary_erosion(processed_mask, structuring_element_erosion)
        elif operation == 'open':
            processed_mask = binary_erosion(processed_mask, structuring_element_erosion)
            processed_mask = binary_dilation(processed_mask, structuring_element_dilation)
        elif operation == 'close':
            processed_mask = binary_dilation(processed_mask, structuring_element_dilation)
            processed_mask = binary_erosion(processed_mask, structuring_element_erosion)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    return processed_mask

def blur(arr, blur_filter_shape=(61, 11), use_gaussian=True):
    """Blur array using a specified kernel to enhance potential bursts.

    Args:
        arr (np.ndarray): Input array to blur.
        blur_filter_shape (tuple): Kernel size (width, height).
        use_gaussian (bool): Whether to use Gaussian blur. If False, use a simple averaging filter.

    Returns:
        np.ndarray: Blurred array.
    """
    # If arr is boolean, convert it to uint8 (0 and 255)
    if arr.dtype == np.bool_:
        arr = np.uint8(arr) * 255

    if use_gaussian:
        return cv2.GaussianBlur(arr, blur_filter_shape, 0)
    else:
        kernel = np.ones(blur_filter_shape, np.float32) / (blur_filter_shape[0] * blur_filter_shape[1])
        return cv2.filter2D(arr, -1, kernel)
    
def apply_rolling_median_filter(mask, window_size=3):
    """
    Apply a rolling median filter across time to the SRB mask.

    Args:
        mask (np.ndarray): Input binary mask to be processed.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: Mask after applying the rolling median filter.
    """
    # Convert the mask to a DataFrame for easier rolling operation
    mask_df = pd.DataFrame(mask)

    # Apply the rolling median filter along the time axis (axis=0)
    filtered_mask_df = mask_df.rolling(window=window_size, axis=0, center=True).median()

    # Fill NaN values that result from the rolling operation
    filtered_mask_df = filtered_mask_df.fillna(0)

    # Convert back to a NumPy array
    filtered_mask = filtered_mask_df.to_numpy()

    return filtered_mask.astype(bool)

def intersect_srb_masks(*masks):
    """
    Compute the intersection of multiple SRB masks.

    Args:
        *masks (np.ndarray): Variable number of binary masks to intersect.

    Returns:
        np.ndarray: A binary mask representing the intersection of all input masks.
    """
    if not masks:
        raise ValueError("At least one mask must be provided.")

    # Initialize the intersection mask with the first mask
    intersection_mask = masks[0].copy()

    # Perform element-wise logical AND with all subsequent masks
    for mask in masks[1:]:
        intersection_mask = np.logical_and(intersection_mask, mask)

    return intersection_mask