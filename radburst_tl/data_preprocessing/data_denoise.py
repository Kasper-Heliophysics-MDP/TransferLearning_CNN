import numpy as np
from scipy.ndimage import binary_opening
from skimage.morphology import binary_opening, rectangle

def remove_vertical_noise(arr, num_std=5, dist=10):
    """
    Remove columns with high variance (vertical noise).

    Args:
        arr (np.ndarray): Array to process.
        num_std (int): Number of standard deviations above the mean to use as threshold for vertical noise.
        dist (int): Distance away from vertical noise column to get replacement column.

    Returns:
        np.ndarray: Processed array with vertical noise removed.
    """
    # Calculate variance of each column
    vars = np.var(arr, axis=0, ddof=0)

    # Calculate the mean and standard deviation of the variances
    mean_var = np.mean(vars)
    std_var = np.std(vars)

    # Calculate the threshold for detecting columns with unwanted vertical noise
    var_threshold = mean_var + num_std * std_var

    # Find indices of columns with variance greater than threshold
    high_var_cols = np.where(vars > var_threshold)[0]

    # Replace high variance columns with other columns some distance away
    arr_cleaned = arr.copy()
    for i in high_var_cols:
        col_to_replace = (i - dist) if (i >= dist) else (i + dist)
        arr_cleaned[:, i] = arr[:, col_to_replace]
        
    return arr_cleaned

def cusum_slope(data, window_size=5):
    # calculate the moving average
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return moving_avg

def detect_srb(data, percentile=75):
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
