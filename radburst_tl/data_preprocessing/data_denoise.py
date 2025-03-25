import cv2
import pandas as pd
import numpy as np
from scipy.ndimage import binary_opening
from skimage.morphology import binary_opening, rectangle

# Actually horizontal noise in spectrogram!!!!
def remove_vertical_noise(arr, num_mean=0.5, num_std=0.01, dist=50):
    """
    Remove columns with high variance (vertical noise).

    Args:
        arr (np.ndarray): Array to process.
        num_std (int): Number of standard deviations above the mean to use as threshold for vertical noise.
        dist (int): Distance away from vertical noise column to get replacement column.

    Returns:
        np.ndarray: Processed array with vertical noise removed.
    """

    # Convert DataFrame to numpy array if necessary
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()

    # Calculate variance of each column
    vars = np.var(arr, axis=0, ddof=0)

    # Calculate the mean and standard deviation of the variances
    mean_var = np.mean(vars)
    std_var = np.std(vars)

    # Calculate the threshold for detecting columns with unwanted vertical noise
    var_threshold = num_mean * mean_var + num_std * std_var

    # Find indices of columns with variance greater than threshold
    low_var_cols = np.where(vars < var_threshold)[0]
    print("Low variance rows:", low_var_cols)

    # Replace high variance columns with other columns some distance away
    arr_cleaned = arr.copy()
    for i in low_var_cols:
        col_to_replace = (i + dist) if (i <= dist) else (i - dist)
        arr_cleaned[:, i] = arr[:, col_to_replace]
        
    return arr_cleaned

# Actually vertical noise in spectrogram!!!!
def remove_horizontal_noise(arr, num_std=0.1, dist=120):
    """
    Remove rows with high variance (horizontal noise). Because the data is vertical-to-horizontal reversed, the horizontal noise is actually vertical noise.

    Args:
        arr (np.ndarray): Array to process.
        num_std (int): Number of standard deviations above the mean to use as threshold for horizontal noise.
        dist (int): Distance away from horizontal noise row to get replacement row.

    Returns:
        np.ndarray: Processed array with horizontal noise removed.
    """

    # Convert DataFrame to numpy array if necessary
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()

    # Calculate variance of each row
    vars = np.var(arr, axis=1, ddof=0)

    # Calculate the mean and standard deviation of the variances
    mean_var = np.mean(vars)
    std_var = np.std(vars)

    # Calculate the threshold for detecting rows with unwanted horizontal noise
    var_threshold = mean_var + num_std * std_var

    # Find indices of rows with variance greater than threshold
    high_var_rows = np.where(vars < var_threshold)[0]
    print("High variance columns:", high_var_rows)

    # Replace high variance rows with other rows some distance away
    arr_cleaned = arr.copy()
    for i in high_var_rows:
        row_to_replace = (i - dist) if (i >= dist) else (i + dist)
        arr_cleaned[i, :] = arr[row_to_replace, :]
        
    return arr_cleaned

