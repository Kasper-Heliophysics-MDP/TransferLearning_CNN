import numpy as np
from scipy.ndimage import binary_opening
from skimage.morphology import binary_opening, rectangle

def cusum_slope(data, window_size=5):
    # 计算移动平均
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return moving_avg

def detect_srb(data, percentile=75):
    # 计算CUSUM-slope
    slope = cusum_slope(data)
    
    # 计算阈值
    threshold = np.percentile(slope, percentile)
    
    # 生成掩码
    mask = slope > threshold
    
    # 确保结构元素和数据都是二维的
    structuring_element = rectangle(1, 4)
    cleaned_mask = binary_opening(mask[np.newaxis, :], footprint=structuring_element)
    
    return cleaned_mask[0]
