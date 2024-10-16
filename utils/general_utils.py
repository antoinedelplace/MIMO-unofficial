import time
from functools import wraps
import numpy as np
import torch
import resource

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result
    return wrapper

def try_wrapper(function, filename, log_path):
    try:
        function()
    except Exception as e:
        with open(log_path, 'a') as log_file:
            log_file.write(f"{filename}: {str(e)}\n")
        print(f"Error {filename}: {str(e)}\n")

def iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) of two binary masks.
    
    Parameters:
    mask1: numpy array of shape (H, W), binary mask (0 or 1)
    mask2: numpy array of shape (H, W), binary mask (0 or 1)
    
    Returns:
    iou: float, the Intersection over Union score
    """
    assert mask1.shape == mask2.shape, "Masks must have the same dimensions"

    # Use bitwise operations to calculate intersection and union
    intersection = torch.sum(mask1 & mask2)
    union = torch.sum(mask1 | mask2)

    if union == 0:
        return 0.0  # Avoid division by zero
    
    return intersection / union

def set_memory_limit(max_memory_gb):
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

def argmedian(x, axis=None):
    if axis is None:
        return np.argpartition(x, len(x) // 2)[len(x) // 2]
    else:
        # Compute argmedian along specified axis
        return np.apply_along_axis(
            lambda x: np.argpartition(x, len(x) // 2)[len(x) // 2],
            axis=axis, arr=x
        )