import time
from functools import wraps
import numpy as np

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

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0  # Avoid division by zero
    
    iou = intersection / union
    return iou