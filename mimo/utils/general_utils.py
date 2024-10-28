import sys
sys.path.append(".")

import time
from functools import wraps
import numpy as np
import torch
import resource
import argparse
import inspect
import traceback
import GPUtil

from mimo.utils.torch_utils import free_gpu_memory

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
        return function()
    except Exception as e:
        free_gpu_memory()
        
        error_trace = traceback.format_exc()

        with open(log_path, 'a') as log_file:
            log_file.write(f"{filename}: {error_trace}\n")
        print(f"Error in {filename}:\n{error_trace}")

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

def parse_args(main_function):
    parser = argparse.ArgumentParser()

    used_short_versions = set("h")

    signature = inspect.signature(main_function)
    for param_name, param in signature.parameters.items():
        short_version = param_name[0]
        if short_version in used_short_versions or not short_version.isalpha():
            for char in param_name[1:]:
                short_version = char
                if char.isalpha() and short_version not in used_short_versions:
                    break
            else:
                short_version = None
        
        if short_version:
            used_short_versions.add(short_version)
            param_call = (f'-{short_version}', f'--{param_name}')
        else:
            param_call = (f'--{param_name}',)

        if param.default is not inspect.Parameter.empty:
            param_type = type(param.default)
            parser.add_argument(*param_call, type=param_type, default=param.default,
                                help=f"Automatically detected argument: {param_name}, default: {param.default}")
        else:
            parser.add_argument(*param_call, required=True,
                                help=f"Required argument: {param_name}")

    args = parser.parse_args()

    return args

def get_gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found.")
        return []
    
    gpu_memory_info = []
    for gpu in gpus:
        memory_used_gb = gpu.memoryUsed / 1024  # Convert MB to GB
        memory_total_gb = gpu.memoryTotal / 1024  # Convert MB to GB
        gpu_memory_info.append({
            'id': gpu.id,
            'name': gpu.name,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb
        })

        print(f"GPU {gpu.id} ({gpu.name}): {memory_used_gb} GB / {memory_total_gb} GB")
    
    return gpu_memory_info