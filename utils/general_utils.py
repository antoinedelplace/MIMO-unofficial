import time
from functools import wraps

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