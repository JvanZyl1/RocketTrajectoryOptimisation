import numpy as np

def convert_ndarray(obj):
    """
    Recursively convert NumPy ndarrays to lists within a dictionary.
    """
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray(element) for element in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj