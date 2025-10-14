import math
import numpy as np


def _is_nan_or_inf(x):
    try:
        return math.isnan(x) or math.isinf(x)
    except Exception:
        return False


def sanitize_value(v):
    # Numpy scalar types
    if isinstance(v, (np.floating, np.float32, np.float64)):
        f = float(v)
        if _is_nan_or_inf(f):
            return None
        return f
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, (np.bool_ , bool)):
        return bool(v)
    if isinstance(v, (np.ndarray,)):
        return sanitize_value(v.tolist())

    # Native Python floats
    if isinstance(v, float):
        if _is_nan_or_inf(v):
            return None
        return v

    # Primitive types that are JSON serializable
    if isinstance(v, (str, int)) or v is None:
        return v

    return str(v)


def sanitize(obj):
    """Recursively sanitize an object into JSON friendly types.

    - Convert numpy types to native Python types
    - Replace NaN/Inf with None
    - Convert nested numpy arrays to lists
    - Convert non-serializable objects to strings
    """
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    # For numpy arrays and scalars
    if isinstance(obj, (np.ndarray, np.generic)):
        try:
            py = obj.tolist()
            return sanitize(py)
        except Exception:
            return sanitize_value(obj)
    # Primitive or other
    return sanitize_value(obj)
