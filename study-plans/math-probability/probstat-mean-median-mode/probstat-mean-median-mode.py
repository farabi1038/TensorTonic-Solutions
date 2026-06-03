import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    mean = float(np.mean(x))
    median = float(np.median(x))
    mode = float(max(sorted(set(x)),key=list(x).count))
    return {"mean":mean,"median":median,"mode":mode}
    