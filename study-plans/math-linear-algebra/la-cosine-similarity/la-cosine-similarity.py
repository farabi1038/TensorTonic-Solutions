import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    
    x= np.asarray(a,dtype=float)
    y= np.asarray(b,dtype=float)

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)

    if x_norm<1e-10 or y_norm<1e-10:
        return 0
    return float(np.dot(x,y)/(x_norm*y_norm))    