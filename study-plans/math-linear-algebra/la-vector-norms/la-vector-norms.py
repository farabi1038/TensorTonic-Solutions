import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    x= np.asarray(v,dtype=np.float64)
    l1 =np.linalg.norm(v,ord=1)
    l2 =np.linalg.norm(v,ord=2)
    l3 =np.linalg.norm(v,ord=np.inf)

    return np.array([l1,l2,l3])
    
    