import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    tensor_x = torch.tensor(x,dtype=torch.float32)
    if method=="relu":
        return torch.clamp(tensor_x,min=0).tolist()
    elif method=="sigmoid":
        return (1/(1+torch.exp(-tensor_x))).tolist()
    elif method=="tanh":
        return ((torch.exp(tensor_x) -torch.exp(-tensor_x))/(torch.exp(tensor_x) +torch.exp(-tensor_x))).tolist()
    elif method=="leaky_relu":
        return torch.where(tensor_x>0,tensor_x, 0.01*tensor_x).tolist()