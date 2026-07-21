import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    max_val = torch.max(logits,dim=1,keepdim = True).values
    exps = torch.exp(logits-max_val)
    return exps/exps.sum(dim=1,keepdim=True)
