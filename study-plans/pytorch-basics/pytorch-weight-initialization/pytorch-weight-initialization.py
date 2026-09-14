import torch

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    if method =='xavier_uniform':
        limit =math.sqrt(6/(fan_out + fan_in))
        return torch.empty(fan_out,fan_in).uniform_(-limit, limit) 
    elif method =='he_normal':
        limit = math.sqrt(2/( fan_in))
        return torch.empty(fan_out,fan_in).normal_(0.0, limit)
    elif method =='he_uniform':
        limit =math.sqrt(6/( fan_in))
        return torch.empty(fan_out,fan_in).uniform_(-limit, limit)
    elif method =='xavier_normal':
        limit =math.sqrt(2/(fan_out + fan_in))
        return torch.empty(fan_out,fan_in).normal_(0.0, limit)
    else:
        raise ValueError(f"Unknown method :{method}")
