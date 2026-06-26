import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred =torch.tensor(pred,dtype=torch.float32)
    
    if method =="mse":
        target = torch.tensor(target,dtype=torch.float32)
        return ((pred-target)**2).mean().item()
        
    elif method =="cross_entropy" :
        target =torch.tensor(target,dtype=torch.long)
        max_logit = pred.max(dim = 1,keepdim= True).values
        shift = pred -max_logit
        log_sum = shift.exp().sum(dim=1).log() +max_logit.squeeze(1)
        correct_logit = pred[torch.arange(pred.shape[0]),target]
        return (log_sum-correct_logit).mean().item()
        
    elif method =="huber" :
        target =torch.tensor(target,dtype=torch.float32)
        diff =(pred-target).abs()
        loss = torch.where(diff<=delta,0.5 *diff **2, delta * (diff -0.5 *delta ))
        return loss.mean().item()
        
    
