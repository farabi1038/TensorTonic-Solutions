import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    W = torch.tensor(w_init,dtype=torch.float32, requires_grad = True)
    last_av_grad = None

    for i, (x,y) in enumerate(micro_batches):
        x_1= torch.tensor(x,dtype= torch.float32)
        y_1= torch.tensor(y,dtype= torch.float32)
        pred = torch.dot(W,x_1)
        loss = (pred-y_1)**2
        loss.backward()

        if (i+1)%accum_steps==0:
            last_av_grad = W.grad.clone()/accum_steps
            with torch.no_grad():
                W-= lr * last_av_grad
            W.grad.zero_()

    return W.detach().tolist(),last_av_grad.tolist()        
