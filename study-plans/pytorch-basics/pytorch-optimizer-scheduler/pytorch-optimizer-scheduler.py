import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    losses =[]
    lr =[]

    for epoch in range(num_epochs):
        model.train()
        c_lr = optimizer.param_groups[0]["lr"]
        lr.append(c_lr)
        train_loss =0.0
        n_batches =0

        for x,y in dataloader:
            optimizer.zero_grad()
            output =model(x)
            loss =criterion(output,y)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            n_batches+=1
        losses.append(train_loss/n_batches)
        scheduler.step()
    return {"losses":losses,"lrs":lr}    