import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    pass
    model.train()
    totalL = 0.0
    n_batches =0

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output,targets)
        loss.backward()
        optimizer.step()

        totalL+=loss.item()
        n_batches+=1
    return totalL/n_batches    
