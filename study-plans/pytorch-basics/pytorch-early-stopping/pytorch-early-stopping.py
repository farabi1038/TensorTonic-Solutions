import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    stopped_epoch = 0
    p_count =0
    train_losses=[]
    val_losses =[]
    best_val_loss = float("inf")
    for epoch in range(max_epochs):
        n_train = 0.0
        train_loss =0.0
        model.train()
        for x,y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss =criterion(output,y)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            n_train+=1
        train_losses.append(train_loss/n_train)  


        n_val = 0.0
        val_loss =0.0
        model.eval()
        with torch.no_grad():
            for x,y in val_loader:
                output = model(x)
                loss =criterion(output,y)
                val_loss+=loss.item()
                n_val+=1
        val_losses.append(val_loss/n_val)

        if val_losses[-1]<best_val_loss:
            best_val_loss =val_losses[-1]
            p_count=0
            
        else:
            p_count+=1
            if p_count>=patience:
                return {"train_losses": train_losses, "val_losses": val_losses, "stopped_epoch": epoch+1}
        
        
    return {"train_losses": train_losses, "val_losses": val_losses, "stopped_epoch": max_epochs}