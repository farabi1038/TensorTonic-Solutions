import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features,hidden_size)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,out_features)

    def forward(self, x):
        x=self.linear1(x)
        x=self.ReLU(x)
        x=self.linear2(x)

        return x
        
