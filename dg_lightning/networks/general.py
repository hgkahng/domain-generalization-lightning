
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)
