
import torch
import torch.nn as nn


class _BackboneBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: str, key: str) -> None:
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt[key])
    
    def freeze_weights(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

class _ConvBackboneBase(_BackboneBase):
    def __init__(self):
        super().__init__()

    @property
    def out_features(self) -> int:
        raise NotImplementedError


class _BertBackboneBase(_BackboneBase):
    def __init__(self):
        super().__init__()
    
    @property
    def out_features(self) -> int:
        raise NotImplementedError
