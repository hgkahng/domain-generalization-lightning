
import torch
import torch.nn as nn

from dg_lightning.networks.base import _ConvBackboneBase
from dg_lightning.networks.general import Flatten


class CNNForMNIST(_ConvBackboneBase):
    def __init__(self, in_channels: int = 3, **kwargs):
        super().__init__()

        self.in_channels: int = in_channels
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

    @property
    def out_channels(self) -> int:
        return 128

    @property
    def out_features(self) -> int:
        return self.out_channels
