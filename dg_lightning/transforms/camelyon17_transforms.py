
import typing

import torch
import torch.nn as nn
import torchvision.transforms as T

from dg_lightning.transforms.general import IdentityTransform
from dg_lightning.transforms.general import CutoutOnFloat


class Camelyon17Transform(nn.Module):
    
    means: typing.Dict[int, typing.Tuple[float]] = {
        0: (0.735, 0.604, 0.701),
        1: (0.610, 0.461, 0.593),
        2: (0.673, 0.483, 0.739),
        3: (0.686, 0.490, 0.620),
        4: (0.800, 0.672, 0.820),
    }
    
    stds: typing.Dict[int, typing.Tuple[float]] = {
        0: (0.184, 0.220, 0.170),
        1: (0.186, 0.219, 0.174),
        2: (0.208, 0.235, 0.133),
        3: (0.202, 0.222, 0.173),
        4: (0.130, 0.159, 0.104),
    }
    
    size: typing.Tuple[int] = (96, 96)
    
    def __init__(self,
                 mean: typing.Tuple[float] = (0.720, 0.560, 0.715),
                 std: typing.Tuple[float] = (0.190, 0.224, 0.170),
                 augmentation: bool = False,
                 randaugment: bool = False,
                 **kwargs):
        super(Camelyon17Transform, self).__init__()
        
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.randaugment = randaugment

        transform = [
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std, inplace=False),
        ]
        if self.augmentation:
            transform = [
                T.RandomHorizontalFlip(0.5),
                T.RandAugment(num_ops=2, magnitude=9, ) if self.randaugment else IdentityTransform(),
                *transform,
                CutoutOnFloat()
            ]

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.ByteTensor) -> torch.FloatTensor:
        return self.transform(x)
