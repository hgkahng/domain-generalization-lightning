
import typing

import torch
import torch.nn as nn

from dg_lightning.transforms.camelyon17_transforms import Camelyon17Transform


SupervisedLearningTransforms: typing.Dict[str, nn.Module] = {
    'camelyon17': Camelyon17Transform,
    'povertymap': None,
}

ContrastiveLearningTransforms: typing.Dict[str, nn.Module] = {
    'camelyon17': None,
    'povertymap': None,
}