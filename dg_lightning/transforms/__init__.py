
import typing

import torch
import torch.nn as nn

from dg_lightning.transforms.camelyon17_transforms import Camelyon17Transform
from dg_lightning.transforms.poverty_transforms import PovertyMapTransform


SupervisedLearningTransforms: typing.Dict[str, nn.Module] = {
    'camelyon17': Camelyon17Transform,
    'poverty': PovertyMapTransform,
}

ContrastiveLearningTransforms: typing.Dict[str, nn.Module] = {
    'camelyon17': None,  # TODO:
    'poverty': None,     # TODO:
}