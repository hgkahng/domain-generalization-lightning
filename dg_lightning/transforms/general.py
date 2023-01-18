
import typing

import torch
import torch.nn as nn


class IdentityTransform(nn.Module):
    """
    This class exists for consistency. Please note with caution that
    Input transformations of text data is handled within each `torch.utils.data.Dataset`.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x



class CutoutOnFloat(nn.Module):
    def __init__(self, div_factor: float = 2., **kwargs):
        super().__init__()
        self.div_factor: float = div_factor

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        assert (x.ndim == 4) and (x.shape[2] == x.shape[3])
        _, _, _, width = x.shape
        cutout_width = self._sample_uniform_single(0., int(width / self.div_factor))
        cutout_center_x = self._sample_uniform_single(0., width)
        cutout_center_y = self._sample_uniform_single(0., width)

        x0 = int(max(0, cutout_center_x - cutout_width / 2))
        y0 = int(max(0, cutout_center_y - cutout_width / 2))
        x1 = int(min(width, cutout_center_x + cutout_width / 2))
        y1 = int(min(width, cutout_center_y + cutout_width / 2))

        # fill with zeros
        x[:, :, x0:x1, y0:y1] = 0.
        return x

    @staticmethod
    def _sample_uniform_single(a: float, b: float) -> float:
        return torch.empty(1).uniform_(a, b).item()

    @staticmethod
    def _sample_uniform_(*size, min_: float, max_: float) -> torch.FloatTensor:
        return torch.zeros(*size).uniform_(min_, max_).flatten()
