
import torch

from torchmetrics.functional import mean_squared_error as tm_mse
from torchmetrics.functional import pearson_corrcoef as tm_pearson_corrcoef


__all__ = [
    'mean_squared_error',
    'pearson_correlation',
]


@torch.no_grad()
def mean_squared_error(preds: torch.FloatTensor,
                       target: torch.FloatTensor,
                       **kwargs) -> torch.FloatTensor:
    """Measures mean squared error."""
    target = target.squeeze().float(); assert target.ndim == 1, f"{target.shape}"
    preds = preds.squeeze().float(); assert preds.ndim == 1, f"{preds.shape}"
    return tm_mse(preds, target)


@torch.no_grad()
def pearson_correlation(preds: torch.FloatTensor,
                        target: torch.FloatTensor,
                        **kwargs) -> torch.FloatTensor:
    """Measures pearson correlation."""
    target = target.squeeze().float(); assert target.ndim == 1, f"{target.shape}"
    preds = preds.squeeze().float(); assert preds.ndim == 1, f"{preds.shape}"
    return tm_pearson_corrcoef(preds, target)
