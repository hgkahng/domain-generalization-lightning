
import typing
import torch
import torch.nn.functional as F

from torchmetrics.functional import accuracy as tm_accuracy
from torchmetrics.functional import f1_score as tm_f1
from torchmetrics.functional import auroc as tm_auroc


__all__ = [
    # 'binary_cross_entropy_with_logits',
    # 'binary_cross_entropy_with_probits',
    'binary_cross_entropy_with_probs',
    'binary_accuracy',
    'binary_f1',
    'binary_auroc'
]


@torch.no_grad()
def binary_cross_entropy_with_logits(preds: torch.FloatTensor,
                                     target: typing.Union[torch.FloatTensor, torch.LongTensor],
                                     **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 1) and (target.ndim == 1) and (preds.shape == target.shape)
    return F.binary_cross_entropy_with_logits(preds, target.float(), reduction='mean')


@torch.no_grad()
def binary_cross_entropy_with_probits(preds: torch.FloatTensor,
                                      target: typing.Union[torch.FloatTensor, torch.LongTensor],
                                      **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 1) and (target.ndim == 1) and (preds.shape == target.shape)
    probs = torch.distributions.Normal(loc=0, scale=1.).cdf(preds)
    return F.binary_cross_entropy(probs, target.float(), reduction='mean')


@torch.no_grad()
def binary_cross_entropy_with_probs(preds: torch.FloatTensor,
                                    target: typing.Union[torch.FloatTensor, torch.LongTensor],
                                    **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 1) and (target.ndim == 1) and (preds.shape == target.shape)
    return F.binary_cross_entropy(preds, target.float(), reduction='mean')


@torch.no_grad()
def binary_accuracy(preds: torch.FloatTensor,
                    target: typing.Union[torch.FloatTensor, torch.LongTensor],
                    threshold: float = 0.5,
                    **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 1) and (target.ndim == 1) and (preds.shape == target.shape)
    return tm_accuracy(preds, target.long(), task='binary', threshold=threshold)


@torch.no_grad()
def binary_f1(preds: torch.FloatTensor,
              target: typing.Union[torch.FloatTensor, torch.LongTensor],
              threshold: float = 0.5,
              **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 1) and (target.ndim == 1) and (preds.shape == target.shape)
    return tm_f1(preds, target.long(), task='binary', threshold=threshold)


@torch.no_grad()
def binary_auroc(preds: torch.FloatTensor,
                 target: typing.Union[torch.FloatTensor, torch.LongTensor],
                 **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 1) and (target.ndim == 1) and (preds.shape == target.shape)
    return tm_auroc(preds, target.long(), task='binary', threshold=None)
