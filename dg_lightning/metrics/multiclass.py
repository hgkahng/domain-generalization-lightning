
import torch
import torch.nn.functional as F

from torchmetrics.functional import accuracy as tm_accuracy
from torchmetrics.functional import f1_score as tm_f1
from torchmetrics.functional import auroc as tm_auroc


__all__ = [
    # 'cross_entropy_with_logits',
    # 'cross_entropy_with_probits',
    'cross_entropy_with_probs',
    'multiclass_accuracy',
    'multiclass_f1',
    'multiclass_auroc',
]


@torch.no_grad()
def cross_entropy_with_logits(preds: torch.FloatTensor,
                              target: torch.LongTensor,
                              **kwargs) -> torch.FloatTensor:
    """Add function docstring."""
    assert (preds.ndim == 2) and (target.ndim == 1) and (preds.shape[0] == target.shape[0])
    return F.cross_entropy(preds, target, reduction='mean')


@torch.no_grad()
def cross_entropy_with_probits(preds: torch.FloatTensor,
                               target: torch.LongTensor,
                               **kwargs) -> torch.FloatTensor:
    """Add function docstring."""
    assert (preds.ndim == 2) and (target.ndim == 1) and (preds.shape[0] == target.shape[0])
    raise NotImplementedError("Work in progress.")


@torch.no_grad()
def cross_entropy_with_probs(preds: torch.FloatTensor,
                             target: torch.LongTensor,
                             **kwargs) -> torch.FloatTensor:
    # TODO: sanity check
    assert (preds.ndim == 2) and (target.ndim == 1) and (preds.shape[0] == target.shape[0])
    probs_j = preds.gather(dim=1, index=target.view(-1, 1))
    return -1 * torch.log(probs_j + 1e-7).mean()


@torch.no_grad()
def multiclass_accuracy(preds: torch.FloatTensor,
                        target: torch.LongTensor,
                        average: str = 'macro',  # TODO: sanity check
                        **kwargs) -> torch.FloatTensor:
    """Measures multiclass accuracy."""
    assert (preds.ndim == 2) and (target.ndim == 1) and (preds.shape[0] == target.shape[0])
    return tm_accuracy(preds, target, task='multiclass', num_classes=preds.shape[1], average=average)


@torch.no_grad()
def multiclass_f1(preds: torch.FloatTensor,
                  target: torch.LongTensor,
                  average: str = 'macro',   # TODO: sanity check
                  **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 2) and (target.ndim == 1) and (preds.shape[0] == target.shape[0])
    return tm_f1(preds, target, task='multiclass', num_classes=preds.shape[1], average=average)


@torch.no_grad()
def multiclass_auroc(preds: torch.FloatTensor,
                     target: torch.LongTensor,
                     average: str = 'macro',  # TODO: sanity check
                     **kwargs) -> torch.FloatTensor:
    assert (preds.ndim == 2) and (target.ndim == 1) and (preds.shape[0] == target.shape[0])
    return tm_auroc(preds, target.long(), task='multiclass', num_classes=preds.shape[1], average=average)
