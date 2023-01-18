
import typing
import torch


from torchmetrics.functional import accuracy as tm_accuracy
from torchmetrics.functional import recall as tm_recall
from torchmetrics.functional import precision as tm_precision
from torchmetrics.functional import f1_score as tm_f1
from torchmetrics.functional import auroc as tm_auroc


__all__ = [
    'multilabel_accuracy',
    'multilabel_recall',
    'multilabel_precision',
    'multilabel_f1',
    'multilabel_auroc',
]


@torch.no_grad()
def multilabel_accuracy(preds: torch.FloatTensor,
                        target: typing.Union[torch.FloatTensor, torch.LongTensor],
                        threshold: float = .5,
                        average: str = 'macro',
                        **kwargs) -> torch.FloatTensor:
    """Measures multilabel accuracy."""
    assert (preds.ndim == 2) and (target.ndim == 2) and (preds.shape == target.shape)
    return tm_accuracy(preds, target.long(), task='multilabel', num_labels=preds.shape[1], threshold=threshold, average=average)


@torch.no_grad()
def multilabel_recall(preds: torch.FloatTensor,
                      target: typing.Union[torch.FloatTensor, torch.LongTensor],
                      threshold: float = .5,
                      average: str = 'macro',
                      **kwargs) -> torch.FloatTensor:
    """Measures multilabel recall (or sensitivity)."""
    assert (preds.ndim == 2) and (target.ndim == 2) and (preds.shape == target.shape)
    return tm_recall(preds, target.long(), task='multilabel', num_labels=preds.shape[1], threshold=threshold, average=average)


@torch.no_grad()
def multilabel_precision(preds: torch.FloatTensor,
                         target: typing.Union[torch.FloatTensor, torch.LongTensor],
                         threshold: float = .5,
                         average: str = 'macro',
                         **kwargs) -> torch.FloatTensor:
    """Measures multilabel precision."""
    assert (preds.ndim == 2) and (target.ndim == 2) and (preds.shape == target.shape)
    return tm_precision(preds, target.long(), task='multilabel', num_labels=preds.shape[1], threshold=threshold, average=average)


@torch.no_grad()
def multilabel_f1(preds: torch.FloatTensor,
                  target: typing.Union[torch.FloatTensor, torch.LongTensor],
                  threshold: float = .5,
                  average: str = 'macro',
                  **kwargs) -> torch.FloatTensor:
    """Measures multilabel f1."""                  
    assert (preds.ndim == 2) and (target.ndim == 2) and (preds.shape == target.shape)
    return tm_f1(preds, target.long(), task='multilabel', num_labels=preds.shape[1], threshold=threshold, average=average)


@torch.no_grad()
def multilabel_auroc(preds: torch.FloatTensor,
                     target: typing.Union[torch.FloatTensor, torch.LongTensor],
                     average: str = 'macro',
                     **kwargs) -> torch.FloatTensor:
    """Measures multilabel auroc."""
    assert (preds.ndim == 2) and (target.ndim == 2) and (preds.shape == target.shape)
    return tm_auroc(preds, target.long(), task='multilabel', num_labels=preds.shape[1], threshold=None, average=average)
