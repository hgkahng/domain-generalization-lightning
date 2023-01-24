
import typing

import torch
import torch.nn.functional as F

from functools import partial

from dg_lightning.metrics.binary import *
from dg_lightning.metrics.multilabel import *
from dg_lightning.metrics.multiclass import *
from dg_lightning.metrics.regression import *
from dg_lightning.metrics.wrappers import (worst_group_metric_wrapper,
                                           macro_average_metric_wrapper)

class MetricEvaluator(object):
    
    _supported_data: typing.List[str] = [
        'rmnist',
        'cmnist',
        'pacs',
        'vlcs',
        'celeba',
        'camelyon17',
        'poverty',
        'iwildcam',
        'rxrx1',
        'fmow',
        'civilcomments',
    ]
    
    _ignore_group_eval: typing.List[str] = [
        'rmnist',
        'cmnist',
        'pacs',
        'vlcs',
        'camelyon17',
        'poverty',
        'iwildcam',
        'rxrx1'
        'fmow',
    ]
    
    def __init__(self,
                 data: str,
                 exclude_metrics: typing.Optional[typing.List[str]] = [],
                 group_worst: typing.Optional[bool] = True,
                 group_macro: typing.Optional[bool] = False,
                 ):
        super().__init__()
        
        self.data: str = data; assert self.data in self._supported_data
        self.do_group_eval: bool = self.data not in self._ignore_group_eval
        
        self.group_worst: bool = group_worst
        self.group_macro: bool = group_macro
        self.exclude_metrics: list = exclude_metrics
        
    def _get_metric_functions(self):
        if self.data == 'rmnist':
            return [
                ('nll', cross_entropy_with_probs),
                ('accuracy', multiclass_accuracy)
            ]
        elif self.data == 'cmnist':
            return [
                ('nll', binary_cross_entropy_with_probs),
                ('accuracy', binary_accuracy)
            ]
        elif self.data == 'pacs':
            return [
                ('nll', cross_entropy_with_probs),
                ('accuracy', multiclass_accuracy),
            ]
        elif self.data == 'vlcs':
            return [
                ('nll', cross_entropy_with_probs),
                ('accuracy', multiclass_accuracy),
            ]
        elif self.data == 'camelyon17':
            return [
                ('nll', binary_cross_entropy_with_probs),
                ('accuracy', binary_accuracy)
            ]
        elif self.data == 'iwildcam':
            return [
                ('nll', cross_entropy_with_probs),
                ('accuracy', multiclass_accuracy),
                ('macro-f1', partial(multiclass_f1, average='macro')),
                ('micro-f1', partial(multiclass_f1, average='micro')),
            ]
        elif self.data == 'rxrx1':
            return [
                ('nll', cross_entropy_with_probs),
                ('accuracy', multiclass_accuracy)
            ]
        elif self.data == 'fmow':
            return [
                ('nll', cross_entropy_with_probs),
                ('accuracy', multiclass_accuracy)
            ]
        elif self.data == 'poverty':
            return [
                ('mse', mean_squared_error),
                ('pearson', pearson_correlation),
            ]
        elif self.data == 'civilcomments':
            return [
                ('nll', binary_cross_entropy_with_probs),
                ('accuracy', binary_accuracy)
            ]
        elif self.data == 'celeba':
            return [
                ('nll', binary_cross_entropy_with_probs),
                ('accuracy', binary_accuracy)
            ]
        elif self.data == 'waterbirds':
            return [
                ('nll', binary_cross_entropy_with_probs),
                ('accuracy', binary_accuracy)
            ]
        else:
            raise ValueError(f"data={self.data} not recognized. Choose one of {self._supported_data}")

    def __call__(self,
                 y_pred: torch.FloatTensor,
                 y_true: typing.Union[torch.FloatTensor, torch.LongTensor],
                 group: torch.LongTensor = None,
                 **kwargs,
                 ) -> typing.Dict[str, torch.Tensor]:
        return self.evaluate(y_pred, y_true, group, **kwargs)

    def evaluate(self,
                 y_pred: torch.FloatTensor,
                 y_true: typing.Union[torch.FloatTensor, torch.LongTensor],
                 group: torch.LongTensor = None,
                 **kwargs,
                 ) -> typing.Dict[str, torch.Tensor]:
        """
        Evaluates model predictions, with data-specific metrics.
        Arguments:
                y_pred: 
                y_true:
                group:
        Returns:
                dictionary
        """
        results = dict()
        for metric_name, metric_fn in self._get_metric_functions():
            
            # exclude
            if metric_name in self.exclude_metrics:
                continue
            
            # compute metric (overall)
            results[metric_name] = metric_fn(y_pred, y_true, **kwargs)
            
            # compute metric (group-wise)
            if (group is not None) and (self.do_group_eval):
                
                if self.group_worst:
                    results[f"{metric_name}_wg"] = worst_group_metric_wrapper(
                        metric_fn=metric_fn, y_pred=y_pred, y_true=y_true, group=group, **kwargs,
                    )

                if self.group_macro:
                    results[f"{metric_name}_macro"] = macro_average_metric_wrapper(
                        metric_fn=metric_fn, y_pred=y_pred, y_true=y_true, group=group, **kwargs,
                    )
                
        return results
