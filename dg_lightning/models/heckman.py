
import os
import copy
import typing
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import List, Dict, Tuple, Union, Any, Optional

from dg_lightning.networks import NetworkInitializer
from dg_lightning.transforms import SupervisedLearningTransforms
from dg_lightning.optimization import (create_optimizer,
                                       create_learning_rate_scheduler)
from dg_lightning.metrics import MetricEvaluator
from dg_lightning.metrics.multilabel import (multilabel_accuracy,
                                             multilabel_f1)
from dg_lightning.utils.lightning_utils import from_argparse_args


data2task = {
    'camelyon17': 'binary',
    'poverty': 'regression',
    'iwildcam': 'multiclass',
    'rxrx1': 'multiclass',
    'pacs': 'multiclass',
    'vlcs': 'multiclass',
}


class HeckmanDGDomainClassifier(pl.LightningModule):
    def __init__(
        self,
        data: str,
        backbone: str,
        train_domains: List[int],
        imagenet: bool = True,
        augmentation: bool = True,
        randaugment: bool = False,
        optimizer: str = 'adamw',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = None,
        max_epochs: int = 5,
        ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['train_domains'])  # TODO: print hparams as table
        self.register_buffer('train_domains', torch.LongTensor(train_domains))
        self.automatic_optimization = False

        # (0) feature extractor
        self.encoder = \
            NetworkInitializer.initialize_backbone(
                name=self.hparams.backbone,
                data=self.hparams.data,
                pretrained=self.hparams.imagenet,
            )

        # (1) domain classifer
        self.predictor = \
            nn.Linear(
                in_features=self.encoder.out_features,
                out_features=len(self.train_domains),  # set with `register_buffer()`
            )

        # (2) input transforms
        self.train_transform: nn.Module = \
            SupervisedLearningTransforms[self.hparams.data](
                augmentation=self.hparams.augmentation,
                randaugment=self.hparams.randaugment,
            )
        self.eval_transform: nn.Module = \
            SupervisedLearningTransforms[self.hparams.data](
                augmentation=False,
                randaugment=False,
            )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:

        s_pred, s_true = self._shared_step(batch, training=True)

        opt = self.optimizers()
        opt.zero_grad()
        loss = self.pretrain_loss(s_pred, s_true)  # TODO: label smoothing & focal
        self.manual_backward(loss)
        opt.step()

        # log training loss
        self.log('train_loss', loss, on_step=True, reduce_fx='mean')

        # update learning rate scheduler
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            if isinstance(sch, torch.optim.lr_scheduler._LRScheduler):
                sch.step()

        return {'s_true': s_true, 's_pred': s_pred}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:
        s_pred, s_true = self._shared_step(batch, training=False)
        return {'s_true': s_true, 's_pred': s_pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:
        s_pred, s_true = self._shared_step(batch, training=False)
        return {'s_true': s_true, 's_pred': s_pred}

    def _shared_step(self, batch: Dict[str, torch.Tensor], training: bool = True) -> Tuple[torch.Tensor]:

        # fetch data
        x, domain = batch['x'], batch['domain']

        # apply input transforms
        if training:
            x = self.train_transform(x)
        else:
            x = self.eval_transform(x)

        # forward; (B, K)
        s_pred_in_probits = self.predictor(self.encoder(x))

        # create domain labels; (B,  )
        s_true_2d = domain.view(-1, 1).eq(self.train_domains.view(1, -1))  # (B, K)
        s_true_1d = s_true_2d.nonzero(as_tuple=True)[1]                    # (B,  )

        return s_pred_in_probits, s_true_1d

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end(outputs, prefix='train')

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end(outputs, prefix='val')

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end(outputs, prefix='test')

    def _shared_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], prefix: str) -> None:

        # concatenate batch outputs
        s_true = torch.cat([out['s_true'] for out in outputs], dim=0)  # (N,  )
        s_pred = torch.cat([out['s_pred'] for out in outputs], dim=0)  # (N, K)

        # process `s_pred` and `s_true` to required format
        s_true = torch.zeros_like(s_pred).scatter(dim=1, index=s_true.view(-1, 1), value=1.)
        s_pred = torch.distributions.Normal(loc=0., scale=1.).cdf(s_pred)

        # evaluate metrics
        val_acc = multilabel_accuracy(s_pred, s_true, average='macro')
        val_f1 = multilabel_f1(s_pred, s_true, average='macro')
        self.log_dict(
            {
                f"{prefix}_accuracy": val_acc,
                f"{prefix}_f1": val_f1,
            },
            prog_bar=True,
        )

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer],
                                            List[torch.optim.lr_scheduler._LRScheduler]]:
        """Add function docstring."""

        optimizer = \
            create_optimizer(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                name=self.hparams.optimizer,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        lr_scheduler = \
            create_learning_rate_scheduler(
                optimizer=optimizer,
                name=self.hparams.lr_scheduler,
                epochs=self.hparams.max_epochs,
            )

        return [optimizer], [lr_scheduler]

    @staticmethod
    def pretrain_loss(s_pred: torch.FloatTensor,
                       s_true: torch.LongTensor,
                       label_smoothing: float = 0.0,
                       focal: bool = False,
                       alpha: typing.Optional[float] = 1.0,
                       gamma: typing.Optional[float] = 2.0,
                       ) -> torch.FloatTensor:
        """
            Loss function used to pretrain domain classifers.
            Supports label smoothing and focal weighting.
        """
        assert (s_pred.shape[0] == s_true.shape[0]), "N"
        assert (s_pred.ndim == 2) and (s_true.ndim == 1), "(N, K) and (N,  )"
        assert (label_smoothing >= 0.0) and (label_smoothing < 0.5)
        normal = torch.distributions.Normal(loc=0, scale=1)

        # transform probits into probabilities
        s_pred_in_probs = normal.cdf(s_pred)  # (N, K) <- (N, K)

        # 1d indicators -> 2d one-hot vectors
        s_true_2d = F.one_hot(s_true, num_classes=s_pred.shape[1]).float()

        # hard targets -> soft targets (with label smoothing)
        s_true_2d_soft = s_true_2d * (1.0 - label_smoothing) + (1.0 - s_true_2d) * label_smoothing

        if not focal:
            return F.binary_cross_entropy(s_pred_in_probs, s_true_2d_soft, weight=None)

        # compute focal weights (for selected; S = 1)
        with torch.no_grad():
            sel_mask = s_true_2d.ge(0.5)  # or .eq(1)
            sel_weights = (1. - s_pred_in_probs).masked_scatter_(
                mask=torch.logical_not(sel_mask), source=torch.zeros_like(s_pred),
            )

        # compute focal weights (for not selected; S = 0)
        with torch.no_grad():
            not_sel_mask = s_true_2d.lt(0.5)  # or .eq(0)
            not_sel_weights = (1. - (1. - s_pred_in_probs)).masked_scatter_(
                mask=torch.logical_not(not_sel_mask), source=torch.zeros_like(s_pred)
            )

        # aggregate focal weights
        weights = (sel_weights + not_sel_weights).clone().detach()
        weights = alpha * torch.pow(weights, gamma)

        return F.binary_cross_entropy(s_pred_in_probs, s_true_2d_soft, weight=weights)

    @classmethod
    def from_argparse_args(cls,
                           args: Union[argparse.Namespace, Dict[str, Any]],
                           **kwargs) -> pl.LightningModule:
        return from_argparse_args(cls, args, **kwargs)


class HeckmanDG(pl.LightningModule):
    def __init__(
        self,
        data: str,
        backbone: str,
        train_domains: List[int],
        imagenet: bool = True,
        augmentation: bool = True,
        randaugment: bool = False,
        g_optimizer: str = 'adamw',
        g_learning_rate: float = 1e-4,
        g_weight_decay: float = 1e-5,
        g_lr_scheduler: str = None,
        f_optimizer: str = 'sgd',
        f_learning_rate: float = 3e-2,
        f_weight_decay: float = 1e-5,
        f_lr_scheduler: str = 'cosine_decay',
        c_optimizer: str = 'adam',
        c_learning_rate: float = 1e-2,
        c_weight_decay: float = 0.,
        c_lr_scheduler: str = None,
        max_epochs: int = 5,
        ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['train_domains'])  # TODO: print for sanity check
        self.register_buffer('train_domains', torch.LongTensor(train_domains))
        self.automatic_optimization = False
        self.task = data2task[self.hparams.data]

        # (0) domain selection encoder ($\varphi_g$)
        self.g_encoder = \
            NetworkInitializer.initialize_backbone(
                name=self.hparams.backbone,
                data=self.hparams.data,
                pretrained=self.hparams.imagenet,
            )

        # (1) domain classifier ($\omega_g$)
        self.g_predictor = \
            nn.Linear(
                in_features=self.g_encoder.out_features,
                out_features=len(self.train_domains),
            )

        # (2) outcome encoder ($\varphi_f$)
        self.f_encoder = \
            NetworkInitializer.initialize_backbone(
                name=self.hparams.backbone,
                data=self.hparams.data,
                pretrained=self.hparams.imagenet,
            )

        # (3) outcome predictor ($\varphi_f$)
        self.f_predictor = \
            nn.Linear(
                in_features=self.f_encoder.out_features,
                out_features=NetworkInitializer._out_features[self.hparams.data][0]
            )

        # (4) correlation (& sigma)
        if self.task == 'multiclass':
            J: int = self.f_predictor.out_features
            K: int = len(self.train_domains)
            self._rho = \
                nn.Parameter(
                    data=torch.randn(K, J+1, requires_grad=True)
                )
        else:
            K: int = len(self.train_domains)
            self._rho = \
                nn.Parameter(
                    data=torch.zeros(K, requires_grad=True)
                )

        self.sigma = \
            nn.Parameter(
                data=torch.ones(1),
                requires_grad=self.task == 'regression'
            )

        # (5) input transforms
        self.train_transform: nn.Module = \
            SupervisedLearningTransforms[self.hparams.data](
                augmentation=self.hparams.augmentation,
                randaugment=self.hparams.randaugment,
            )
        self.eval_transform: nn.Module = \
            SupervisedLearningTransforms[self.hparams.data](
                augmentation=False,
                randaugment=False,
            )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:

        # forward
        s_pred, s_true = self._shared_g_step(batch, training=True)  # probits, 1d
        y_pred, y_true = self._shared_f_step(batch, training=True)  # probits, 1d
        rho = torch.tanh(self._rho[s_true])                         # (B,  ) or (B, J+1)

        # compute loss
        loss = self.loss_function(
            y_pred=y_pred, y_true=y_true, s_pred=s_pred, s_true=s_true,
            rho=rho, sigma=self.sigma
        )

        # get optimizers
        g_opt, f_opt, corr_opt = self.optimizers()
        g_opt.zero_grad(); f_opt.zero_grad(); corr_opt.zero_grad()
        self.manual_backward(loss)
        g_opt.step(); f_opt.step(); corr_opt.step()

        self.log('train_loss', loss.item(), on_step=True, prog_bar=True, reduce_fx='mean')

        if self.trainer.is_last_batch:
            g_sch, f_sch, corr_sch = self.lr_schedulers()
            g_sch.step(); f_sch.step(); corr_sch.step()

        return {
            's_true': s_true, 's_pred': s_pred,
            'y_true': y_true, 'y_pred': y_pred,
            'eval_group': batch['eval_group']
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:

        # forward (inference)
        s_pred, s_true = self._shared_g_step(batch, training=False)
        y_pred, y_true = self._shared_f_step(batch, training=False)

        return {
            's_true': s_true, 's_pred': s_pred,
            'y_true': y_true, 'y_pred': y_pred,
            'eval_group': batch['eval_group']
        }

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:

        # forward (inference)
        s_pred, s_true = self._shared_g_step(batch, training=False)  # unnecessary?
        y_pred, y_true = self._shared_f_step(batch, training=False)

        return {
            's_true': s_true, 's_pred': s_pred,
            'y_true': y_true, 'y_pred': y_pred,
            'eval_group': batch['eval_group']
        }

    def _shared_g_step(self,
                       batch: Dict[str, torch.Tensor],
                       training: bool = False) -> Tuple[torch.Tensor]:
        # fetch data
        x, domain = batch['x'], batch['domain']

        # apply input transforms
        if training:
            x = self.train_transform(x)
        else:
            x = self.eval_transform(x)

        # forward; (B, K)
        s_pred_in_probits = self.g_predictor(self.g_encoder(x))

        # create domain labels; (B,  )
        s_true_2d = domain.view(-1, 1).eq(self.train_domains.view(1, -1))  # (B, K)
        s_true_1d = s_true_2d.nonzero(as_tuple=True)[1]                    # (B,  )

        return s_pred_in_probits, s_true_1d  # (B, K),  (B,  )

    def _shared_f_step(self,
                       batch: Dict[str, torch.Tensor],
                       training: bool = False) -> Tuple[torch.Tensor]:
        # fetch data
        x, y = batch['x'], batch['y']

        # apply input transforms
        if training:
            x = self.train_transform(x)
        else:
            x = self.eval_transform(x)

        # forward
        y_pred_in_probits = self.f_predictor(self.f_encoder(x))
        if (y_pred_in_probits.ndim == 2) and (y_pred_in_probits.size(1) == 1):
            y_pred_in_probits.squeeze_(1)  # (B,  ) <- (B,  1)

        return y_pred_in_probits, y

    def train_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end(outputs, prefix='train')

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end(outputs, prefix='val')

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        self._shared_epoch_end(outputs, prefix='test')

    def _shared_epoch_end(self,
                          outputs: List[Dict[str, torch.Tensor]],
                          prefix: str) -> None:

        # concatenate batch outputs
        eval_group = torch.cat([out['eval_group'] for out in outputs], dim=0)
        y_true = torch.cat([out['y_true'] for out in outputs], dim=0)
        y_pred = torch.cat([out['y_pred'] for out in outputs], dim=0)
        y_probs = self._process_pred_for_eval(y_pred)  # probits to probs

        # evaluate metrics
        evaluator = MetricEvaluator(data=self.hparams.data)
        metrics = evaluator(y_probs, y_true, group=eval_group)

        self.log_dict(
            {f"{prefix}_{name}": val for name, val in metrics.items()},
            prog_bar=True, on_epoch=True,
        )

    def _process_pred_for_eval(self, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        normal = torch.distributions.Normal(loc=0., scale=1.)
        if self.task == 'regression':
            return y_pred                    # as-is
        elif self.task == 'binary':
            normal = torch.distributions.Normal(loc=0., scale=1.)
            return normal.cdf(y_pred)        # probs
        elif self.task == 'multiclass':
            # FIXME: this is a proxy
            return F.softmax(y_pred, dim=1)  # probs
        else:
            raise NotImplementedError

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer],
                                            List[torch.optim.lr_scheduler._LRScheduler]]:
        """Add function docstring."""

        g_opt = \
            create_optimizer(
                params=[
                    {'params': filter(lambda p: p.requires_grad, self.g_encoder.parameters())},
                    {'params': filter(lambda p: p.requires_grad, self.g_predictor.parameters())},
                ],
                name=self.hparams.g_optimizer,
                lr=self.hparams.g_learning_rate,
                weight_decay=self.hparams.g_weight_decay,
            )

        f_opt = \
            create_optimizer(
                params=[
                    {'params': filter(lambda p: p.requires_grad, self.f_encoder.parameters())},
                    {'params': filter(lambda p: p.requires_grad, self.f_predictor.parameters())},
                ],
                name=self.hparams.f_optimizer,
                lr=self.hparams.f_learning_rate,
                weight_decay=self.hparams.f_weight_decay,
            )

        corr_opt = \
            create_optimizer(
                params=filter(lambda p: p.requires_grad, [self._rho, self.sigma]),
                name=self.hparams.c_optimizer,
                lr=self.hparams.c_learning_rate,
                weight_decay=self.hparams.c_weight_decay,
            )

        g_lr_sch = \
            create_learning_rate_scheduler(
                optimizer=g_opt,
                name=self.hparams.g_lr_scheduler,
                epochs=self.hparams.max_epochs
            )

        f_lr_sch = \
            create_learning_rate_scheduler(
                optimizer=f_opt,
                name=self.hparams.f_lr_scheduler,
                epochs=self.hparams.max_epochs
            )

        corr_lr_sch = \
            create_learning_rate_scheduler(
                optimizer=corr_opt,
                name=self.hparams.c_lr_scheduler,
                epochs=self.hparams.max_epochs
            )

        return [g_opt, f_opt, corr_opt], [g_lr_sch, f_lr_sch, corr_lr_sch]

    def loss_function(self, y_pred, y_true, s_pred, s_true, rho, sigma) -> torch.FloatTensor:
        if self.task == 'regression':
            return self.cross_domain_regression_loss(
                y_pred=y_pred, y_true=y_true, s_pred=s_pred, s_true=s_true,
                rho=rho, sigma=sigma
            )
        elif self.task == 'binary':
            return self.cross_domain_binary_classification_loss(
                y_pred=y_pred, y_true=y_true, s_pred=s_pred, s_true=s_true,
                rho=rho,
            )
        elif self.task == 'multiclass':
            raise NotImplementedError
        else:
            raise ValueError

    @staticmethod
    def cross_domain_regression_loss(y_pred: torch.FloatTensor,
                                     y_true: torch.FloatTensor,
                                     s_pred: torch.FloatTensor,
                                     s_true: torch.LongTensor,
                                     rho: torch.FloatTensor,
                                     sigma: Union[torch.FloatTensor, float],
                                     eps: Optional[float] = 1e-7, ) -> torch.FloatTensor:
        """
        # TODO: improve function docstring.
        Loss function used for regression tasks.\n
        Arguments:
            y_pred : (N,  )
            y_true : (N,  )
            s_pred : (N, K)
            s_true : (N,  )
            rho    : (N,  )
            sigma  : (1,  )
        Returns:
            ...
        """

        # shape check
        assert (y_pred.ndim == 1) and (y_true.ndim == 1) and (y_pred.shape == y_true.shape)
        assert (s_pred.ndim == 2) and (s_true.ndim == 1) and (s_pred.shape[0] == s_pred.shape[0])
        assert (y_true.shape[0] == s_true.shape[0]) and (s_true.shape[0] == rho.shape[0])

        # standard normal distribution
        normal = torch.distributions.Normal(loc=0., scale=1.)

        # Gather values from `s_pred` those that correspond to the true domains; shape = (N,  )
        s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).squeeze()

        # -log{p(S_k = 1, y)} = -log{p(y)} -log{p(S_k = 1 | y)}; shape = (N,  )
        loss_selected = \
            - 1.0 * torch.log(
                normal.cdf(
                    (s_pred_k + rho * (y_true - y_pred).div(eps + sigma)) / (eps + torch.sqrt(1 - rho**2))
                ) + eps
            ) \
            + 0.5 * (
                torch.log(2 * torch.pi * (sigma ** 2)) + F.mse_loss(y_pred, y_true, reduction='none').div(eps + sigma**2)
            )

        # domain indicators in 2d; (N, K) <- (N,  )
        s_true_2d = \
            torch.zeros_like(s_pred).scatter_(
                dim=1, index=s_true.view(-1, 1), src=torch.ones_like(s_pred),
            )

        # -log{Pr(S_l = 0)} for l \neq k; shape = (N(K-1),  )
        loss_not_selected = - torch.log(1 - normal.cdf(s_pred) + eps)                          # (N     , K)
        loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.)  # (N     , K)
        loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())            # (N(K-1),  )

        return torch.cat([loss_selected, loss_not_selected], dim=0).mean()

    @staticmethod
    def cross_domain_binary_classification_loss(y_pred: torch.FloatTensor,
                                                y_true: torch.LongTensor,
                                                s_pred: torch.FloatTensor,
                                                s_true: torch.LongTensor,
                                                rho: torch.FloatTensor,
                                                eps: Optional[float] = 1e-7) -> torch.FloatTensor:
        """
        Loss function used for regression tasks.\n
        Arguments:
            y_pred : 1d `torch.FloatTensor` of shape (N,  ); in probits.
            y_true : 1d `torch.LongTensor`  of shape (N,  ); with values in {0, 1}.
            s_pred : 2d `torch.FloatTensor` of shape (B, K); in probits.
            s_true : 1d `torch.LongTensor`  of shape (N,  ); with values in [0, K-1].
            rho    : 1d `torch.FloatTensor` of shape (N,  ); with values in [-1, 1].
        Returns:
            ...
        """

        # shape check
        assert (y_pred.ndim == 1) and (y_true.ndim == 1) and (y_pred.shape == y_true.shape)
        assert (s_pred.ndim == 2) and (s_true.ndim == 1) and (s_pred.shape[0] == s_pred.shape[0])
        assert (y_true.shape[0] == s_true.shape[0]) and (s_true.shape[0] == rho.shape[0])

        normal = torch.distributions.Normal(loc=0., scale=1.)

        # Gather from `s_pred`, values with indices that correspond to the true domains
        s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).squeeze()  # (N,  )

        # - log Pr[S_k = 1, Y = 1]; shape = (N,  )
        loss_selected_pos = - y_true.float() * torch.log(
            HeckmanDG.bivariate_normal_cdf(a=s_pred_k, b=y_pred, rho=rho) + eps
        )
        loss_selected_pos = torch.nan_to_num(loss_selected_pos, nan=0., posinf=0., neginf=0.)
        loss_selected_pos = loss_selected_pos[y_true.bool()]  # Y = 1

        # - log Pr[S_k = 1, Y = 0]; shape = (N,  )
        loss_selected_neg = - (1 - y_true.float()) * torch.log(
            normal.cdf(s_pred_k) - HeckmanDG.bivariate_normal_cdf(a=s_pred_k, b=y_pred, rho=rho) + eps
        )
        loss_selected_neg = torch.nan_to_num(loss_selected_neg, nan=0., posinf=0., neginf=0.)
        loss_selected_neg = loss_selected_neg[(1 - y_true).bool()]  # Y = 0

        loss_selected = torch.cat([loss_selected_pos, loss_selected_neg], dim=0)

        # Create a 2d indicator for `s_true`
        #   Shape; (N, K)
        s_true_2d = torch.zeros_like(s_pred).scatter_(
            dim=1, index=s_true.view(-1, 1), src=torch.ones_like(s_pred)
        )

        # -\log Pr[S_l = 0] for l \neq k
        loss_not_selected = - torch.log(1 - normal.cdf(s_pred) + eps)                          # (N, K)
        loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.)  # (N, K)
        loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())            # (NK - N,  )

        return torch.cat([loss_selected, loss_not_selected], dim=0).mean()

    @staticmethod
    def bivariate_normal_cdf(a: torch.FloatTensor,
                             b: torch.FloatTensor,
                             rho: torch.FloatTensor,
                             steps: Optional[int] = 100) -> torch.FloatTensor:
        """
        Approximation of standard bivariate normal cdf using the trapezoid rule.
        The decomposition is based on:
            Drezner, Z., & Wesolowsky, G. O. (1990).
            On the computation of the bivariate normal integral.
            Journal of Statistical Computation and Simulation, 35(1-2), 101-107.
        Arguments:
            a: 1d `torch.FloatTensor` of shape (N,  )
            b: 1d `torch.FloatTensor` of shape (N,  )
            rho: 1d `torch.FloatTensor` of shape (N,  )
        Returns:
            1d `torch.FloatTensor` of shape (N,  )
        """

        normal = torch.distributions.Normal(loc=0., scale=1.)
        a, b = a.view(-1, 1), b.view(-1, 1)  # for proper broadcasting with x

        HeckmanDG

        grids: List[torch.FloatTensor] = [
            HeckmanDG.linspace_with_grads(start=0, stop=r, steps=steps) for r in rho
        ]                              #  N * (steps,  )
        x = torch.stack(grids, dim=0)  # (N, steps)
        y = 1 / torch.sqrt(1 - torch.pow(x, 2)) * torch.exp(
            - (torch.pow(a, 2) + torch.pow(b, 2) - 2 * a * b * x) / (2 * (1 - torch.pow(x, 2)))
        )

        return \
            normal.cdf(a.squeeze()) * normal.cdf(b.squeeze()) + \
            (1 / (2 * torch.pi)) * torch.trapezoid(y=y, x=x)

    @staticmethod
    def linspace_with_grads(start: torch.FloatTensor,
                            stop: torch.FloatTensor,
                            steps: Optional[int] = 100) -> torch.Tensor:
        """
        Creates a 1d grid while preserving gradients associated with `start` and `stop`.
        Reference:
            https://github.com/esa/torchquad/blob/4be241e8462949abcc8f1ace48d7f8e5ee7dc136/torchquad/integration/utils.py#L7
        """
        grid = torch.linspace(0, 1, steps, device=stop.device)   # create 0 ~ 1 equally spaced grid
        grid *= stop - start                                     # scale grid to desired range
        grid += start
        return grid

    @staticmethod
    def cross_domain_multiclass_classification_loss(y_pred,
                                                    y_true,
                                                    s_pred,
                                                    s_true,
                                                    rho,
                                                    eps: Optional[float] = 1e-7) -> torch.FloatTensor:
        raise NotImplementedError

    @classmethod
    def from_argparse_args(cls,
                           args: Union[argparse.Namespace, Dict[str, Any]],
                           **kwargs) -> pl.LightningModule:
        return from_argparse_args(cls, args, **kwargs)
