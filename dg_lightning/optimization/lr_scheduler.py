
import math
import typing
import warnings

import torch
import torch.optim as optim


def create_learning_rate_scheduler(
    optimizer: optim.Optimizer, 
    name: str,
    **kwargs,
    ) -> optim.lr_scheduler._LRScheduler:
    """
    Arguments:
        optimizer: `optim.Optimizer`
        name: `str`
        epochs: `int`
        **kwargs:
    Returns:
        a `torch.optim.lr_scheduler._LRScheduler` instance
    """

    if name == 'cosine_decay':
        return LinearWarmupCosineDecayLR(
            optimizer=optimizer,
            total_epochs=kwargs.get('epochs', None),  # raise error
            warmup_epochs=kwargs.get('warmup_epochs', 0),
            min_lr=kwargs.get('min_lr', 0.)
        )
    elif name == 'step':
        # use for `PovertyMap`
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=1,
            gamma=0.96,
        )
    elif name == 'cyclic':
        raise NotImplementedError
    elif (name == 'none') or (name is None):
        return None
    else:
        raise ValueError(f"name=`{name}` not recognized.")


class LinearWarmupCosineDecayLR(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float = .0,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs,
        ) -> None:
        """
        Arguments:
            optimizer (`~torch.optim.Optimizer`):
                the optimizer for which to schedule the learning rate.
            total_epochs (`int`):
                The total number of training epochs.
            warmup_epochs (`int`):
                The number of steps for the warmup phase.
            min_lr (`float`):
                The start/final learning rate for linear warmup & cosine decay.
            last_epoch (`int`):
                The index of the last epoch when resuming training.
        Returns:
            `torch.optim.lr_scheduler._LRScheduler` with the appropriate schedule.
        """

        self._total_epochs: int = total_epochs
        self._warmup_epochs: int = warmup_epochs
        self._warmup_start_lrs: typing.List[float] = [min_lr] * len(optimizer.param_groups)
        self._decay_end_lrs: typing.List[float] = [min_lr] * len(optimizer.param_groups)

        # TODO; remove; left for backwards compatibility
        if 'warmup_start_lr' in kwargs:
            self._warmup_start_lrs: typing.List[float] = [kwargs['warmup_start_lr']] * len(optimizer.param_groups)
        if 'min_decay_lr' in kwargs:
            self._decay_end_lrs: typing.List[float] = [kwargs['min_decay_lr']] * len(optimizer.param_groups)
        
        super(LinearWarmupCosineDecayLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)  # type: ignore

    def get_lr(self) -> typing.List[float]:
        """A must-implement function for classes that inherit `torch.optim.lr_scheduler._LRScheduler`."""
        if not self._get_lr_called_within_step:  # type: ignore
            warnings.warn("To get the last learning rate computed by the scheduler, "
                            "please use `get_last_lr()`.", UserWarning)
        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:  # type: ignore
            # learning rate during the warmup phase
            return [
                self.linear_warmup_lr(
                    t=self.last_epoch,  # type: ignore
                    T=self._warmup_epochs,
                    base_lr=base_lr,
                    min_lr=min_lr
                )
                for base_lr, min_lr in zip(self.base_lrs, self._warmup_start_lrs)  # type: ignore
            ]
        else:
            # learning rate during the cosine decay phase
            epochs_after_warmup: int = self.last_epoch - self._warmup_epochs  # type: ignore
            remaining_epochs: int = self._total_epochs - self._warmup_epochs
            return [
                self.cosine_decay_lr(
                    t=epochs_after_warmup,
                    T=remaining_epochs,
                    base_lr=base_lr,
                    min_lr=min_lr
                )
                for base_lr, min_lr in zip(self.base_lrs, self._decay_end_lrs)  # type: ignore
            ]

    @staticmethod
    def linear_warmup_lr(t: int, T: int, base_lr: float, min_lr: float) -> float:
        return min_lr + (base_lr - min_lr) * (t / T)

    @staticmethod
    def cosine_decay_lr(t: int, T: int, base_lr: float, min_lr: float) -> float:
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t / T))
