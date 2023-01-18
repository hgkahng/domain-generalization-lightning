
import inspect
import argparse

from typing import Union, Dict, Any

import pytorch_lightning as pl


def from_argparse_args(cls,
                       args: Union[argparse.Namespace, Dict[str, Any]],
                       **kwargs) -> Union[pl.LightningModule, pl.LightningDataModule]:
    init_arg_names = [k for k in inspect.signature(cls.__init__).parameters]
    init_kwargs = {k: v for k, v in vars(args).items() if k in init_arg_names}
    if kwargs:
        init_kwargs.update(kwargs)
    return cls(**init_kwargs)
