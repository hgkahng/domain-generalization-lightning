

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)  # appending project folder to list of system paths

import yaml
import typing
import logging
import warnings
import argparse

import torch
import pytorch_lightning as pl


from dg_lightning.models.heckman import HeckmanDGDomainClassifier
from dg_lightning.models.heckman import HeckmanDG
from dg_lightning.datasets.wilds.camelyon17 import Camelyon17DataModule

from dg_lightning.utils.rich_utils import print_args_as_table


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    raise NotImplementedError  # FIXME:


def get_state_dict(ckpt_file: str, keys: typing.List[str] = []) -> typing.Any:
    ckpt = torch.load(ckpt_file, map_location='cpu')
    _state_dict = None
    for k in keys:
        try:
            _state_dict = ckpt[k]
            break
        except KeyError as _:
            continue
    if _state_dict is None:
        raise KeyError(f"Failed to load state using keys in {keys}")
    return _state_dict


def freeze_weights(*modules: typing.Iterable[torch.nn.Module]):
    # TODO: handle batch norm
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False

def main(args: argparse.Namespace) -> None:
    
    # set random seed
    pl.seed_everything(args.seed)

    # print configurations
    print_args_as_table(args)

    # data
    if args.data == 'camelyon17':
        dm = Camelyon17DataModule.from_argparse_args(args)
        dm.setup(stage=None)
    else:
        raise NotImplementedError

    # model
    model = HeckmanDG.from_argparse_args(args)

    # FIXME: load selection weights
    # TODO: add `args.pretrained_model_ckpt`
    if os.path.isfile(args.pretrained_model_ckpt):
        # TODO: exceptions
        model.g_encoder.load_state_dict(
            get_state_dict(
                ckpt_file=args.pretrained_model_ckpt,
                keys=['g_encoder', 'encoder', ]
            )
        )
        # TODO: exceptions
        model.g_predictor.load_state_dict(
            get_state_dict(
                ckpt_file=args.pretrained_model_ckpt,
                keys=['g_predictor', 'predictor', ]
            )
        )

    # FIXME: freeze selection model weights
    if args.freeze_g_encoder:
        freeze_weights(model.g_encoder)
    if args.freeze_g_predictor:
        freeze_weights(model.g_predictor)

    # trainer
    trainer = pl.Trainer(
        accelerator='gpu' if len(args.gpu) > 0 else 'cpu',
        devices=args.gpus,  # list of gpu numbers
        max_epochs=args.max_epochs,
        num_nodes=1,
        logger=[],
        callbacks=[],
        replace_sampler_ddp=True,  # FIXME:
    )

    # fit
    trainer.fit(
        model=model,
        dm=dm,
    )

    # test
    trainer.test(
        model=model,
        dm=dm,
    )


    raise NotImplementedError  # FIXME:


if __name__ == "__main__":

    warnings.filterwarnings(action='ignore')
    args = parse_arguments()
    try:
        main(args)
    except KeyboardInterrupt as _:
        sys.exit(0)
