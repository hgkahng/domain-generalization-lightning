
import os
import sys
import yaml

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)  # appending project folder to list of system paths

import yaml
import logging
import warnings
import argparse

import torch
import pytorch_lightning as pl

from dg_lightning.models.heckman import HeckmanDGDomainClassifier
from dg_lightning.datasets.wilds.camelyon17 import Camelyon17DataModule

from dg_lightning.utils.rich_utils import print_args_as_table


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(prog="HeckmanDG Domain Classifier", add_help=True)
    parser = HeckmanDGDomainClassifier.add_model_specific_args(parser)
    parser = Camelyon17DataModule.add_data_specific_args(parser)

    # trainer arguments
    parser.add_argument('--gpus', nargs='+', type=int, default=[], help='')
    parser.add_argument('--max_epochs', type=int, default=5, help='')

    # misc arguments
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='')
    parser.add_argument('--hash', type=str, default=None, help='')

    # override argument defaults from yaml config files
    parser.add_argument('--config', action='append')
    args, _ = parser.parse_known_args()
    if args.config is not None:
        assert isinstance(args.config, list)
        for conf_fname in args.config:
            assert conf_fname.endswith('.yaml')
            with open(conf_fname, 'r') as f:
                cfg = yaml.safe_load(f)
                cfg = {k: v for k, v in cfg.items() if k in vars(args)}
                parser.set_defaults(**cfg)

    args, _ = parser.parse_known_args()
    if args.hash is None:
        import datetime
        setattr(args, 'hash', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    
    return args


def main(args: argparse.Namespace) -> None:

    # set random seed
    pl.seed_everything(args.seed)

    # print configurations
    print_args_as_table(args)

    # data
    if args.data == 'camelyon17':
        dm = Camelyon17DataModule.from_argparse_args(args)
        dm.setup(stage=None)
    elif args.data == 'pacs':
        raise NotImplementedError
    else:
        raise ValueError
    
    # (3) pretraining module
    model = HeckmanDGDomainClassifier.from_argparse_args(args)
    trainer = pl.Trainer(
        accelerator='gpu' if len(args.gpus) > 0 else 'cpu',
        devices=args.gpus,
        max_epochs=args.max_epochs,
        num_nodes=1,
        logger=[
            pl.loggers.CSVLogger(
                save_dir=os.path.join(
                    args.checkpoint_dir,
                    f"{args.data}",
                    f"{model.__class__.__name__}"
                ),
                name=args.hash,
                version=0,
            )
        ],
        callbacks=[
            pl.callbacks.RichProgressBar(leave=True),
            pl.callbacks.RichModelSummary(max_depth=1),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(args.checkpoint_dir, f"{model.__class__.__name__}", args.hash),
                monitor='val_f1',
                mode='max',
            ),
        ],
        replace_sampler_ddp=True,  # FIXME:
    )

    # fit
    trainer.fit(
        model=model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm._id_val_dataloader(),  # FIXME: 
    )

    # test
    trainer.test(
        model=model,
        ckpt_path='best',
        dataloaders=dm._id_val_dataloader(),  # FIXME: 
    )


if __name__ == "__main__":

    warnings.filterwarnings(action='ignore')
    args = parse_arguments()
    try:
        main(args)
    except KeyboardInterrupt as _:
        sys.exit(0)
