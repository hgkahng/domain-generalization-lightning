
import os
import sys

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
from dg_lightning.datasets.wilds.poverty import PovertyMapDataModule

from dg_lightning.utils.rich_utils import print_args_as_table
from dg_lightning.utils.logging_utils import modify_lightning_logger_settings
from dg_lightning.utils.argparse_utils import create_datetime_hash
from dg_lightning.utils.argparse_utils import override_defaults_given_yaml_config_file


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(prog="Pretrain HeckmanDG Domain Classifier", add_help=True)
    
    # data arguments
    parser.add_argument('--data', type=str, default='camelyon17',
                        choices=('camelyon17', 'poverty', 'iwildcam', 'rxrx1'),
                        help='Data name (default: camelyon17)')
    parser.add_argument('--train_domains', type=str, nargs='+', default=[0, 3, 4],
                        help='Training domain indicators')
    parser.add_argument('--validation_domains', type=str, nargs='+', default=[1],
                        help='OOD validation domain indicators')
    parser.add_argument('--test_domains', type=str, nargs='+', default=[2],
                        help='OOD test domain indicators')

    # model arguments
    parser.add_argument('--backbone', type=str, default='densenet121',
                        choices=('densnet121', 'resnet18', 'resnet50'),
                        help='Neural network architecture (default: densnet121)')
    parser.add_argument('--imagenet', action='store_true',
                        help='Use ImageNet pretrained weights (default: False)')
    parser.add_argument('--augmentation', action='store_true',
                        help='Apply data augmentation during training (default: False)')
    parser.add_argument('--randaugment', action='store_true',
                        help='Apply RandAugment during training (default: False)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=('sgd', 'adam', 'adamw'),
                        help='Optimizer name (default: sgd)')
    parser.add_argument('--learning_rate', type=float, default=3e-2,
                        help='Base learning rate (default: 0.03)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay factor (default: 0.00001)')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        help='Learning rate scheduler name (default: None)')
    
    # dataloader arguments
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of CPU threads used for data loading (default: 16)')
    parser.add_argument('--prefetch_factor', type=int, default=8,
                        help='Number of training batches prefetched by each CPU (default: 8)')
    
    # trainer arguments
    parser.add_argument('--gpus', nargs='*', type=int, default=[],
                        help='GPU numbers (default: [])')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Maximum number of training epochs (default: 30)')

    # misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Root checkpoint directory (default: checkpoints)')
    parser.add_argument('--hash', type=str, default=None,
                        help='')

    # override argument defaults from yaml config files
    parser.add_argument('--config', action='append', help='Configuration yaml file(s)')
    parser = override_defaults_given_yaml_config_file(parser)
    args, _ = parser.parse_known_args()
    
    # create datetime hash if not specified
    if args.hash is None:
        setattr(args, 'hash', create_datetime_hash())
    
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
    
    elif args.data == 'poverty':
        
        dm = PovertyMapDataModule.from_argparse_args(args)
        dm.setup(stage=None)
        
        # FIXME: this is currently a hack ... find a cleaner way to do it.
        setattr(args, 'train_domains', dm.train_domains)

    elif args.data == 'pacs':
    
        raise NotImplementedError
    
    else:
    
        raise ValueError
    
    # model
    model = HeckmanDGDomainClassifier.from_argparse_args(args)

    # logging & checkpointing
    save_dir = os.path.join(
        args.checkpoint_dir,
        args.data,
        f'{model.__class__.__name__}',  # FIXME: 
        f'{args.hash}'
    )
    logging.getLogger('pytorch_lightning').info(f"Save directory: {save_dir}")

    # trainer
    trainer = pl.Trainer(
        accelerator='gpu' if len(args.gpus) > 0 else 'cpu',
        devices=args.gpus,
        max_epochs=args.max_epochs,
        num_nodes=1,
        logger=[
            pl.loggers.CSVLogger(save_dir=save_dir)
        ],
        callbacks=[
            pl.callbacks.RichProgressBar(leave=True),
            pl.callbacks.RichModelSummary(max_depth=1),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(monitor='val_f1', mode='max', patience=3),
            pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                monitor='val_f1',
                mode='max',
            ),
        ],
        log_every_n_steps=100,       # log every 100 batch steps
        enable_model_summary=False,  # added as callback
        replace_sampler_ddp=True,    # FIXME: only valid with multiple gpus
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
        dataloaders=dm._id_val_dataloader(),      # FIXME: use id_test data if available
    )


if __name__ == "__main__":

    modify_lightning_logger_settings()
    warnings.filterwarnings(action='ignore')
    args = parse_arguments()
    try:
        main(args)
    except KeyboardInterrupt as _:
        sys.exit(0)
