
import os
import time
import typing
import inspect
import functools
import argparse

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from ray.util.multiprocessing import Pool as RayPool
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader, ConcatDataset


class SingleCamelyon17(torch.utils.data.Dataset):
    _allowed_hospitals = [0, 1, 2, 3, 4]
    def __init__(self,
                 root: str = './data/wilds/camelyon17_v1.0',
                 hospital: int = 0,
                 split: str = None,
                 in_memory: typing.Optional[int] = 0) -> None:
        super().__init__()

        self.root = root
        self.hospital = int(hospital)
        self.split = split
        self.in_memory = in_memory

        if self.hospital not in self._allowed_hospitals:
            raise IndexError

        if self.split is not None:
            if self.split not in ['train', 'val']:
                raise KeyError

        # Read metadata
        metadata = pd.read_csv(
            os.path.join(self.root, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'}
        )

        # Keep rows of metadata specific to `hospital` & `split`
        rows_to_keep = (metadata['center'] == hospital)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['split'] == 0)  # train: 0
        elif self.split == 'val':
            rows_to_keep = rows_to_keep & (metadata['split'] == 1)  # val: 1
        else:
            pass
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files: list = [
            os.path.join(
                self.root,
                f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            ) for patient, node, x, y in
            metadata.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)
        ]
        if self.in_memory > 0:
            start = time.time()
            print(f'Loading {len(self.input_files):,} images in memory (hospital={self.hospital}, split={self.split}).', end=' ')
            self.inputs = self.load_images(self.input_files, p=self.in_memory, as_tensor=True)
            print(f'Elapsed Time: {time.time() - start:.2f} seconds.')
        else:
            self.inputs = None
        self.targets = torch.LongTensor(metadata['tumor'].values)
        self.domains = torch.LongTensor(metadata['center'].values)
        self.eval_groups = torch.LongTensor(metadata['slide'].values)
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.inputs is not None:
            return self.inputs[index]
        else:
            return read_image(self.input_files[index], mode=ImageReadMode.RGB)

    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]

    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def load_images(filenames: typing.List[str],
                    p: int,
                    as_tensor: bool = True,
                    ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
        """
        Load images with multiprocessing if p > 0.
        Arguments:
            filenames: list of filename strings.
            p: int for number of cpu threads to use for data loading.
            as_tensor: bool, returns a stacked tensor if True, a list of tensor images if False.
        Returns:
            ...
        """
        with RayPool(processes=p) as pool:
            images = pool.map(functools.partial(read_image, mode=ImageReadMode.RGB), filenames)
            pool.close(); pool.join(); time.sleep(5.0)

        return torch.stack(images, dim=0) if as_tensor else images

    @property
    def domain_indicator(self) -> int:
        return self.hospital

    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (3, 96, 96)
    
    @property
    def num_classes(self) -> int:
        return 2


class Camelyon17DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = './data/wilds/camelyon17_v1.0',
        train_domains: typing.Iterable[int] = [0, 3, 4],
        validation_domains: typing.Iterable[int] = [1],
        test_domains: typing.Iterable[int] = [2],
        batch_size: int = 32,
        num_workers: typing.Optional[int] = 4,
        prefetch_factor: typing.Optional[int] = 2,
        ) -> None:
        
        super().__init__()
        self.save_hyperparameters()
        
        # dataset arguments
        self.root = root
        self.train_domains = [int(d) for d in train_domains]
        self.validation_domains = [int(d) for d in validation_domains]
        self.test_domains = [int(d) for d in test_domains]
        
        # dataloader arguments
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def prepare_data(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError

    def setup(self, stage: typing.Optional[str] = None) -> None:
        
        # collection of datasets
        self._train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()
        self._id_test_datasets = list()
        self._ood_test_datasets = list()

        if (stage is None) or (stage == 'fit') or (stage == 'validate'):
            
            # (1) train / id-validation
            for domain in self.train_domains:
                self._train_datasets += [
                    SingleCamelyon17(root=self.root, hospital=domain, split='train')
                ]
                self._id_validation_datasets += [
                    SingleCamelyon17(root=self.root, hospital=domain, split='val')
                ]

            # (2) ood-validation
            for domain in self.validation_domains:
                self._ood_validation_datasets += [
                    SingleCamelyon17(root=self.root, hospital=domain, split=None)
                ]
        
        if (stage is None) or (stage == 'test'):            
            
            # (3) ood-test
            for domain in self.test_domains:
                self._ood_test_datasets += [
                    SingleCamelyon17(root=self.root, hospital=domain, split=None)
                ]

    def train_dataloader(self,
                         sampler: typing.Optional[torch.utils.data.Sampler] = None) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._train_datasets),
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            **self.general_loader_config
        )

    def val_dataloader(self) -> DataLoader:
        if len(self._ood_validation_datasets) > 0:
            return self._ood_val_dataloader()
        else:
            return self._id_val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._ood_test_dataloader()

    def _id_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._id_validation_datasets),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self.general_loader_config
        )

    def _ood_val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._ood_validation_datasets),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self.general_loader_config
        )

    def _id_test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def _ood_test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self._ood_test_datasets),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self.general_loader_config
        )

    @property
    def general_loader_config(self) -> typing.Dict[str, typing.Any]:
        return {
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor,
        }

    @classmethod
    def from_argparse_args(cls,
                           args: argparse.Namespace,
                           ) -> pl.LightningDataModule:
        init_arg_names = [k for k in inspect.signature(cls.__init__).parameters]
        init_kwargs = {k: v for k, v in vars(args).items() if k in init_arg_names}
        return cls(**init_kwargs)
