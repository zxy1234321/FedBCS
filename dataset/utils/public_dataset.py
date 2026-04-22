from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim


class PublicDataset:
    NAME = None
    SETTING = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loader = None
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> DataLoader:
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


def random_loaders(train_dataset: datasets,
                   setting: PublicDataset) -> DataLoader:
    public_scale = setting.args.public_len
    y_train = train_dataset.targets
    n_train = len(y_train)
    idxs = np.random.permutation(n_train)
    if public_scale != None:
        idxs = idxs[0:public_scale]
    train_sampler = SubsetRandomSampler(idxs)
    train_loader = DataLoader(train_dataset, batch_size=setting.args.public_batch_size, sampler=train_sampler, num_workers=4)
    setting.train_loader = train_loader

    return setting.train_loader
