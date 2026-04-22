from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim
import random
from config_examples.training_example import config


def _worker_init_fn(worker_id):
    random.seed(1337 + worker_id)


class FederatedDataset:
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loaders = []
        self.test_loader = []
        self.args = args
        self.config = config

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list) -> nn.Module:
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
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


def partition_digits_domain_skew_loaders(train_datasets: list, test_datasets: list,
                                         setting: FederatedDataset) -> Tuple[list, list]:
    for index in range(len(train_datasets)):
        train_dataset = train_datasets[index]
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size,
                             num_workers=0, pin_memory=True, worker_init_fn=_worker_init_fn)
        setting.train_loaders.append(train_loader)

    for index in range(len(test_datasets)):
        test_dataset = test_datasets[index]
        test_loader = DataLoader(test_dataset,
                                 batch_size=16, shuffle=False, num_workers=0)
        setting.test_loader.append(test_loader)

    return setting.train_loaders, setting.test_loader
