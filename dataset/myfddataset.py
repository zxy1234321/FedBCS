from typing import Tuple
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from utils.conf import data_path
from dataset.utils.federated_dataset import FederatedDataset, partition_digits_domain_skew_loaders
from dataset.utils.mri_dataset import Prostate
from monai.transforms import *
from backbone.models import UNet as MRI_UNet
from backbone.models import UNet_FSR as MRI_UNet_FSR
from dataset.utils.get_date_from_src import get_datasets, get_datasets_5fold, get_mri_datasets_5fold


class TNBCDataset(FederatedDataset):
    NAME = 'tnbc'
    DOMAINS_LIST = ['tcia', 'crc', 'kirc', 'tnbc']

    N_SAMPLES_PER_Class = None
    N_CLASS = 2

    def __init__(self, args):
        super().__init__(args)
        self.train_transforms = Compose(
                [
                    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
                    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                    Resized(keys=["image", "label"], spatial_size=[256, 256]),
                    ScaleIntensityd(keys=["image"], allow_missing_keys=True),
                    EnsureTyped(keys=["image", "label"]),
                ])

        self.val_transform = Compose(
                [
                    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
                    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                    Resized(keys=["image", "label"], spatial_size=[256, 256]),
                    ScaleIntensityd(keys=["image"], allow_missing_keys=True),
                    EnsureTyped(keys=["image", "label"]),
                ])

    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        train_dataset_list = []
        val_dataset_list = []
        using_list = self.DOMAINS_LIST

        if hasattr(self.args, 'fold') and self.args.fold > 0:
            print(f"Loading 5-fold data: fold {self.args.fold}")
            datasets = get_datasets_5fold(using_list, self.NAME, self.args.fold,
                                          self.train_transforms, self.val_transform)
        else:
            datasets = get_datasets(using_list, self.args.source_key, self.NAME,
                                    self.train_transforms, self.val_transform)

        for domain in using_list:
            train_dataset = datasets[domain]["train"]
            val_dataset = datasets[domain]["val"]
            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)

        traindls, testdls = partition_digits_domain_skew_loaders(train_dataset_list, val_dataset_list, self)
        return traindls, testdls

    @staticmethod
    def get_backbone(parti_num, names_list, args, channel_ratio=0, mode='ori') -> Module:
        net_list = []
        for j in range(parti_num):
            arch = names_list[j]
            if arch == 'fsr':
                net_list.append(MRI_UNet_FSR())
            elif arch == 'unet':
                net_list.append(MRI_UNet())
            else:
                raise ValueError(f"Unknown model: {arch}")
        return net_list


class MRIDataset(FederatedDataset):
    NAME = 'mri'
    SETTING = 'mri'
    DOMAINS_LIST = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']

    N_SAMPLES_PER_Class = None
    N_CLASS = 2

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        using_list = self.DOMAINS_LIST
        print(f'using_list: {using_list}')

        if hasattr(self.args, 'fold') and self.args.fold > 0:
            print(f"Loading MRI 5-fold data: fold {self.args.fold}")
            train_dataset_list, val_dataset_list = get_mri_datasets_5fold(using_list, self.args.fold)
        else:
            train_dataset_list = []
            val_dataset_list = []
        for domain in using_list:
            train_dataset = Prostate(site=domain, base_path="./data/MRI", split='train')
            test_dataset = Prostate(site=domain, base_path="./data/MRI", split='test')
            train_dataset_list.append(train_dataset)
            val_dataset_list.append(test_dataset)

        traindls, testdls = partition_digits_domain_skew_loaders(train_dataset_list, val_dataset_list, self)
        print(f'len(traindls): {len(traindls)}')
        print(f'len(testdls): {len(testdls)}')
        return traindls, testdls

    @staticmethod
    def get_backbone(parti_num, names_list, args, channel_ratio=0, mode='ori') -> Module:
        net_list = []
        for j in range(parti_num):
            arch = names_list[j]
            if arch == 'fsr':
                net_list.append(MRI_UNet_FSR())
            elif arch == 'unet':
                net_list.append(MRI_UNet())
            else:
                raise ValueError(f"Unknown model: {arch}")
        return net_list


