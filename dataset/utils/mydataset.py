import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler


class FedDataset(Dataset):
    def __init__(self, base_dir, labeled_file, unlabeled_file=None, split='train', transform=None, train_num=None):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        with open(labeled_file, 'r') as f:
            self.labeled_data = [l.split() for l in f.readlines()]
        if train_num is not None:
            self.labeled_data = self.labeled_data[:train_num]
        self.num_labeled = len(self.labeled_data)
        if (unlabeled_file is not None) and (split == 'train'):
            with open(unlabeled_file, 'r') as f:
                self.unlabeled_data = [l.split() for l in f.readlines()]
            self.num_unlabeled = len(self.unlabeled_data)
            self.all_data = self.labeled_data + self.unlabeled_data
        else:
            self.all_data = self.labeled_data
            self.num_unlabeled = 0
        print(f'total: {len(self.all_data)} samples,labeled:{self.num_labeled},unlabeled:{self.num_unlabeled}')

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        case = self.all_data[idx]
        if (self.split == 'train') and (self.num_unlabeled > 0):
            if (idx < self.num_labeled):
                image = cv2.imread(os.path.join(self.base_dir, case[0]), cv2.IMREAD_UNCHANGED)
                label = cv2.imread(os.path.join(self.base_dir, case[1]), cv2.IMREAD_UNCHANGED)
                if image is None or label is None:
                    raise ValueError(f"Error reading image or label at index {idx}: {case}")
                sample = {"image": image, "label": label}
            else:
                image = cv2.imread(os.path.join(self.base_dir, case[0]), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Error reading image at index {idx}: {case}")
                sample = {"image": image, "label": np.zeros_like(image[:, :, 0])}
        else:
            image = cv2.imread(os.path.join(self.base_dir, case[0]), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Error reading image at index {idx}: {case}")
            image = image[:, :, :3]
            label = cv2.imread(os.path.join(self.base_dir, case[1]), cv2.IMREAD_UNCHANGED)
            label[label > 0] = 1
            sample = {"image": image, "label": label}
        sample = self.transform(sample)
        sample["idx"] = os.path.basename(case[0])
        return sample


