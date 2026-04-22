import sys
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import SimpleITK as sitk
import random
import cv2
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])


def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=transform, fold=0, fold_file=None):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'ISBI': 3, 'ISBI_1.5': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split

        base_path = base_path if base_path is not None else './data/MRI'

        all_images, all_labels, case_indices = [], [], {}
        sitedir = os.path.join(base_path, site)

        ossitedir = np.load(os.path.join(base_path, "{}-dir.npy".format(site))).tolist()

        current_idx = 0
        for sample in ossitedir:
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                case_name = sample[:6]
                imgdir = os.path.join(sitedir, case_name + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                image_v = convert_from_nii_to_png(image_v)

                case_start_idx = len(all_images)
                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if (np.all(label == 0)):
                        continue
                    image = np.array(image_v[i - 1:i + 2, :, :])
                    image = np.transpose(image, (1, 2, 0))

                    all_labels.append(label)
                    all_images.append(image)
                case_end_idx = len(all_images)

                if case_name not in case_indices:
                    case_indices[case_name] = (case_start_idx, case_end_idx)

        all_labels = np.array(all_labels).astype(int)
        all_images = np.array(all_images)

        if fold > 0 and fold_file is not None:
            with open(fold_file, 'r') as f:
                fold_cases = [line.strip() for line in f if line.strip()]

            selected_indices = []
            for case_name in fold_cases:
                if case_name in case_indices:
                    start, end = case_indices[case_name]
                    selected_indices.extend(range(start, end))

            self.images = all_images[selected_indices]
            self.labels = all_labels[selected_indices]
        else:
            index = np.load(os.path.join(base_path, "{}-index.npy".format(site))).tolist()
            all_labels = all_labels[index]
            all_images = all_images[index]

            trainlen = 0.8 * len(all_labels) * 0.8
            vallen = 0.8 * len(all_labels) - trainlen
            testlen = 0.2 * len(all_labels)

            if (split == 'train'):
                self.images, self.labels = all_images[:int(trainlen)], all_labels[:int(trainlen)]
            elif (split == 'val'):
                self.images, self.labels = all_images[int(trainlen):int(trainlen + vallen)], all_labels[int(trainlen):int(trainlen + vallen)]
            else:
                self.images, self.labels = all_images[int(trainlen + vallen):], all_labels[int(trainlen + vallen):]

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image, (2, 0, 1))
            image = torch.Tensor(image)

            label = self.transform(label)
        sample = {'image': image, 'label': label}
        return sample


class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask
