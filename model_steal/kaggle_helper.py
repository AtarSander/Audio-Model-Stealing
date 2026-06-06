import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
import torchvision.datasets as tv_datasets
from tqdm import tqdm

from model_steal.utils import train_transform, val_transform


def get_kaggle_indoors_splits(base_path, use_indexed=True):
    images_dir = os.path.join(base_path, "indoorCVPR_09", "Images")

    folder_cls = IndexedImageFolder if use_indexed else tv_datasets.ImageFolder
    full_ds_train = folder_cls(root=images_dir, transform=train_transform)
    full_ds_val = folder_cls(root=images_dir, transform=val_transform)

    num_samples = len(full_ds_train)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(0.8 * num_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    return Subset(full_ds_train, train_idx), Subset(full_ds_val, test_idx), full_ds_val # type: ignore


class IndexedImageFolder(tv_datasets.ImageFolder):
    def __getitem__(self, idx): # type: ignore
        image, label = super().__getitem__(idx)
        return image, label, idx
