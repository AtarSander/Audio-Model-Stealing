import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        all_labels = sorted({d["text"] for d in self.dataset})
        self.label_to_text = {i: t for i, t in enumerate(all_labels)}
        self.text_to_label = {t: i for i, t in enumerate(all_labels)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = self.text_to_label[item["text"]]
        if self.transform:
            image = self.transform(image)
        return image, label, idx


class HFDatasetWrapperValidation(HFDatasetWrapper):
    def __getitem__(self, idx): # type: ignore
        image, label, _ = super().__getitem__(idx)
        return image, label


class HFDatasetForLogits(Dataset):
    def __init__(self, hf_dataset, logit_path, transform=None):
        self.dataset = hf_dataset
        self.logits = np.load(logit_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        soft_target = torch.tensor(self.logits[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, soft_target