import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
SEED = 42


def reset_seeds(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class LogitsDataset(Dataset):
    def __init__(self, folder_dataset, logit_path):
        self.dataset = folder_dataset
        self.logits = np.load(logit_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        actual_idx = self.dataset.indices[idx] if isinstance(self.dataset, Subset) else idx
        image, _ = self.dataset[idx]  # type: ignore
        soft_target = torch.tensor(self.logits[actual_idx], dtype=torch.float32)
        return image, soft_target


def get_resnet34(num_classes: int):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


