from datasets import load_dataset
import kagglehub
from loguru import logger
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_steal.huggingface_utils import HFDatasetForLogits
from model_steal.kaggle_helper import get_kaggle_indoors_splits
from model_steal.utils import DEVICE, LEARNING_RATE
from utils import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LogitsDataset,
    get_resnet34,
    reset_seeds,
    train_transform,
)


def train_knock_off(model, train_loader, epochs, save_path):
    criterion = KnockOffLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Distilling Epoch {epoch + 1}/{epochs}"):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        logger.info(f"Loss: {running_loss / len(train_loader.dataset):.4f}")
    torch.save(model.state_dict(), save_path)


class KnockOffLoss(nn.Module):
    def __init__(self):
        super(KnockOffLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.mse(outputs, targets)

if __name__ == "__main__":
    reset_seeds()
    indoors_path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")

    indoors_train_subset, _, _ = get_kaggle_indoors_splits(
        indoors_path, use_indexed=False
    )

    cubs_raw = load_dataset("cassiekang/cub200_dataset")
    caltech_raw = load_dataset("ilee0022/Caltech-256")

    ds_indoors_soft = LogitsDataset(
        indoors_train_subset, "model_steal/logits/indoors_on_indoors.npy"
    )
    ds_cubs_soft = HFDatasetForLogits(
        cubs_raw["train"], "model_steal/logits/cubs_on_cubs.npy", train_transform
    )
    ds_caltech_indoors_soft = HFDatasetForLogits(
        caltech_raw["train"], "model_steal/logits/indoors_on_caltech.npy", train_transform
    )
    ds_caltech_cubs_soft = HFDatasetForLogits(
        caltech_raw["train"], "model_steal/logits/cubs_on_caltech.npy", train_transform
    )

    loader_indoors_soft = DataLoader(ds_indoors_soft, batch_size=BATCH_SIZE, shuffle=True)
    loader_cubs_soft = DataLoader(ds_cubs_soft, batch_size=BATCH_SIZE, shuffle=True)
    loader_caltech_indoors_soft = DataLoader(
        ds_caltech_indoors_soft, batch_size=BATCH_SIZE, shuffle=True
    )
    loader_caltech_cubs_soft = DataLoader(
        ds_caltech_cubs_soft, batch_size=BATCH_SIZE, shuffle=True
    )

    m1 = get_resnet34(num_classes=67)
    m1.to(DEVICE)
    train_knock_off(m1, loader_indoors_soft, EPOCHS, "model_steal/models/distilled_indoors_on_indoors.pt")

    m2 = get_resnet34(num_classes=200)
    m2.to(DEVICE)
    train_knock_off(m2, loader_cubs_soft, EPOCHS, "model_steal/models/distilled_cubs_on_cubs.pt")

    m3 = get_resnet34(num_classes=67)
    m3.to(DEVICE)
    train_knock_off(m3, loader_caltech_indoors_soft, EPOCHS, "model_steal/models/distilled_caltech_on_indoors.pt")

    m4 = get_resnet34(num_classes=200)
    m4.to(DEVICE)
    train_knock_off(m4, loader_caltech_cubs_soft, EPOCHS, "model_steal/models/distilled_caltech_on_cubs.pt")