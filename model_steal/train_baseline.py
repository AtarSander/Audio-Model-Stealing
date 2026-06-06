import os

from datasets import load_dataset
import kagglehub
from loguru import logger
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_steal.huggingface_utils import HFDatasetWrapper
from model_steal.kaggle_helper import get_kaggle_indoors_splits
from model_steal.utils import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    get_resnet34,
    train_transform,
    val_transform,
)


def train_baseline(model, train_loader, val_loader, epochs, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        logger.info(
            f"Epoch {epoch + 1} Complete. Train Loss: {running_loss / len(train_loader.dataset):.4f} | Val Acc: {acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("model_steal/models", exist_ok=True)

    indoors_path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")
    indoors_train, indoors_test, _ = get_kaggle_indoors_splits(indoors_path, use_indexed=True)
    cubs_raw = load_dataset("cassiekang/cub200_dataset")

    cubs_train_ds = HFDatasetWrapper(cubs_raw["train"], transform=train_transform)
    cubs_test_ds = HFDatasetWrapper(cubs_raw["test"], transform=val_transform)

    indoors_train_loader = DataLoader(
        indoors_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    indoors_val_loader = DataLoader(
        indoors_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    cubs_train_loader = DataLoader(
        cubs_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    cubs_val_loader = DataLoader(
        cubs_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    indoors_model = get_resnet34(num_classes=67)
    train_baseline(
        indoors_model,
        indoors_train_loader,
        indoors_val_loader,
        EPOCHS,
        "model_steal/models/baseline_indoors.pt",
    )

    cubs_model = get_resnet34(num_classes=200)
    train_baseline(
        cubs_model,
        cubs_train_loader,
        cubs_val_loader,
        EPOCHS,
        "model_steal/models/baseline_cubs.pt",
    )
