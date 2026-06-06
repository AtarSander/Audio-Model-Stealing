import os
from datasets import load_dataset
import kagglehub
from loguru import logger
from model_steal.baseline_models import HFDatasetWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import models, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42

# Ensure match with pipeline splits
np.random.seed(SEED)

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


# --- Custom Soft Target Dataset (HF Variant) ---
class LogitDistillationHFDataset(Dataset):
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


# --- Custom Soft Target Dataset (Local Folder Variant) ---
class LogitDistillationFolderDataset(Dataset):
    def __init__(self, folder_dataset, logit_path):
        self.dataset = folder_dataset  # This will receive the Split Subset
        self.logits = np.load(logit_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Resolve real structural index from Subset layout wrapper
        actual_idx = self.dataset.indices[idx] if isinstance(self.dataset, Subset) else idx
        image, _ = self.dataset[idx]
        soft_target = torch.tensor(self.logits[actual_idx], dtype=torch.float32)
        return image, soft_target


class SoftTargetLoss(nn.Module):
    def __init__(self):
        super(SoftTargetLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.mse(outputs, targets)


def train_distilled_model(model, train_loader, epochs, save_path):
    criterion = SoftTargetLoss()
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


def evaluate_model(model, test_loader, name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = (correct / total) * 100
    logger.success(f"Accuracy of model [{name}]: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    logger.info("Resolving Kaggle and HF cache locations...")
    indoors_path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")
    images_dir = os.path.join(indoors_path, "indoorCVPR_09", "Images")

    # Re-verify deterministic train/test subset splits
    full_ds_train = tv_datasets.ImageFolder(root=images_dir, transform=train_transform)
    full_ds_val = tv_datasets.ImageFolder(root=images_dir, transform=val_transform)

    num_samples = len(full_ds_train)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(0.8 * num_samples)
    indoors_train_subset = Subset(full_ds_train, indices[:split])
    indoors_test_subset = Subset(full_ds_val, indices[split:])

    cubs_raw = load_dataset("cassiekang/cub200_dataset")
    caltech_raw = load_dataset("ilee0022/Caltech-256")

    # 1. Prepare Distillation Datasets
    ds_indoors_soft = LogitDistillationFolderDataset(
        indoors_train_subset, "model_steal/logits/indoors_on_indoors.npy"
    )
    ds_cubs_soft = LogitDistillationHFDataset(
        cubs_raw["train"], "model_steal/logits/cubs_on_cubs.npy", train_transform
    )

    ds_caltech_indoors_soft = LogitDistillationHFDataset(
        caltech_raw["train"], "model_steal/logits/indoors_on_caltech.npy", train_transform
    )
    ds_caltech_cubs_soft = LogitDistillationHFDataset(
        caltech_raw["train"], "model_steal/logits/cubs_on_caltech.npy", train_transform
    )

    # Loaders
    loader_indoors_soft = DataLoader(ds_indoors_soft, batch_size=BATCH_SIZE, shuffle=True)
    loader_cubs_soft = DataLoader(ds_cubs_soft, batch_size=BATCH_SIZE, shuffle=True)
    loader_caltech_indoors_soft = DataLoader(
        ds_caltech_indoors_soft, batch_size=BATCH_SIZE, shuffle=True
    )
    loader_caltech_cubs_soft = DataLoader(
        ds_caltech_cubs_soft, batch_size=BATCH_SIZE, shuffle=True
    )

    # 2. Train the 4 Distilled Models
    logger.info("Training Model 1: Indoors dataset -> Indoors soft targets")
    m1 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    m1.fc = nn.Linear(m1.fc.in_features, 67)
    m1.to(DEVICE)
    # train_distilled_model(m1, loader_indoors_soft, EPOCHS, "model_steal/models/distilled_indoors_on_indoors.pt")

    logger.info("Training Model 2: CUBS dataset -> CUBS soft targets")
    m2 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    m2.fc = nn.Linear(m2.fc.in_features, 200)
    m2.to(DEVICE)
    # train_distilled_model(m2, loader_cubs_soft, EPOCHS, "model_steal/models/distilled_cubs_on_cubs.pt")

    logger.info("Training Model 3: Caltech256 dataset -> Indoors soft targets (67 classes)")
    m3 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    m3.fc = nn.Linear(m3.fc.in_features, 67)
    m3.to(DEVICE)
    # train_distilled_model(m3, loader_caltech_indoors_soft, EPOCHS, "model_steal/models/distilled_caltech_on_indoors.pt")

    logger.info("Training Model 4: Caltech256 dataset -> CUBS soft targets (200 classes)")
    m4 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    m4.fc = nn.Linear(m4.fc.in_features, 200)
    m4.to(DEVICE)
    # train_distilled_model(m4, loader_caltech_cubs_soft, EPOCHS, "model_steal/models/distilled_caltech_on_cubs.pt")

    # 3. Final Evaluation Suite
    logger.info("--- Starting Universal Evaluation Matrix ---")

    class StandardHFEvalDS(HFDatasetWrapper):
        def __getitem__(self, idx):
            a, b, c = super().__getitem__(idx)
            return a, b

    indoors_test_loader = DataLoader(indoors_test_subset, batch_size=BATCH_SIZE)
    cubs_test_loader = DataLoader(
        StandardHFEvalDS(cubs_raw["test"], val_transform), batch_size=BATCH_SIZE
    )
    m1.load_state_dict(torch.load("model_steal/models/distilled_indoors_on_indoors.pt"))
    m2.load_state_dict(torch.load("model_steal/models/distilled_cubs_on_cubs.pt"))
    m3.load_state_dict(torch.load("model_steal/models/distilled_caltech_on_indoors.pt"))
    m4.load_state_dict(torch.load("model_steal/models/distilled_caltech_on_cubs.pt"))
    # Load Baseline Models
    base_indoors = models.resnet34()
    base_indoors.fc = nn.Linear(base_indoors.fc.in_features, 67)
    base_indoors.load_state_dict(torch.load("model_steal/models/baseline_indoors.pt"))
    base_indoors.to(DEVICE)
    base_cubs = models.resnet34()
    base_cubs.fc = nn.Linear(base_cubs.fc.in_features, 200)
    base_cubs.load_state_dict(torch.load("model_steal/models/baseline_cubs.pt"))
    base_cubs.to(DEVICE)

    # Process evaluation suite execution matrix
    evaluate_model(base_indoors, indoors_test_loader, "Baseline MIT Indoors (Kaggle)")
    evaluate_model(m1, indoors_test_loader, "Distilled Indoors-on-Indoors")
    evaluate_model(m3, indoors_test_loader, "Distilled Caltech-on-Indoors")

    evaluate_model(base_cubs, cubs_test_loader, "Baseline CUBS200")
    evaluate_model(m2, cubs_test_loader, "Distilled CUBS-on-CUBS")
    evaluate_model(m4, cubs_test_loader, "Distilled Caltech-on-CUBS")
