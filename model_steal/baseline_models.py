import os

from datasets import load_dataset
import kagglehub
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import models, transforms
from tqdm import tqdm

# --- Configuration & Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Standard ImageNet Transforms for ResNet
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


# --- Custom HF Wrapper Dataset ---
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


# --- Custom PyTorch Folder Wrapper with Index Tracking ---
class IndexedImageFolder(tv_datasets.ImageFolder):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, label, idx


# --- Train / Test Split Maker for Kaggle Folder ---
def get_kaggle_indoors_datasets(base_path):
    # Dataset structure: path/Images/class_name/xxx.jpg
    images_dir = os.path.join(base_path, "indoorCVPR_09", "Images")

    # We create two distinct folder instances to apply different transforms safely
    full_ds_train = IndexedImageFolder(root=images_dir, transform=train_transform)
    full_ds_val = IndexedImageFolder(root=images_dir, transform=val_transform)

    # Stratified or randomized deterministic split (80% train, 20% test)
    num_samples = len(full_ds_train)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(0.8 * num_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    return Subset(full_ds_train, train_idx), Subset(full_ds_val, test_idx), full_ds_val


# --- Model Helper ---
def get_resnet34(num_classes: int):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# --- Common Training Loop ---
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

        # Validation
        model.eval()
        correct = 0
        total = 0
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
            logger.info(f"Saved new best model checkpoint to {save_path}")


# --- Generate Logits Function ---
@torch.no_grad()
def extract_logits(model, dataloader, num_samples, num_classes):
    model.eval()
    logit_matrix = np.zeros((num_samples, num_classes), dtype=np.float32)
    for images, _, indices in tqdm(dataloader, desc="Extracting Logits"):
        images = images.to(DEVICE)
        outputs = model(images)
        logit_matrix[indices.numpy()] = outputs.cpu().numpy()
    return logit_matrix


if __name__ == "__main__":
    os.makedirs("model_steal/models", exist_ok=True)
    os.makedirs("model_steal/logits", exist_ok=True)

    # 1. Download and Prepare Kaggle Data & HF Data
    logger.info("Downloading Indoor Scenes from Kaggle Hub...")
    indoors_path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")
    logger.info(f"Kaggle Indoors dataset path: {indoors_path}")
    indoors_train, indoors_test, indoors_full_seq = get_kaggle_indoors_datasets(indoors_path)
    logger.info("Loading remaining datasets from Hugging Face Hub...")
    cubs_raw = load_dataset("cassiekang/cub200_dataset")
    caltech_raw = load_dataset("ilee0022/Caltech-256")

    cubs_train_ds = HFDatasetWrapper(cubs_raw["train"], transform=train_transform)
    cubs_test_ds = HFDatasetWrapper(cubs_raw["test"], transform=val_transform)
    caltech_all_ds = HFDatasetWrapper(caltech_raw["train"], transform=val_transform)

    # Loaders
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

    # Sequential order extraction loaders for soft targets
    # Notice we pass the full *undistorted train subset wrapper* for target logit matrix consistency
    indoors_seq_loader = DataLoader(
        IndexedImageFolder(
            root=os.path.join(indoors_path, "indoorCVPR_09", "Images"), transform=val_transform
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    cubs_seq_loader = DataLoader(
        cubs_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    caltech_seq_loader = DataLoader(
        caltech_all_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 2. Train Baselines
    logger.info("--- Training MIT Indoors Baseline (Kaggle Edition) ---")
    indoors_model = get_resnet34(num_classes=67)
    train_baseline(
        indoors_model,
        indoors_train_loader,
        indoors_val_loader,
        EPOCHS,
        "model_steal/models/baseline_indoors.pt",
    )

    logger.info("--- Training CUBS200 Baseline ---")
    cubs_model = get_resnet34(num_classes=200)
    train_baseline(
        cubs_model, cubs_train_loader, cubs_val_loader, EPOCHS, "model_steal/models/baseline_cubs.pt"
    )

    # 3. Generate and Save Logits
    indoors_model.load_state_dict(torch.load("model_steal/models/baseline_indoors.pt"))
    cubs_model.load_state_dict(torch.load("model_steal/models/baseline_cubs.pt"))

    logger.info("--- Generating Logit Target Repositories ---")

    # Save Indoors baseline predictions on everything
    # We extract logits for the entire Indoors directory to map correctly during soft-distillation steps
    np.save(
        "model_steal/logits/indoors_on_indoors.npy",
        extract_logits(indoors_model, indoors_seq_loader, len(indoors_full_seq), 67),
    )
    np.save(
        "model_steal/logits/indoors_on_caltech.npy",
        extract_logits(indoors_model, caltech_seq_loader, len(caltech_all_ds), 67),
    )

    # Save CUBS baseline predictions on everything
    np.save(
        "model_steal/logits/cubs_on_cubs.npy",
        extract_logits(cubs_model, cubs_seq_loader, len(cubs_train_ds), 200),
    )
    np.save(
        "model_steal/logits/cubs_on_caltech.npy",
        extract_logits(cubs_model, caltech_seq_loader, len(caltech_all_ds), 200),
    )

    logger.success("Phase 1 & 2 Complete! Logits and models saved safely.")
