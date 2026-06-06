import os

from datasets import load_dataset
import kagglehub
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_steal.huggingface_utils import HFDatasetWrapper
from model_steal.kaggle_helper import IndexedImageFolder, get_kaggle_indoors_splits
from model_steal.utils import (
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    get_resnet34,
    reset_seeds,
    val_transform,
)


@torch.no_grad()
def infer_logits_for_baseline(model, dataloader, num_samples, num_classes):
    model.eval()
    logit_matrix = np.zeros((num_samples, num_classes), dtype=np.float32)
    for images, _, indices in tqdm(dataloader, desc="Extracting Logits"):
        images = images.to(DEVICE)
        outputs = model(images)
        logit_matrix[indices.numpy()] = outputs.cpu().numpy()
    return logit_matrix


if __name__ == "__main__":
    reset_seeds()
    os.makedirs("model_steal/logits", exist_ok=True)

    indoors_path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")
    _, _, indoors_full_seq = get_kaggle_indoors_splits(indoors_path, use_indexed=True)
    cubs_raw = load_dataset("cassiekang/cub200_dataset")
    caltech_raw = load_dataset("ilee0022/Caltech-256")

    cubs_train_ds = HFDatasetWrapper(cubs_raw["train"], transform=val_transform)
    caltech_all_ds = HFDatasetWrapper(caltech_raw["train"], transform=val_transform)

    indoors_seq_loader = DataLoader(
        IndexedImageFolder(root=os.path.join(indoors_path, "indoorCVPR_09", "Images"), transform=val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    cubs_seq_loader = DataLoader(cubs_train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    caltech_seq_loader = DataLoader(caltech_all_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    indoors_model = get_resnet34(num_classes=67)
    indoors_model.load_state_dict(torch.load("model_steal/models/baseline_indoors.pt"))

    cubs_model = get_resnet34(num_classes=200)
    cubs_model.load_state_dict(torch.load("model_steal/models/baseline_cubs.pt"))

    np.save("model_steal/logits/indoors_on_indoors.npy", infer_logits_for_baseline(indoors_model, indoors_seq_loader, len(indoors_full_seq), 67))
    np.save("model_steal/logits/indoors_on_caltech.npy", infer_logits_for_baseline(indoors_model, caltech_seq_loader, len(caltech_all_ds), 67))
    np.save("model_steal/logits/cubs_on_cubs.npy", infer_logits_for_baseline(cubs_model, cubs_seq_loader, len(cubs_train_ds), 200))
    np.save("model_steal/logits/cubs_on_caltech.npy", infer_logits_for_baseline(cubs_model, caltech_seq_loader, len(caltech_all_ds), 200))