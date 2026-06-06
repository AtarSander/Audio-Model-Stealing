import os

from datasets import load_dataset
import kagglehub
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader

from model_steal.huggingface_utils import HFDatasetWrapperValidation
from model_steal.kaggle_helper import get_kaggle_indoors_splits
from model_steal.utils import DEVICE
from utils import (
    BATCH_SIZE,
    get_resnet34,
    reset_seeds,
    val_transform,
)


def evaluate_model(model, test_loader, name, save_dir="model_steal/metrics"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds) * 100
    logger.success(f"Accuracy of model [{name}]: {acc:.2f}%")

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    logger.info(f"[{name}] Macro Precision: {report['macro avg']['precision']:.4f}") # type: ignore
    logger.info(f"[{name}] Macro Recall: {report['macro avg']['recall']:.4f}") # type: ignore
    logger.info(f"[{name}] Macro F1-Score: {report['macro avg']['f1-score']:.4f}") # type: ignore

    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_report.txt"), "w") as f:
        f.write(classification_report(all_labels, all_preds, zero_division=0)) # type: ignore

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    
    if len(np.unique(all_labels)) <= 10:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    else:
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
        
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(save_dir, f"{safe_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    return acc


if __name__ == "__main__":
    reset_seeds()
    indoors_path = kagglehub.dataset_download("itsahmad/indoor-scenes-cvpr-2019")

    _, indoors_test_subset, _ = get_kaggle_indoors_splits(
        indoors_path, use_indexed=False
    )

    cubs_raw = load_dataset("cassiekang/cub200_dataset")

    indoors_test_loader = DataLoader(indoors_test_subset, batch_size=BATCH_SIZE)
    cubs_test_loader = DataLoader(
        HFDatasetWrapperValidation(cubs_raw["test"], val_transform), batch_size=BATCH_SIZE
    )

    m1 = get_resnet34(67)
    m1.load_state_dict(torch.load("model_steal/models/distilled_indoors_on_indoors.pt"))
    m1.to(DEVICE)

    m2 = get_resnet34(200)
    m2.load_state_dict(torch.load("model_steal/models/distilled_cubs_on_cubs.pt"))
    m2.to(DEVICE)

    m3 = get_resnet34(67)
    m3.load_state_dict(torch.load("model_steal/models/distilled_caltech_on_indoors.pt"))
    m3.to(DEVICE)

    m4 = get_resnet34(200)
    m4.load_state_dict(torch.load("model_steal/models/distilled_caltech_on_cubs.pt"))
    m4.to(DEVICE)

    base_indoors = get_resnet34(67)
    base_indoors.load_state_dict(torch.load("model_steal/models/baseline_indoors.pt"))
    base_indoors.to(DEVICE)

    base_cubs = get_resnet34(200)
    base_cubs.load_state_dict(torch.load("model_steal/models/baseline_cubs.pt"))
    base_cubs.to(DEVICE)

    evaluate_model(base_indoors, indoors_test_loader, "Baseline MIT Indoors (Kaggle)")
    evaluate_model(m1, indoors_test_loader, "Distilled Indoors-on-Indoors")
    evaluate_model(m3, indoors_test_loader, "Distilled Caltech-on-Indoors")

    evaluate_model(base_cubs, cubs_test_loader, "Baseline CUBS200")
    evaluate_model(m2, cubs_test_loader, "Distilled CUBS-on-CUBS")
    evaluate_model(m4, cubs_test_loader, "Distilled Caltech-on-CUBS")