import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_metrics(csv_dir="model_steal/stats", save_dir="model_steal/plots"):
    os.makedirs(save_dir, exist_ok=True)

    knockoff_files = [
        "calteh_cubs_knockoff.csv",
        "cubs_cubs_knockoff.csv",
        "calteh_indoors_knockoff.csv",
        "indoors_indoors_knockoff.csv",
    ]

    baseline_files = [
        "baseline_cubs200_training.csv",
        "baseline_indoors_training.csv",
    ]

    for file_name in knockoff_files:
        file_path = os.path.join(csv_dir, file_name)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["distill_loss"], label="Train Loss", color="blue", linewidth=2)
        
        title_name = file_name.replace(".csv", "").replace("_", " ").title()
        plt.title(f"Training Loss - {title_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"{file_name.replace('.csv', '')}_plot.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    for file_name in baseline_files:
        file_path = os.path.join(csv_dir, file_name)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        fig, ax1 = plt.subplots(figsize=(9, 5))

        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color=color)
        line1 = ax1.plot(df["epoch"], df["train_loss"], color=color, label="Train Loss", linewidth=2)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Val Accuracy", color=color)
        line2 = ax2.plot(df["epoch"], df["val_accuracy"], color=color, label="Val Accuracy", linewidth=2)
        ax2.tick_params(axis="y", labelcolor=color)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        title_name = file_name.replace(".csv", "").replace("_", " ").title()
        plt.title(f"Baseline Training Metrics - {title_name}")
        fig.tight_layout()
        
        save_path = os.path.join(save_dir, f"{file_name.replace('.csv', '')}_plot.png")
        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    plot_training_metrics()