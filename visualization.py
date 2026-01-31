from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


def _save_or_show(fig, save_path: str | Path | None, show: bool = True):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)



def plot_original_corrupted(clean_ds, corrupt_ds, n=4, save_path: str | Path | None = None):  # NEW ADDITION
    """Visualize original and corrupted samples side by side."""
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        clean_img, _ = clean_ds[i]
        corr_img, _ = corrupt_ds[i]
        axes[0, i].imshow(clean_img.squeeze(), cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].imshow(corr_img.squeeze(), cmap="gray")
        axes[1, i].set_title("Corrupted")
        axes[1, i].axis("off")
    _save_or_show(fig, save_path, show=True)


def plot_reconstructions(
    model,
    clean_ds,
    corrupt_ds,
    device="cuda",
    n=4,
    save_path: str | Path | None = None,
):
    """Visualize reconstructions for clean and corrupted inputs."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i in range(n):
        clean_img, _ = clean_ds[i]
        corr_img, _ = corrupt_ds[i]

        with torch.no_grad():
            clean_out = model(clean_img.unsqueeze(0).to(device))
            corr_out = model(corr_img.unsqueeze(0).to(device))

        axes[0, i].imshow(clean_img.squeeze(), cmap="gray")
        axes[0, i].set_title("Clean")
        axes[0, i].axis("off")

        axes[1, i].imshow(corr_img.squeeze(), cmap="gray")
        axes[1, i].set_title("Corrupted")
        axes[1, i].axis("off")

        out = corr_out if corr_out.shape == corr_img.unsqueeze(0).shape else corr_out.view(1, 1, 28, 28)
        axes[2, i].imshow(out.squeeze().cpu(), cmap="gray")
        axes[2, i].set_title("Reconstruction")
        axes[2, i].axis("off")

    _save_or_show(fig, save_path)


def plot_loss_curves(
    history: dict,
    title: str = "Loss Curves",
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    _save_or_show(fig, save_path)


def plot_loss_comparison( 
    mse_history: dict,
    ssim_history: dict,
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mse_history.get("train_loss", []), label="MSE Train")
    ax.plot(mse_history.get("val_loss", []), label="MSE Val")
    ax.plot(ssim_history.get("train_loss", []), label="SSIM Train")
    ax.plot(ssim_history.get("val_loss", []), label="SSIM Val")
    ax.set_title("MSE vs SSIM Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    _save_or_show(fig, save_path)


def plot_error_distribution(
    errors_clean: Iterable[float],
    errors_corrupt: Iterable[float],
    threshold: float | None = None,
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors_clean, bins=50, alpha=0.6, label="Clean")
    ax.hist(errors_corrupt, bins=50, alpha=0.6, label="Corrupted")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.4f}")
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Count")
    ax.legend()
    _save_or_show(fig, save_path, show=True)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels=("Clean", "Corrupted"),
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    _save_or_show(fig, save_path, show=True)
