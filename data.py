from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

SEED = 42


class TransformDataset(Dataset):
    """
    Wraps a base dataset and applies the necessary transformations.
    """

    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def load_mnist(data_dir: str | Path = "data") -> Tuple[MNIST, MNIST]:
    """Load MNIST train/test datasets."""
    data_dir = str(data_dir)
    mnist_train = MNIST(data_dir, train=True, download=True)
    mnist_test = MNIST(data_dir, train=False, download=True)
    return mnist_train, mnist_test


def stratified_indices(labels: torch.Tensor, val_frac: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create stratified train/val indices."""
    rng = torch.Generator().manual_seed(SEED)
    train_idx_list = []
    val_idx_list = []

    classes = torch.unique(labels)
    for c in classes:
        idx_c = torch.where(labels == c)[0]
        idx_c = idx_c[torch.randperm(len(idx_c), generator=rng)]
        n_val_c = int(round(len(idx_c) * val_frac))
        val_idx_list.append(idx_c[:n_val_c])
        train_idx_list.append(idx_c[n_val_c:])

    train_idx = torch.cat(train_idx_list)
    val_idx = torch.cat(val_idx_list)
    train_idx = train_idx[torch.randperm(len(train_idx), generator=rng)]
    val_idx = val_idx[torch.randperm(len(val_idx), generator=rng)]
    return train_idx, val_idx


def build_dataloaders(
    base_dataset,
    train_idx,
    val_idx,
    batch_size: int = 32,
    transform_train=None,
    transform_val=None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    # Two dataset "views" with different transforms
    train_dataset = TransformDataset(base_dataset, transform=transform_train)
    val_dataset = TransformDataset(base_dataset, transform=transform_val)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader


def default_transforms():
    transform_train = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ]
    )
    return transform_train, transform_val


def corrupt_images(
    images: torch.Tensor,
    noise_type: str = "gaussian",
    noise_level: float = 0.8,
) -> torch.Tensor:
    """Add noise to images (expects N x 1 x 28 x 28)."""
    if noise_type == "gaussian":
        noise = torch.randn_like(images) * noise_level
        corrupted = images + noise
    elif noise_type == "salt_pepper":
        corrupted = images.clone()
        probs = torch.rand_like(images)
        corrupted[probs < (noise_level / 2)] = 0.0
        corrupted[probs > (1 - noise_level / 2)] = 1.0
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    return torch.clamp(corrupted, 0.0, 1.0)


def make_corrupted_subset(
    test_dataset: MNIST,
    fraction: float = 0.1,
    noise_type: str = "gaussian",
    noise_level: float = 0.8,
) -> Tuple[TensorDataset, TensorDataset]:
    """Take a subset of MNIST test set and return (clean, corrupted) datasets."""
    n_total = len(test_dataset)
    n_subset = max(1, int(math.ceil(n_total * fraction)))
    rng = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n_total, generator=rng)[:n_subset]

    images = test_dataset.data[indices].unsqueeze(1).float() / 255.0
    labels = test_dataset.targets[indices].long()
    corrupted = corrupt_images(images, noise_type=noise_type, noise_level=noise_level)

    clean_ds = TensorDataset(images, labels)
    corrupt_ds = TensorDataset(corrupted, labels)
    return clean_ds, corrupt_ds


# Data loader for corrupted MNIST from disk (user-provided)
DATA_PATH = "data/corruptmnist"


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = TensorDataset(train_images, train_target)
    test_set = TensorDataset(test_images, test_target)

    return train_set, test_set


def corrupt_mnist_placeholder():  # NEW ADDITION
    """Return random tensors (placeholder version of corrupt_mnist)."""
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784)
    return train, test
