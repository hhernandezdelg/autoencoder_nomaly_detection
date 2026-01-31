from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from tqdm.auto import tqdm


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:  # NEW ADDITION
    if x.dim() == 2:
        side = int(x.shape[1] ** 0.5)
        return x.view(x.size(0), 1, side, side)
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def ssim_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:  
    """Manual SSIM loss (returns 1 - SSIM)."""
    x = _ensure_4d(x)
    y = _ensure_4d(y)
    mu_x = x.mean(dim=(2, 3), keepdim=True)
    mu_y = y.mean(dim=(2, 3), keepdim=True)

    sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = num / den
    return 1.0 - ssim.mean()


def train_autoencoder_sgd(
    model,
    train_loader,
    val_loader,
    epochs,
    device="cuda",
    lr=1e-2,
    loss_fn="mse",
):
    """Autoencoder training with SGD and selectable loss."""
    model = model.to(device)
    if loss_fn == "ssim":
        criterion = ssim_loss
    else:
        criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for imgs, _ in loop:
            imgs = imgs.to(device)
            outputs = model(imgs)
            if outputs.shape != imgs.shape:
                imgs_flat = imgs.view(imgs.size(0), -1)
                loss = criterion(outputs, imgs_flat)
            else:
                loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                if outputs.shape != imgs.shape:
                    imgs_flat = imgs.view(imgs.size(0), -1)
                    loss = criterion(outputs, imgs_flat)
                else:
                    loss = criterion(outputs, imgs)
                val_loss += loss.item() * imgs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    return history


def reconstruction_errors(model, dataloader, device="cuda"): 
    """Return per-sample reconstruction errors and labels."""
    model = model.to(device)
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for imgs, y in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            if outputs.shape != imgs.shape:
                imgs_flat = imgs.view(imgs.size(0), -1)
                outputs_flat = outputs
            else:
                imgs_flat = imgs.view(imgs.size(0), -1)
                outputs_flat = outputs.view(outputs.size(0), -1)

            batch_err = ((outputs_flat - imgs_flat) ** 2).mean(dim=1)
            errors.append(batch_err.cpu())
            labels.append(y.cpu())
    return torch.cat(errors), torch.cat(labels)


def select_threshold(errors_clean, errors_corrupt, num_steps=200): 
    """Pick a threshold that maximizes F1 over clean vs corrupt."""
    labels = torch.cat([torch.zeros_like(errors_clean), torch.ones_like(errors_corrupt)])
    scores = torch.cat([errors_clean, errors_corrupt])
    min_s, max_s = scores.min().item(), scores.max().item()
    best = {"threshold": min_s, "f1": -1.0}

    for i in range(num_steps + 1):
        thr = min_s + (max_s - min_s) * (i / num_steps)
        preds = (scores >= thr).long()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.numpy(), preds.numpy(), average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"threshold": thr, "precision": precision, "recall": recall, "f1": f1}
    return best


def confusion_from_threshold(errors, labels, threshold): 
    preds = (errors >= threshold).long()
    tn, fp, fn, tp = confusion_matrix(labels.numpy(), preds.numpy(), labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(), preds.numpy(), average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_metrics_json(payload: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def evaluate_anomaly_detection( 
    model,
    clean_loader,
    corrupt_loader,
    device="cuda",
    threshold: float | None = None,
):
    """Compute threshold, confusion matrix, and precision/recall/F1."""
    clean_errors, _ = reconstruction_errors(model, clean_loader, device=device)
    corrupt_errors, _ = reconstruction_errors(model, corrupt_loader, device=device)
    if threshold is None:
        best = select_threshold(clean_errors, corrupt_errors)
        threshold = best["threshold"]
    labels = torch.cat([torch.zeros_like(clean_errors), torch.ones_like(corrupt_errors)])
    errors = torch.cat([clean_errors, corrupt_errors])
    metrics = confusion_from_threshold(errors, labels, threshold)
    metrics["threshold_method"] = "f1"
    return metrics, clean_errors, corrupt_errors
