from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import nn
from tqdm.auto import tqdm


class Trainer(nn.Module):
    def __init__(self, base_model, num_classes, model_name="default_model", lr=1e-3):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.num_classes = num_classes
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()

        # Metric tracking
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.train_aucs, self.val_aucs = [], []
        self.train_balanced_accs, self.val_balanced_accs = [], []

    def forward(self, x):
        return self.model(x)

    # Optimizer
    def get_optimizer(self, lr=None):
        lr = lr if lr is not None else self.lr
        if lr is None:
            raise ValueError("Learning rate is not defined")
        return torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

    def train_one_epoch(self, dataloader, optimizer, device):
        self.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        probs_all = []
        targets_all = []

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = self(x)
            loss = self.criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            probs_all.append(probs.detach().cpu())
            targets_all.append(y.detach().cpu())

        # Flatten predictions and targets
        all_probs = torch.cat(probs_all)
        all_targets = torch.cat(targets_all)
        all_preds = all_probs.argmax(dim=1)

        avg_loss = epoch_loss / total
        acc = correct / total

        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(all_targets.numpy(), all_preds.numpy())

        # AUC (optional, handle exception)
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"Could not compute train AUC: {e}")
            auc = float("nan")

        # Store metrics
        self.train_losses.append(avg_loss)
        self.train_accs.append(acc)
        self.train_balanced_accs.append(balanced_acc)
        self.train_aucs.append(auc)

        return avg_loss, acc, balanced_acc, auc

    def validate_one_epoch(self, dataloader, device):
        self.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0
        probs_all = []
        targets_all = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = self(x)
                loss = self.criterion(logits, y)

                epoch_loss += loss.item() * x.size(0)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                probs_all.append(probs.cpu())
                targets_all.append(y.cpu())

        # Flatten predictions and targets
        all_probs = torch.cat(probs_all)
        all_targets = torch.cat(targets_all)
        all_preds = all_probs.argmax(dim=1)

        avg_loss = epoch_loss / total
        acc = correct / total

        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(all_targets.numpy(), all_preds.numpy())

        # AUC (optional, handle exception)
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"Could not compute val AUC: {e}")
            auc = float("nan")

        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_accs.append(acc)
        self.val_balanced_accs.append(balanced_acc)  # NEW: store balanced accuracy
        self.val_aucs.append(auc)

        return avg_loss, acc, balanced_acc, auc

    def plot_metrics(self, save_dir="./output", filename="final_metrics.png", test_loss=None, test_acc=None, test_bal_acc=None):
        clear_output(wait=True)
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(16, 5))

        # -- Use nicer colors
        train_color = "#1f77b4"  # blue
        val_color = "#ff7f0e"  # orange
        test_color = "#2ca02c"  # green

        # -- Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss", color=train_color)
        if len(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, label="Val Loss", color=val_color)
        if test_loss is not None:
            plt.scatter(
                epochs[-1],
                test_loss,
                color=test_color,
                s=80,
                marker="*",
                label="Test Loss",
                edgecolors="black",
                zorder=5,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # -- Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accs, label="Train Acc", color=train_color)
        if len(self.val_accs) > 0:
            plt.plot(epochs, self.val_accs, label="Val Acc", color=val_color)
        if test_acc is not None:
            plt.scatter(
                epochs[-1],
                test_acc,
                color=test_color,
                s=80,
                marker="*",
                label="Test Acc",
                edgecolors="black",
                zorder=5,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # -- Balanced Accuracy plot
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.train_balanced_accs, label="Train Bal Acc", color=train_color)
        if len(self.val_balanced_accs) > 0:
            plt.plot(epochs, self.val_balanced_accs, label="Val Bal Acc", color=val_color)
        if test_bal_acc is not None:
            plt.scatter(
                epochs[-1],
                test_bal_acc,
                color=test_color,
                s=80,
                marker="*",
                label="Test Bal Acc",
                edgecolors="black",
                zorder=5,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Accuracy")
        plt.legend()

        # -- Save final plot
        save_dir_ = Path(save_dir) / self.model_name
        save_dir_.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir_ / filename, bbox_inches="tight")

        plt.show()


def train_transfer_model(config, model_name, train_loader, val_loader, epochs=10, save_dir="./output", fold=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(save_dir, exist_ok=True)

    # Get number of classes.
    num_classes = 10

    # Build base model using config
    base_model = config["model_fn"](num_classes)

    # Wrap it in Trainer wrapper
    trainer = Trainer(base_model, num_classes=num_classes, model_name=model_name, lr=config["lr"])
    trainer.to(device)

    # Optimizer (optimizer is defined within the Trainer class).
    optimizer = trainer.get_optimizer()

    # Print the layers to be trained
    n_trainable = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    # print(f"Trainable parameters in [{trainer}]: {n_trainable:,}")

    # -- Step 6.6 Training loop
    for epoch in range(epochs):
        train_loss, train_acc, train_bal_acc, train_auc = trainer.train_one_epoch(
            train_loader, optimizer=optimizer, device=device
        )
        val_loss, val_acc, val_bal_acc, val_auc = trainer.validate_one_epoch(val_loader, device=device)

        print(f"\nEpoch {epoch+1}/{epochs}")
        # Live plot
        # trainer.plot_metrics(save_path=f"epoch_{epoch+1}.png")

    # After all epochs finished and test evaluation done
    trainer.plot_metrics(
        save_dir="./output",
        filename="final_metrics.png",
    )

    # Save metrics
    metrics = {
        "train_loss": trainer.train_losses,
        "train_auc": trainer.train_aucs,
        "train_acc": trainer.train_accs,
        "train_bal_acc": trainer.train_balanced_accs,
        "val_loss": trainer.val_losses,
        "val_auc": trainer.val_aucs,
        "val_acc": trainer.val_accs,
        "val_bal_acc": trainer.val_balanced_accs,
    }

    # Save model
    torch.save(trainer.model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    print(f"Model weights saved to: {save_dir}")

    # Step 6.10: Save metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {save_dir}/metrics.json")

    return metrics


def train_autoencoder(model, train_loader, val_loader, epochs, device="cuda"):
    """
    Dedicated training loop for Autoencoders (Anomaly Detection).
    Focuses on Reconstruction Loss (MSE).
    """
    model = model.to(device)

    # Autoencoder use MSE Loss (Input vs Output)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        # tqdm for progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in loop:
            imgs, _ = batch

            # Move to device
            imgs = imgs.to(device)

            # Forward pass
            outputs = model(imgs)

            # Calculate Loss
            if outputs.shape != imgs.shape:
                imgs_flat = imgs.view(imgs.size(0), -1)
                loss = criterion(outputs, imgs_flat)
            else:
                loss = criterion(outputs, imgs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                imgs, _ = batch
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

        print(
            f"Epoch {epoch+1} | Train Loss (MSE): {avg_train_loss:.6f} | Val Loss (MSE): {avg_val_loss:.6f}"
        )

    return history


def evaluate_test_set(  # NEW ADDITION
    model,
    model_path: str,
    test_loader,
    device: str | None = None,
):
    model.load_state_dict(torch.load(model_path))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    # Test set evaluation
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Statistics
            test_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)  # no softmax
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            # For AUC, we still need softmax probabilities
            probs = F.softmax(outputs, dim=1)

            all_preds.append(predicted.cpu())
            all_targets.append(y_batch.cpu())
            all_probs.append(probs.cpu())

    # Combine all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    # Compute average loss and accuracy
    test_loss /= total
    test_acc = correct / total

    # Compute balanced accuracy
    test_balanced_acc = balanced_accuracy_score(all_targets.numpy(), all_preds.numpy())

    # Compute AUC
    try:
        auc = roc_auc_score(all_targets.numpy(), all_probs.numpy(), multi_class="ovr", average="macro")
    except Exception as e:
        print(f"Could not compute test AUC: {e}")
        auc = float("nan")

    print(f"Test → Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Balanced Acc: {test_balanced_acc:.4f}, AUC: {auc:.4f}")
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_balanced_acc": test_balanced_acc,
        "test_auc": auc,
    }


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:  # NEW ADDITION
    if x.dim() == 2:
        side = int(x.shape[1] ** 0.5)
        return x.view(x.size(0), 1, side, side)
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def ssim_loss(x: torch.Tensor, y: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:  # NEW ADDITION
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


def train_autoencoder_sgd(  # NEW ADDITION
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


def reconstruction_errors(model, dataloader, device="cuda"):  # NEW ADDITION
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


def select_threshold(errors_clean, errors_corrupt, num_steps=200):  # NEW ADDITION
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


def confusion_from_threshold(errors, labels, threshold):  # NEW ADDITION
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


def save_metrics_json(payload: dict, path: str | Path):  # NEW ADDITION
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def evaluate_anomaly_detection(  # NEW ADDITION
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
