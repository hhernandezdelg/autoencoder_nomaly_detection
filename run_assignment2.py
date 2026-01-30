#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from assignment_2.data import (
    build_dataloaders,
    default_transforms,
    load_mnist,
    make_corrupted_subset,
    stratified_indices,
)
from assignment_2.models import ConvAutoencoder, DeepAutoencoder, SimpleAutoencoder
from assignment_2.training import (
    evaluate_anomaly_detection,
    save_metrics_json,
    train_autoencoder_sgd,
)
from assignment_2.visualization import (
    plot_confusion_matrix,
    plot_error_distribution,
    plot_loss_comparison,
    plot_loss_curves,
    plot_original_corrupted,
    plot_reconstructions,
)


def set_seed(seed: int) -> None:
    """Set seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_best_model(histories: dict) -> str:
    """Select the best model based on final validation loss."""
    best_name = None
    best_loss = None
    for name, history in histories.items():
        if not history["val_loss"]:
            continue
        final_loss = history["val_loss"][-1]
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            best_name = name
    if best_name is None:
        raise RuntimeError("No validation losses were recorded")
    return best_name


def main() -> int:
    """Run the full assignment two pipeline and save outputs to disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="assignment_2/output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--noise-level", type=float, default=0.8)
    parser.add_argument("--noise-type", default="gaussian")
    parser.add_argument("--corrupt-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mnist_train, mnist_test = load_mnist(args.data_dir)
    transform_train, transform_val = default_transforms()

    # This section prepares the data loaders for training and validation
    train_idx, val_idx = stratified_indices(mnist_train.targets, val_frac=args.val_frac)
    train_loader, val_loader = build_dataloaders(
        mnist_train,
        train_idx,
        val_idx,
        batch_size=args.batch_size,
        transform_train=transform_train,
        transform_val=transform_val,
        num_workers=4,
        pin_memory=True,
    )

    clean_ds, corrupt_ds = make_corrupted_subset(
        mnist_test,
        fraction=args.corrupt_fraction,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
    )

    # This section visualizes clean and corrupted samples
    plot_original_corrupted(
        clean_ds,
        corrupt_ds,
        n=4,
        save_path=output_dir / "corrupted_samples.png",
    )

    models = {
        "simple_linear": SimpleAutoencoder(),
        "deep_linear": DeepAutoencoder(),
        "conv": ConvAutoencoder(),
    }

    # This section trains each model with mean squared error and SGD
    histories = {}
    for name, model in models.items():
        model_out = output_dir / name
        model_out.mkdir(parents=True, exist_ok=True)

        history = train_autoencoder_sgd(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            device=device,
            lr=1e-2,
            loss_fn="mse",
        )
        histories[name] = history

        # NEW ADDITION: save final model weights for this architecture
        torch.save(model.state_dict(), model_out / "model_mse.pth")

        save_metrics_json(
            {"loss": history, "model": name, "loss_name": "mse"},
            model_out / "train_mse.json",
        )
        plot_loss_curves(
            history,
            title=f"{name} mse loss",
            save_path=model_out / "train_mse.png",
        )
        plot_reconstructions(
            model,
            clean_ds,
            corrupt_ds,
            device=device,
            n=4,
            save_path=model_out / "reconstructions.png",
        )

    ssim_model = SimpleAutoencoder()
    ssim_history = train_autoencoder_sgd(
        ssim_model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=1e-2,
        loss_fn="ssim",
    )

    # NEW ADDITION: save final model weights for SSIM variant
    torch.save(ssim_model.state_dict(), output_dir / "simple_linear" / "model_ssim.pth")

    save_metrics_json(
        {"loss": ssim_history, "model": "simple_linear", "loss_name": "ssim"},
        output_dir / "simple_linear" / "train_ssim.json",
    )
    plot_loss_comparison(
        histories["simple_linear"],
        ssim_history,
        save_path=output_dir / "simple_linear" / "loss_mse_vs_ssim.png",
    )

    # This section selects the best model and evaluates anomaly detection
    best_name = select_best_model(histories)
    best_model = models[best_name]

    clean_loader = torch.utils.data.DataLoader(
        clean_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    corrupt_loader = torch.utils.data.DataLoader(
        corrupt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    anomaly_metrics, clean_errors, corrupt_errors = evaluate_anomaly_detection(
        best_model,
        clean_loader,
        corrupt_loader,
        device=device,
    )

    save_metrics_json(
        {"best_model": best_name, "anomaly_metrics": anomaly_metrics},
        output_dir / "anomaly_metrics.json",
    )

    plot_error_distribution(
        clean_errors.numpy(),
        corrupt_errors.numpy(),
        threshold=anomaly_metrics["threshold"],
        save_path=output_dir / "error_distribution.png",
    )

    cm = np.array(
        [
            [anomaly_metrics["tn"], anomaly_metrics["fp"]],
            [anomaly_metrics["fn"], anomaly_metrics["tp"]],
        ]
    )
    plot_confusion_matrix(cm, save_path=output_dir / "confusion_matrix.png")

    summary = {
        "best_model": best_name,
        "noise_level": args.noise_level,
        "noise_type": args.noise_type,
        "corrupt_fraction": args.corrupt_fraction,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
