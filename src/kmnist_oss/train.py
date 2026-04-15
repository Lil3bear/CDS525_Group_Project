import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim

from kmnist_oss.data import get_dataloaders, set_seed
from kmnist_oss.model import SimpleCNN

ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def get_loss_setup(loss_name: str):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(), False, "CrossEntropyLoss"
    if loss_name == "label_smoothing":
        try:
            return (
                nn.CrossEntropyLoss(label_smoothing=0.1),
                False,
                "CrossEntropyLoss(label_smoothing=0.1)",
            )
        except TypeError:
            return nn.NLLLoss(), True, "NLLLoss with log_softmax fallback"
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_loss(logits, targets, loss_fn, use_log_softmax: bool):
    if use_log_softmax:
        return loss_fn(F.log_softmax(logits, dim=1), targets)
    return loss_fn(logits, targets)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, use_log_softmax: bool):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = compute_loss(logits, labels, loss_fn, use_log_softmax)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, use_log_softmax: bool):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = compute_loss(logits, labels, loss_fn, use_log_softmax)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    run_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    loss_name: str,
    seed: int = 42,
    device_name: str = "auto",
    data_dir: str = "data",
    val_split: float = 0.1,
    num_workers: int = 0,
):
    ensure_output_dirs()
    set_seed(seed)
    device = resolve_device(device_name)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
        num_workers=num_workers,
    )

    model = SimpleCNN().to(device)
    loss_fn, use_log_softmax, loss_description = get_loss_setup(loss_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics_path = RESULTS_DIR / f"{run_name}_metrics.csv"
    config_path = RESULTS_DIR / f"{run_name}_config.json"
    checkpoint_path = CHECKPOINTS_DIR / f"best_{run_name}.pt"

    rows = []
    best_val_accuracy = -1.0

    config = {
        "run_name": run_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "loss_name": loss_name,
        "loss_description": loss_description,
        "seed": seed,
        "device": str(device),
        "data_dir": data_dir,
        "val_split": val_split,
        "num_workers": num_workers,
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint_path),
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            use_log_softmax,
        )
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            use_log_softmax,
        )
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            use_log_softmax,
        )

        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

        print(
            f"[{run_name}] "
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_acc={val_accuracy:.4f} "
            f"test_acc={test_accuracy:.4f}"
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_path, index=False)

    best_row = metrics_df.loc[metrics_df["val_accuracy"].idxmax()]
    summary = {
        **config,
        "best_epoch": int(best_row["epoch"]),
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_test_accuracy": float(metrics_df["test_accuracy"].max()),
        "final_test_accuracy": float(metrics_df.iloc[-1]["test_accuracy"]),
    }

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved config to {config_path}")
    print(f"Saved checkpoint to {checkpoint_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple CNN on KMNIST.")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "label_smoothing"],
        default="cross_entropy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        run_name=args.run_name,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        loss_name=args.loss,
        seed=args.seed,
        device_name=args.device,
        data_dir=args.data_dir,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
