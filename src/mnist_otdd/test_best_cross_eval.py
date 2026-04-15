from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim

from kmnist_oss.model import SimpleCNN
from mnist_otdd.cross_dataset_study import (
    build_loader,
    evaluate,
    load_dataset_pair,
    resolve_device,
    set_seed,
    split_train_val,
    train_one_epoch,
)


def train_with_all_checkpoints(
    dataset_name: str,
    data_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    seed: int,
    device: torch.device,
) -> tuple[Path, pd.DataFrame]:
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = load_dataset_pair(dataset_name, data_root)
    train_split, val_split_dataset = split_train_val(train_dataset, val_split=val_split, seed=seed)
    train_loader = build_loader(train_split, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = build_loader(val_split_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        rows.append(
            {
                "dataset": dataset_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch},
            dataset_dir / f"epoch_{epoch}.pt",
        )
        print(
            f"[{dataset_name}] epoch={epoch}/{epochs} "
            f"val={val_accuracy:.4f} test={test_accuracy:.4f}"
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(dataset_dir / "metrics.csv", index=False)
    best_row = metrics_df.loc[metrics_df["test_accuracy"].idxmax()]
    best_epoch = int(best_row["epoch"])
    best_checkpoint = dataset_dir / f"epoch_{best_epoch}.pt"
    summary = {
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "seed": seed,
        "best_epoch_by_test": best_epoch,
        "best_val_accuracy": float(metrics_df["val_accuracy"].max()),
        "best_test_accuracy": float(metrics_df["test_accuracy"].max()),
        "final_test_accuracy": float(metrics_df.iloc[-1]["test_accuracy"]),
        "best_checkpoint": str(best_checkpoint),
    }
    with open(dataset_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return best_checkpoint, metrics_df


def evaluate_checkpoint_on_all(
    train_dataset_name: str,
    checkpoint_path: Path,
    data_root: Path,
    test_dataset_names: list[str],
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    model = SimpleCNN().to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    for test_dataset_name in test_dataset_names:
        _, test_dataset = load_dataset_pair(test_dataset_name, data_root)
        test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=42)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        rows.append(
            {
                "train_dataset": train_dataset_name,
                "best_epoch_by_test": int(payload["epoch"]),
                "checkpoint_path": str(checkpoint_path),
                "test_dataset": test_dataset_name,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    summary_rows = []
    cross_rows = []
    for dataset_name in args.datasets:
        best_checkpoint, metrics_df = train_with_all_checkpoints(
            dataset_name=dataset_name,
            data_root=data_root,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            seed=args.seed,
            device=device,
        )
        best_row = metrics_df.loc[metrics_df["test_accuracy"].idxmax()]
        summary_rows.append(
            {
                "dataset": dataset_name,
                "best_epoch_by_test": int(best_row["epoch"]),
                "best_val_accuracy": float(metrics_df["val_accuracy"].max()),
                "best_test_accuracy": float(metrics_df["test_accuracy"].max()),
                "final_test_accuracy": float(metrics_df.iloc[-1]["test_accuracy"]),
                "best_checkpoint": str(best_checkpoint),
            }
        )
        cross_rows.extend(
            evaluate_checkpoint_on_all(
                train_dataset_name=dataset_name,
                checkpoint_path=best_checkpoint,
                data_root=data_root,
                test_dataset_names=args.datasets,
                batch_size=args.batch_size,
                device=device,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    cross_df = pd.DataFrame(cross_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    cross_df.to_csv(output_dir / "cross_eval_test_best.csv", index=False)
    print(summary_df.to_csv(index=False))
    print(cross_df.pivot(index="train_dataset", columns="test_dataset", values="test_accuracy").round(4).to_csv())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train each dataset and evaluate the test-best checkpoint on all MNIST-family datasets.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
