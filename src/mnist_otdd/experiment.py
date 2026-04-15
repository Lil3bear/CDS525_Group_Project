from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import functional as TF

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent

from kmnist_oss.model import SimpleCNN  # noqa: E402
from otdd.pytorch.sotdd import compute_pairwise_distance  # noqa: E402


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    return (image_tensor - MNIST_MEAN) / MNIST_STD


def apply_domain_transform(image: Image.Image, domain_name: str, idx: int, seed: int) -> torch.Tensor:
    if domain_name == "clean":
        tensor = TF.to_tensor(image)
    elif domain_name == "rotate15":
        tensor = TF.to_tensor(TF.rotate(image, angle=15.0, fill=0))
    elif domain_name == "rotate30":
        tensor = TF.to_tensor(TF.rotate(image, angle=30.0, fill=0))
    elif domain_name == "invert":
        tensor = 1.0 - TF.to_tensor(image)
    elif domain_name == "noise":
        tensor = TF.to_tensor(image)
        generator = torch.Generator().manual_seed(seed * 1000003 + idx)
        noise = torch.randn(tensor.shape, generator=generator) * 0.35
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
    return normalize_tensor(tensor)


class IndexedMNISTDomainDataset(Dataset):
    def __init__(
        self,
        base_dataset: MNIST,
        indices: np.ndarray,
        domain_name: str,
        seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.domain_name = domain_name
        self.seed = seed
        self.targets = torch.as_tensor(self.base_dataset.targets)[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_idx = int(self.indices[item])
        image = Image.fromarray(self.base_dataset.data[base_idx].numpy())
        image_tensor = apply_domain_transform(image, self.domain_name, base_idx, self.seed)
        return image_tensor, int(self.base_dataset.targets[base_idx])


@dataclass
class DistilledSubset:
    name: str
    dataset: Dataset
    indices: np.ndarray


def label_to_indices(targets: torch.Tensor) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    targets_np = targets.cpu().numpy()
    for label in sorted(np.unique(targets_np)):
        result[int(label)] = np.flatnonzero(targets_np == label)
    return result


def materialize_features(dataset: Dataset, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs = []
    ys = []
    for images, labels in loader:
        xs.append(images.to(device))
        ys.append(labels.to(device))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def sample_balanced_positions(targets: torch.Tensor, subset_size: int, seed: int) -> np.ndarray:
    by_label = label_to_indices(targets)
    rng = np.random.default_rng(seed)
    num_classes = len(by_label)
    base = subset_size // num_classes
    remainder = subset_size % num_classes
    selected = []
    for offset, label in enumerate(sorted(by_label)):
        count = base + (1 if offset < remainder else 0)
        label_positions = by_label[label]
        count = min(count, len(label_positions))
        chosen = rng.choice(label_positions, size=count, replace=False)
        selected.extend(chosen.tolist())
    selected = np.array(selected, dtype=np.int64)
    if len(selected) < subset_size:
        remaining = np.setdiff1d(np.arange(len(targets)), selected, assume_unique=False)
        extra = rng.choice(remaining, size=subset_size - len(selected), replace=False)
        selected = np.concatenate([selected, extra])
    rng.shuffle(selected)
    return selected


def select_prototype_positions(
    features: torch.Tensor,
    targets: torch.Tensor,
    subset_size: int,
    reverse: bool,
) -> np.ndarray:
    flat = features.view(features.size(0), -1)
    by_label = label_to_indices(targets)
    num_classes = len(by_label)
    base = subset_size // num_classes
    remainder = subset_size % num_classes
    selected = []
    for offset, label in enumerate(sorted(by_label)):
        count = base + (1 if offset < remainder else 0)
        positions = by_label[label]
        class_features = flat[positions]
        centroid = class_features.mean(dim=0, keepdim=True)
        distances = torch.norm(class_features - centroid, dim=1)
        order = torch.argsort(distances, descending=reverse)
        chosen = positions[order[: min(count, len(order))].cpu().numpy()]
        selected.extend(chosen.tolist())
    return np.array(selected, dtype=np.int64)


def build_distilled_subsets(
    domain_dataset: IndexedMNISTDomainDataset,
    subset_size: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> list[DistilledSubset]:
    rng = np.random.default_rng(seed)
    positions = np.arange(len(domain_dataset))
    random_positions = rng.choice(positions, size=subset_size, replace=False)
    balanced_positions = sample_balanced_positions(domain_dataset.targets, subset_size, seed)
    features, labels = materialize_features(domain_dataset, batch_size=batch_size, device=device)
    prototype_positions = select_prototype_positions(features, labels, subset_size, reverse=False)
    hard_positions = select_prototype_positions(features, labels, subset_size, reverse=True)
    subsets = []
    for name, local_positions in [
        ("random", random_positions),
        ("balanced_random", balanced_positions),
        ("prototype", prototype_positions),
        ("hard", hard_positions),
    ]:
        selected_indices = domain_dataset.indices[local_positions]
        subset_dataset = IndexedMNISTDomainDataset(
            base_dataset=domain_dataset.base_dataset,
            indices=selected_indices,
            domain_name=domain_dataset.domain_name,
            seed=domain_dataset.seed,
        )
        subsets.append(DistilledSubset(name=name, dataset=subset_dataset, indices=selected_indices))
    return subsets


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples


def train_classifier(
    train_dataset: Dataset,
    test_dataset: Dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size_local = labels.size(0)
            total_loss += loss.item() * batch_size_local
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size_local
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_examples,
                "train_accuracy": total_correct / total_examples,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
    final_row = history[-1]
    best_test_accuracy = max(row["test_accuracy"] for row in history)
    return {
        "final_train_loss": final_row["train_loss"],
        "final_train_accuracy": final_row["train_accuracy"],
        "final_test_loss": final_row["test_loss"],
        "final_test_accuracy": final_row["test_accuracy"],
        "best_test_accuracy": best_test_accuracy,
    }


def compute_sotdd_distance(
    dataset_a: Dataset,
    dataset_b: Dataset,
    batch_size: int,
    num_projections: int,
    device: torch.device,
) -> float:
    loader_a = DataLoader(dataset_a, batch_size=batch_size, shuffle=False)
    loader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=False)
    distances = compute_pairwise_distance(
        list_D=[loader_a, loader_b],
        device=str(device),
        num_projections=num_projections,
        dimension=784,
        num_channels=1,
        num_moments=5,
        use_conv=False,
        precision="float",
        p=2,
        chunk=min(500, num_projections),
    )
    return float(distances[0])


def build_domain_indices(targets: torch.Tensor, domain_sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = np.arange(len(targets))
    rng.shuffle(positions)
    return positions[:domain_sample_size]


def create_domain_datasets(
    data_root: Path,
    domain_sample_size: int,
    seed: int,
    domain_names: list[str],
) -> tuple[dict[str, IndexedMNISTDomainDataset], dict[str, IndexedMNISTDomainDataset]]:
    train_base = MNIST(root=data_root, train=True, download=True)
    test_base = MNIST(root=data_root, train=False, download=True)
    train_indices = build_domain_indices(torch.as_tensor(train_base.targets), domain_sample_size, seed)
    test_indices = build_domain_indices(torch.as_tensor(test_base.targets), min(2000, len(test_base)), seed + 7)
    train_domains = {}
    test_domains = {}
    for offset, domain_name in enumerate(domain_names):
        train_domains[domain_name] = IndexedMNISTDomainDataset(train_base, train_indices, domain_name, seed + offset)
        test_domains[domain_name] = IndexedMNISTDomainDataset(test_base, test_indices, domain_name, seed + offset)
    return train_domains, test_domains


def save_heatmap(distance_df: pd.DataFrame, output_path: Path) -> None:
    pivot = distance_df.pivot(index="domain_a", columns="domain_b", values="sotdd_distance")
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.values, cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.values[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pairwise MNIST Domain s-OTDD Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_scatter(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for strategy, group in results_df.groupby("strategy"):
        ax.scatter(group["distance_to_full"], group["final_test_accuracy"], label=strategy, alpha=0.8)
    ax.set_xlabel("s-OTDD distance to full domain")
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Distance vs Distilled Subset Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_domains, test_domains = create_domain_datasets(
        data_root=Path(args.data_root),
        domain_sample_size=args.domain_sample_size,
        seed=args.seed,
        domain_names=args.domains,
    )

    domain_rows = []
    domain_names = list(train_domains)
    for index_a, domain_a in enumerate(domain_names):
        for index_b, domain_b in enumerate(domain_names):
            if index_b < index_a:
                continue
            if domain_a == domain_b:
                distance = 0.0
            else:
                distance = compute_sotdd_distance(
                    train_domains[domain_a],
                    train_domains[domain_b],
                    batch_size=args.distance_batch_size,
                    num_projections=args.num_projections,
                    device=device,
                )
            domain_rows.append({"domain_a": domain_a, "domain_b": domain_b, "sotdd_distance": distance})
            if domain_a != domain_b:
                domain_rows.append({"domain_a": domain_b, "domain_b": domain_a, "sotdd_distance": distance})
            print(f"[domain-distance] {domain_a} vs {domain_b}: {distance:.4f}")
    domain_df = pd.DataFrame(domain_rows)
    domain_df.to_csv(output_dir / "domain_distances.csv", index=False)
    save_heatmap(domain_df, output_dir / "domain_distance_heatmap.png")

    result_rows = []
    for domain_name, full_train_dataset in train_domains.items():
        for subset_size in args.subset_sizes:
            subsets = build_distilled_subsets(
                domain_dataset=full_train_dataset,
                subset_size=subset_size,
                seed=args.seed + subset_size,
                batch_size=args.feature_batch_size,
                device=device,
            )
            for subset in subsets:
                distance_to_full = compute_sotdd_distance(
                    subset.dataset,
                    full_train_dataset,
                    batch_size=args.distance_batch_size,
                    num_projections=args.num_projections,
                    device=device,
                )
                metrics = train_classifier(
                    train_dataset=subset.dataset,
                    test_dataset=test_domains[domain_name],
                    epochs=args.epochs,
                    batch_size=args.train_batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    device=device,
                )
                row = {
                    "domain": domain_name,
                    "subset_size": subset_size,
                    "strategy": subset.name,
                    "distance_to_full": distance_to_full,
                    **metrics,
                }
                result_rows.append(row)
                print(
                    f"[distill] domain={domain_name} size={subset_size} strategy={subset.name} "
                    f"distance={distance_to_full:.4f} acc={metrics['final_test_accuracy']:.4f}"
                )

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(output_dir / "distillation_results.csv", index=False)
    save_scatter(results_df, output_dir / "distance_vs_accuracy.png")

    summary = {
        "device": str(device),
        "data_root": str(args.data_root),
        "domain_sample_size": args.domain_sample_size,
        "subset_sizes": args.subset_sizes,
        "epochs": args.epochs,
        "num_projections": args.num_projections,
        "distance_batch_size": args.distance_batch_size,
        "train_batch_size": args.train_batch_size,
        "feature_batch_size": args.feature_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST domain distance and data distillation experiment.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default=str(Path("artifacts") / "mnist_otdd" / "results"))
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[200, 500, 1000])
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["clean", "rotate15", "rotate30", "noise", "invert"],
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--domain-sample-size", type=int, default=3000)
    parser.add_argument("--num-projections", type=int, default=500)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
