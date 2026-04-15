from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import fetch_openml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import EMNIST, FashionMNIST, KMNIST, MNIST, USPS
from torchvision.transforms import functional as TF

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent

from kmnist_oss.model import SimpleCNN  # noqa: E402
from otdd.pytorch.sotdd import compute_pairwise_distance  # noqa: E402


RESULTS_SUBDIR = "cross_dataset_results"
GLOBAL_MEAN = 0.5
GLOBAL_STD = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def normalize_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    return (image_tensor - GLOBAL_MEAN) / GLOBAL_STD


def denormalize_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(image_tensor * GLOBAL_STD + GLOBAL_MEAN, 0.0, 1.0)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")


class TensorDigitDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, dataset_name: str) -> None:
        self.images = images.float()
        self.labels = labels.long()
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index])


class WrappedTorchvisionDataset(Dataset):
    def __init__(self, base_dataset, dataset_name: str):
        self.base_dataset = base_dataset
        self.dataset_name = dataset_name
        self.targets = torch.as_tensor(self.base_dataset.targets).long()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        if isinstance(image, Image.Image):
            image = TF.resize(image, [28, 28])
            image = TF.to_tensor(image)
        else:
            image = TF.resize(image, [28, 28])
        image = normalize_tensor(image)
        return image, int(label)


def load_kmnist_with_fallback(data_root: Path) -> tuple[Dataset, Dataset]:
    try:
        train = KMNIST(root=data_root, train=True, download=False)
        test = KMNIST(root=data_root, train=False, download=False)
        return WrappedTorchvisionDataset(train, "KMNIST"), WrappedTorchvisionDataset(test, "KMNIST")
    except Exception:
        cache_dir = ensure_dir(data_root / "KMNIST" / "openml_cache")
        train_path = cache_dir / "train.pt"
        test_path = cache_dir / "test.pt"
        if train_path.exists() and test_path.exists():
            train_payload = torch.load(train_path)
            test_payload = torch.load(test_path)
        else:
            dataset = fetch_openml(name="Kuzushiji-MNIST", version=1, as_frame=False, parser="auto")
            images = dataset.data.reshape(-1, 28, 28).astype(np.float32)
            labels = dataset.target.astype(np.int64)
            train_payload = {
                "images": torch.from_numpy(images[:60000]).unsqueeze(1) / 255.0,
                "labels": torch.from_numpy(labels[:60000]),
            }
            test_payload = {
                "images": torch.from_numpy(images[60000:]).unsqueeze(1) / 255.0,
                "labels": torch.from_numpy(labels[60000:]),
            }
            torch.save(train_payload, train_path)
            torch.save(test_payload, test_path)
        return (
            TensorDigitDataset(normalize_tensor(train_payload["images"]), train_payload["labels"], "KMNIST"),
            TensorDigitDataset(normalize_tensor(test_payload["images"]), test_payload["labels"], "KMNIST"),
        )


def load_dataset_pair(dataset_name: str, data_root: Path) -> tuple[Dataset, Dataset]:
    if dataset_name == "MNIST":
        train = MNIST(root=data_root, train=True, download=True)
        test = MNIST(root=data_root, train=False, download=True)
        return WrappedTorchvisionDataset(train, dataset_name), WrappedTorchvisionDataset(test, dataset_name)
    if dataset_name == "FashionMNIST":
        train = FashionMNIST(root=data_root, train=True, download=True)
        test = FashionMNIST(root=data_root, train=False, download=True)
        return WrappedTorchvisionDataset(train, dataset_name), WrappedTorchvisionDataset(test, dataset_name)
    if dataset_name == "EMNISTDigits":
        train = EMNIST(root=data_root, split="digits", train=True, download=True)
        test = EMNIST(root=data_root, split="digits", train=False, download=True)
        return WrappedTorchvisionDataset(train, dataset_name), WrappedTorchvisionDataset(test, dataset_name)
    if dataset_name == "USPS":
        train = USPS(root=data_root, train=True, download=True)
        test = USPS(root=data_root, train=False, download=True)
        return WrappedTorchvisionDataset(train, dataset_name), WrappedTorchvisionDataset(test, dataset_name)
    if dataset_name == "KMNIST":
        return load_kmnist_with_fallback(data_root)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def split_train_val(dataset: Dataset, val_split: float, seed: int) -> tuple[Dataset, Dataset]:
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples


def materialize_dataset(dataset: Dataset, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images = []
    labels = []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def compute_sotdd_distance(dataset_a: Dataset, dataset_b: Dataset, batch_size: int, num_projections: int, device: torch.device) -> float:
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


def sample_dataset(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    if len(dataset) <= sample_size:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    subset, _ = random_split(dataset, [sample_size, len(dataset) - sample_size], generator=generator)
    return subset


def train_real_dataset(
    dataset_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    seed: int,
    device: torch.device,
) -> tuple[SimpleCNN, pd.DataFrame, dict]:
    train_split, val_split_dataset = split_train_val(train_dataset, val_split=val_split, seed=seed)
    train_loader = build_loader(train_split, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = build_loader(val_split_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    best_val_accuracy = -1.0
    checkpoint_path = output_dir / f"{dataset_name}_best.pt"

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
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        print(
            f"[real:{dataset_name}] epoch={epoch}/{epochs} "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} test_acc={test_accuracy:.4f}"
        )

    best_payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / f"{dataset_name}_metrics.csv", index=False)
    summary = {
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "best_val_accuracy": float(metrics_df["val_accuracy"].max()),
        "best_test_accuracy": float(metrics_df["test_accuracy"].max()),
        "final_test_accuracy": float(metrics_df.iloc[-1]["test_accuracy"]),
    }
    with open(output_dir / f"{dataset_name}_config.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return model, metrics_df, summary


def evaluate_model_on_all(model: SimpleCNN, model_name: str, test_datasets: dict[str, Dataset], batch_size: int, device: torch.device) -> list[dict]:
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    for test_name, dataset in test_datasets.items():
        loader = build_loader(dataset, batch_size=batch_size, shuffle=False, seed=42)
        test_loss, test_accuracy = evaluate(model, loader, loss_fn, device)
        rows.append(
            {
                "train_dataset": model_name,
                "test_dataset": test_name,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
    return rows


def compute_gap_matrix(train_datasets: dict[str, Dataset], gap_sample_size: int, num_projections: int, batch_size: int, device: torch.device, output_dir: Path) -> pd.DataFrame:
    names = list(train_datasets)
    sampled = {
        name: sample_dataset(dataset, gap_sample_size, seed=100 + idx)
        for idx, (name, dataset) in enumerate(train_datasets.items())
    }
    rows = []
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if j < i:
                continue
            if i == j:
                distance = 0.0
            else:
                distance = compute_sotdd_distance(sampled[name_a], sampled[name_b], batch_size=batch_size, num_projections=num_projections, device=device)
            rows.append({"dataset_a": name_a, "dataset_b": name_b, "gap": distance})
            if name_a != name_b:
                rows.append({"dataset_a": name_b, "dataset_b": name_a, "gap": distance})
            print(f"[gap] {name_a} vs {name_b}: {distance:.4f}")
    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(output_dir / "dataset_gap_matrix.csv", index=False)
    return gap_df


def compute_gap_correlation(cross_eval_df: pd.DataFrame, gap_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    merged = cross_eval_df.merge(
        gap_df,
        left_on=["train_dataset", "test_dataset"],
        right_on=["dataset_a", "dataset_b"],
        how="left",
    )
    non_diagonal = merged[merged["train_dataset"] != merged["test_dataset"]].copy()
    pearson_value = pearsonr(non_diagonal["gap"], non_diagonal["test_accuracy"])
    spearman_value = spearmanr(non_diagonal["gap"], non_diagonal["test_accuracy"])
    summary = pd.DataFrame(
        [
            {
                "pearson_r": pearson_value.statistic,
                "pearson_p": pearson_value.pvalue,
                "spearman_rho": spearman_value.statistic,
                "spearman_p": spearman_value.pvalue,
            }
        ]
    )
    non_diagonal.to_csv(output_dir / "gap_vs_ood_pairs.csv", index=False)
    summary.to_csv(output_dir / "gap_vs_ood_summary.csv", index=False)
    return non_diagonal


def build_gap_aware_distilled_dataset(
    train_datasets: dict[str, Dataset],
    gap_df: pd.DataFrame,
    ipc: int,
    feature_sample_size: int,
    seed: int,
    output_dir: Path,
) -> TensorDigitDataset:
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    centrality = (1.0 / mean_gaps).to_dict()
    total_weight = sum(centrality.values())
    normalized_weights = {name: value / total_weight for name, value in centrality.items()}

    images_by_dataset = {}
    labels_by_dataset = {}
    for name, dataset in train_datasets.items():
        sampled = sample_dataset(dataset, feature_sample_size, seed=seed + len(name))
        images, labels = materialize_dataset(sampled)
        images_by_dataset[name] = images
        labels_by_dataset[name] = labels

    proto_images = []
    proto_labels = []
    provenance_rows = []
    for label in range(10):
        weighted_vectors = []
        for name in train_datasets:
            class_images = images_by_dataset[name][labels_by_dataset[name] == label]
            if len(class_images) == 0:
                continue
            class_flat = denormalize_tensor(class_images).view(len(class_images), -1).numpy()
            take_count = max(1, int(round(ipc * len(train_datasets) * normalized_weights[name])))
            rng = np.random.default_rng(seed + label * 17 + len(name))
            choice = rng.choice(len(class_flat), size=min(take_count, len(class_flat)), replace=False)
            weighted_vectors.append(class_flat[choice])
            provenance_rows.append({"dataset": name, "label": label, "selected_samples": int(len(choice)), "weight": normalized_weights[name]})
        vectors = np.concatenate(weighted_vectors, axis=0)
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=ipc, random_state=seed, n_init=10)
        kmeans.fit(vectors)
        centroids = np.clip(kmeans.cluster_centers_, 0.0, 1.0).reshape(ipc, 1, 28, 28)
        proto_images.append(torch.from_numpy(centroids).float())
        proto_labels.append(torch.full((ipc,), label, dtype=torch.long))

    images = torch.cat(proto_images, dim=0)
    labels = torch.cat(proto_labels, dim=0)
    dataset = TensorDigitDataset(normalize_tensor(images), labels, "GapAwareDistilled")
    pd.DataFrame(provenance_rows).to_csv(output_dir / "distilled_dataset_provenance.csv", index=False)
    torch.save({"images": images, "labels": labels}, output_dir / "distilled_dataset.pt")
    return dataset


def train_distilled_dataset(
    distilled_dataset: Dataset,
    test_datasets: dict[str, Dataset],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict]]:
    train_loader = build_loader(distilled_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy}
        for test_name, dataset in test_datasets.items():
            loader = build_loader(dataset, batch_size=batch_size, shuffle=False, seed=seed)
            _, accuracy = evaluate(model, loader, loss_fn, device)
            row[f"test_accuracy_{test_name}"] = accuracy
        rows.append(row)
        print(f"[distilled] epoch={epoch}/{epochs} train_acc={train_accuracy:.4f}")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "distilled_training_metrics.csv", index=False)
    cross_rows = evaluate_model_on_all(model, "GapAwareDistilled", test_datasets, batch_size=batch_size, device=device)
    pd.DataFrame(cross_rows).to_csv(output_dir / "distilled_cross_eval.csv", index=False)
    return metrics_df, cross_rows


def compute_distilled_gaps(distilled_dataset: Dataset, real_train_datasets: dict[str, Dataset], gap_sample_size: int, num_projections: int, batch_size: int, device: torch.device, output_dir: Path) -> pd.DataFrame:
    rows = []
    sampled_distilled = sample_dataset(distilled_dataset, min(gap_sample_size, len(distilled_dataset)), seed=909)
    for index, (name, dataset) in enumerate(real_train_datasets.items()):
        sampled_real = sample_dataset(dataset, gap_sample_size, seed=707 + index)
        gap = compute_sotdd_distance(sampled_distilled, sampled_real, batch_size=batch_size, num_projections=num_projections, device=device)
        rows.append({"source": "GapAwareDistilled", "target": name, "gap": gap})
        print(f"[distilled-gap] GapAwareDistilled vs {name}: {gap:.4f}")
    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(output_dir / "distilled_to_real_gap.csv", index=False)
    return gap_df


def plot_training_curves(all_metrics: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for dataset_name, metrics in all_metrics.items():
        axes[0].plot(metrics["epoch"], metrics["train_loss"], marker="o", label=dataset_name)
        axes[1].plot(metrics["epoch"], metrics["train_accuracy"] * 100, marker="o", label=dataset_name)
        axes[2].plot(metrics["epoch"], metrics["test_accuracy"] * 100, marker="o", label=dataset_name)
    axes[0].set_title("Training Loss")
    axes[1].set_title("Training Accuracy")
    axes[2].set_title("In-Domain Test Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy (%)")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].legend()
    save_figure(fig, output_dir / "all_datasets_training_curves.png")


def plot_heatmap(matrix_df: pd.DataFrame, value_column: str, index_column: str, column_column: str, title: str, output_path: Path, as_percentage: bool = False) -> None:
    pivot = matrix_df.pivot(index=index_column, columns=column_column, values=value_column)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.values, cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            text = f"{value * 100:.1f}" if as_percentage else f"{value:.2f}"
            ax.text(col, row, text, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    save_figure(fig, output_path)


def plot_gap_vs_accuracy(non_diagonal_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for train_name, group in non_diagonal_df.groupby("train_dataset"):
        ax.scatter(group["gap"], group["test_accuracy"], label=train_name, alpha=0.8)
    ax.set_title("Dataset Gap vs OOD Accuracy")
    ax.set_xlabel("s-OTDD dataset gap")
    ax.set_ylabel("OOD test accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_path)


def plot_in_domain_bar(summary_rows: list[dict], output_path: Path) -> None:
    df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["dataset"], df["best_test_accuracy"] * 100)
    ax.set_title("In-Domain Best Test Accuracy")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, output_path)


def plot_distilled_gap(gap_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(gap_df["target"], gap_df["gap"])
    ax.set_title("Distilled Dataset Gap to Real Datasets")
    ax.set_xlabel("Target Dataset")
    ax.set_ylabel("s-OTDD gap")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, output_path)


def plot_distilled_cross_eval(cross_rows: list[dict], output_path: Path) -> None:
    df = pd.DataFrame(cross_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["test_dataset"], df["test_accuracy"] * 100)
    ax.set_title("Distilled Dataset Model Performance")
    ax.set_xlabel("Test Dataset")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, output_path)


def run_study(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))
    real_runs_dir = ensure_dir(output_dir / "real_dataset_runs")
    distilled_dir = ensure_dir(output_dir / "distillation")

    train_datasets = {}
    test_datasets = {}
    for dataset_name in args.datasets:
        train_dataset, test_dataset = load_dataset_pair(dataset_name, Path(args.data_root))
        train_datasets[dataset_name] = train_dataset
        test_datasets[dataset_name] = test_dataset
        print(f"Loaded {dataset_name}: train={len(train_dataset)} test={len(test_dataset)}")

    all_metrics = {}
    summaries = []
    cross_rows = []
    for dataset_name in args.datasets:
        model, metrics_df, summary = train_real_dataset(
            dataset_name=dataset_name,
            train_dataset=train_datasets[dataset_name],
            test_dataset=test_datasets[dataset_name],
            output_dir=real_runs_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            seed=args.seed,
            device=device,
        )
        all_metrics[dataset_name] = metrics_df
        summaries.append(summary)
        cross_rows.extend(evaluate_model_on_all(model, dataset_name, test_datasets, batch_size=args.batch_size, device=device))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "real_dataset_summary.csv", index=False)
    cross_eval_df = pd.DataFrame(cross_rows)
    cross_eval_df.to_csv(output_dir / "cross_dataset_performance.csv", index=False)

    gap_df = compute_gap_matrix(
        train_datasets=train_datasets,
        gap_sample_size=args.gap_sample_size,
        num_projections=args.num_projections,
        batch_size=args.distance_batch_size,
        device=device,
        output_dir=output_dir,
    )
    non_diagonal_df = compute_gap_correlation(cross_eval_df, gap_df, output_dir)

    distilled_dataset = build_gap_aware_distilled_dataset(
        train_datasets=train_datasets,
        gap_df=gap_df,
        ipc=args.distill_ipc,
        feature_sample_size=args.distill_source_sample_size,
        seed=args.seed,
        output_dir=distilled_dir,
    )
    distilled_metrics_df, distilled_cross_rows = train_distilled_dataset(
        distilled_dataset=distilled_dataset,
        test_datasets=test_datasets,
        output_dir=distilled_dir,
        epochs=args.distill_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )
    distilled_gap_df = compute_distilled_gaps(
        distilled_dataset=distilled_dataset,
        real_train_datasets=train_datasets,
        gap_sample_size=args.gap_sample_size,
        num_projections=args.num_projections,
        batch_size=args.distance_batch_size,
        device=device,
        output_dir=distilled_dir,
    )

    plot_training_curves(all_metrics, output_dir)
    plot_in_domain_bar(summaries, output_dir / "in_domain_accuracy_bar.png")
    plot_heatmap(cross_eval_df, "test_accuracy", "train_dataset", "test_dataset", "Cross-Dataset Accuracy Heatmap", output_dir / "cross_dataset_accuracy_heatmap.png", as_percentage=True)
    plot_heatmap(gap_df, "gap", "dataset_a", "dataset_b", "Dataset Gap Heatmap", output_dir / "dataset_gap_heatmap.png")
    plot_gap_vs_accuracy(non_diagonal_df, output_dir / "gap_vs_ood_accuracy.png")
    plot_distilled_gap(distilled_gap_df, distilled_dir / "distilled_gap_bar.png")
    plot_distilled_cross_eval(distilled_cross_rows, distilled_dir / "distilled_cross_dataset_accuracy.png")

    distilled_metrics_df.to_csv(distilled_dir / "distilled_epoch_metrics.csv", index=False)
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "datasets": args.datasets,
                "epochs": args.epochs,
                "distill_epochs": args.distill_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "val_split": args.val_split,
                "gap_sample_size": args.gap_sample_size,
                "num_projections": args.num_projections,
                "distance_batch_size": args.distance_batch_size,
                "distill_ipc": args.distill_ipc,
                "distill_source_sample_size": args.distill_source_sample_size,
                "device": str(device),
                "seed": args.seed,
            },
            file,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset MNIST-family study with gap-aware distillation.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default=str(Path("artifacts") / "mnist_otdd" / RESULTS_SUBDIR))
    parser.add_argument("--datasets", type=str, nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--distill-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--gap-sample-size", type=int, default=1200)
    parser.add_argument("--num-projections", type=int, default=200)
    parser.add_argument("--distance-batch-size", type=int, default=128)
    parser.add_argument("--distill-ipc", type=int, default=25)
    parser.add_argument("--distill-source-sample-size", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_study(parse_args())
