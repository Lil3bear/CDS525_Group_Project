from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from kmnist_oss.model import SimpleCNN
from mnist_otdd.cross_dataset_study import (
    build_loader,
    compute_gap_matrix,
    ensure_dir,
    evaluate,
    load_dataset_pair,
    materialize_dataset,
    plot_heatmap,
    resolve_device,
    save_figure,
    set_seed,
    split_train_val,
    TensorDigitDataset,
    train_one_epoch,
)


class SoftLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, hard_labels: torch.Tensor, soft_labels: torch.Tensor):
        self.images = images.float()
        self.hard_labels = hard_labels.long()
        self.soft_labels = soft_labels.float()

    def __len__(self) -> int:
        return len(self.hard_labels)

    def __getitem__(self, index: int):
        return self.images[index], self.hard_labels[index], self.soft_labels[index]


def sample_subset(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    if len(dataset) <= sample_size:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    subset, _ = random_split(dataset, [sample_size, len(dataset) - sample_size], generator=generator)
    return subset


def train_simplecnn(
    train_dataset: Dataset,
    test_dataset: Dataset,
    run_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    seed: int,
    device: torch.device,
) -> tuple[Path, pd.DataFrame]:
    train_split, val_split_dataset = split_train_val(train_dataset, val_split=val_split, seed=seed)
    train_loader = build_loader(train_split, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = build_loader(val_split_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    best_test = -1.0
    best_checkpoint = output_dir / f"{run_name}_best_test.pt"
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        rows.append(
            {
                "run_name": run_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        if test_accuracy > best_test:
            best_test = test_accuracy
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_checkpoint)
        print(f"[{run_name}] epoch={epoch}/{epochs} val={val_accuracy:.4f} test={test_accuracy:.4f}")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / f"{run_name}_metrics.csv", index=False)
    return best_checkpoint, metrics_df


def evaluate_checkpoint_on_all(
    checkpoint_path: Path,
    train_name: str,
    datasets: dict[str, tuple[Dataset, Dataset]],
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    model = SimpleCNN().to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    loss_fn = nn.CrossEntropyLoss()
    rows = []
    for test_name, (_, test_dataset) in datasets.items():
        test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=42)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        rows.append(
            {
                "method": train_name,
                "checkpoint_path": str(checkpoint_path),
                "best_epoch": int(payload["epoch"]),
                "test_dataset": test_name,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
    return rows


def average_state_dicts(checkpoints: list[Path], device: torch.device) -> dict[str, torch.Tensor]:
    avg = None
    for ckpt in checkpoints:
        state = torch.load(ckpt, map_location=device)["model_state_dict"]
        if avg is None:
            avg = {k: v.detach().clone() for k, v in state.items()}
        else:
            for key in avg:
                avg[key] += state[key]
    assert avg is not None
    for key in avg:
        avg[key] /= len(checkpoints)
    return avg


class SimpleMoE(nn.Module):
    def __init__(self, experts: list[SimpleCNN]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        for expert in self.experts:
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, len(experts)),
        )

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=1)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (expert_logits * gate_weights.unsqueeze(-1)).sum(dim=1)


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 4):
        super().__init__()
        self.base = deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad = False
        self.down = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4):
        super().__init__()
        self.base = deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad = False
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))


class LoRASimpleCNN(SimpleCNN):
    def __init__(self, base_model: SimpleCNN, rank: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            LoRAConv2d(base_model.features[0], rank=rank),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            LoRAConv2d(base_model.features[3], rank=rank),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.embedding = LoRALinear(base_model.embedding, rank=rank)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.output = LoRALinear(base_model.output, rank=rank)


def train_model(model: nn.Module, train_dataset: Dataset, test_datasets: dict[str, tuple[Dataset, Dataset]], run_name: str, output_dir: Path, epochs: int, batch_size: int, learning_rate: float, seed: int, device: torch.device) -> list[dict]:
    train_loader = build_loader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"[{run_name}] epoch={epoch}/{epochs} train_acc={train_accuracy:.4f}")
    checkpoint = output_dir / f"{run_name}.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    rows = []
    for test_name, (_, test_dataset) in test_datasets.items():
        test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        rows.append({"method": run_name, "test_dataset": test_name, "test_loss": test_loss, "test_accuracy": test_accuracy})
    return rows


@torch.no_grad()
def build_distilled_dataset(
    train_datasets: dict[str, tuple[Dataset, Dataset]],
    expert_checkpoints: dict[str, Path],
    gap_df: pd.DataFrame,
    sample_size_per_domain: int,
    ipc: int,
    batch_size: int,
    device: torch.device,
) -> SoftLabelDataset:
    experts = {}
    for name, checkpoint in expert_checkpoints.items():
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
        model.eval()
        experts[name] = model

    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    weights = (1.0 / mean_gaps).to_dict()
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    images_by_domain = {}
    labels_by_domain = {}
    embeddings_by_domain = {}
    for idx, (name, (train_dataset, _)) in enumerate(train_datasets.items()):
        sampled = sample_subset(train_dataset, sample_size_per_domain, seed=100 + idx)
        images, labels = materialize_dataset(sampled)
        feature_dict = experts[name](images.to(device), return_features=True)
        images_by_domain[name] = images
        labels_by_domain[name] = labels
        embeddings_by_domain[name] = feature_dict["embedding"].cpu()

    selected_images = []
    selected_labels = []
    for label in range(10):
        embeds = []
        for name in train_datasets:
            mask = labels_by_domain[name] == label
            label_embeds = embeddings_by_domain[name][mask]
            if len(label_embeds) == 0:
                continue
            embeds.append(label_embeds)
        if not embeds:
            raise RuntimeError(f"No samples found for label {label} across source datasets.")
        all_embeds = torch.cat(embeds, dim=0).numpy()
        n_clusters = min(ipc, len(all_embeds))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(all_embeds)
        centers = kmeans.cluster_centers_
        chosen = []
        pool_embeds = torch.cat(embeds, dim=0)
        pool_images = torch.cat([images_by_domain[name][labels_by_domain[name] == label] for name in train_datasets if (labels_by_domain[name] == label).any()], dim=0)
        for center in centers:
            dists = torch.norm(pool_embeds - torch.tensor(center).float(), dim=1)
            idx = int(torch.argmin(dists))
            if idx in chosen:
                order = torch.argsort(dists)
                for alt in order.tolist():
                    if alt not in chosen:
                        idx = alt
                        break
            chosen.append(idx)
        selected_images.append(pool_images[chosen])
        selected_labels.append(torch.full((len(chosen),), label, dtype=torch.long))

    images = torch.cat(selected_images, dim=0)
    hard_labels = torch.cat(selected_labels, dim=0)
    soft_batches = []
    loader = DataLoader(TensorDigitDataset(images, hard_labels, "distilled"), batch_size=batch_size, shuffle=False)
    for batch_images, _ in loader:
        batch_images = batch_images.to(device)
        logits_sum = None
        for name, expert in experts.items():
            logits = expert(batch_images) * weights[name]
            logits_sum = logits if logits_sum is None else logits_sum + logits
        soft_batches.append(F.softmax(logits_sum, dim=1).cpu())
    soft_labels = torch.cat(soft_batches, dim=0)
    return SoftLabelDataset(images, hard_labels, soft_labels)


def train_student_on_soft(dataset: SoftLabelDataset, test_datasets: dict[str, tuple[Dataset, Dataset]], output_dir: Path, epochs: int, batch_size: int, learning_rate: float, temperature: float, alpha: float, seed: int, device: torch.device) -> list[dict]:
    student = SimpleCNN().to(device)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    rows = []
    for epoch in range(1, epochs + 1):
        student.train()
        total_examples = 0
        total_correct = 0
        for images, hard_labels, soft_labels in loader:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)
            optimizer.zero_grad()
            logits = student(images)
            kd = kl_loss(F.log_softmax(logits / temperature, dim=1), soft_labels) * (temperature ** 2)
            ce = ce_loss(logits, hard_labels)
            loss = alpha * kd + (1.0 - alpha) * ce
            loss.backward()
            optimizer.step()
            total_examples += hard_labels.size(0)
            total_correct += (logits.argmax(dim=1) == hard_labels).sum().item()
        print(f"[sotdd_distill] epoch={epoch}/{epochs} train_acc={total_correct / total_examples:.4f}")
    torch.save({"model_state_dict": student.state_dict()}, output_dir / "sotdd_distill_student.pt")
    for test_name, (_, test_dataset) in test_datasets.items():
        test_loader = build_loader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
        test_loss, test_accuracy = evaluate(student, test_loader, ce_loss, device)
        rows.append({"method": "sotdd_distill", "test_dataset": test_name, "test_loss": test_loss, "test_accuracy": test_accuracy})
    return rows


def plot_method_bars(result_df: pd.DataFrame, output_path: Path) -> None:
    pivot = result_df.pivot(index="method", columns="test_dataset", values="test_accuracy")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pivot.columns))
    width = 0.14
    for idx, method in enumerate(pivot.index):
        ax.bar(x + idx * width, pivot.loc[method].values * 100, width=width, label=method)
    ax.set_xticks(x + width * (len(pivot.index) - 1) / 2, pivot.columns)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Unified Benchmark Across MNIST-family Datasets")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, output_path)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))
    datasets = {name: load_dataset_pair(name, Path(args.data_root)) for name in args.datasets}

    expert_dir = ensure_dir(output_dir / "experts")
    expert_checkpoints = {}
    expert_summary_rows = []
    for name, (train_dataset, test_dataset) in datasets.items():
        checkpoint, metrics = train_simplecnn(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            run_name=f"expert_{name}",
            output_dir=expert_dir,
            epochs=args.individual_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            seed=args.seed,
            device=device,
        )
        expert_checkpoints[name] = checkpoint
        best_row = metrics.loc[metrics["test_accuracy"].idxmax()]
        expert_summary_rows.append({"method": f"expert_{name}", "test_dataset": name, "test_accuracy": float(best_row["test_accuracy"])})

    result_rows = []

    min_train_size = min(len(train_dataset) for train_dataset, _ in datasets.values())
    balanced_parts = []
    for idx, (name, (train_dataset, _)) in enumerate(datasets.items()):
        subset = sample_subset(train_dataset, min_train_size, seed=200 + idx)
        balanced_parts.append(subset)
    merged_train = ConcatDataset(balanced_parts)
    merged_test = datasets["MNIST"][1]
    merge_ckpt, _ = train_simplecnn(
        train_dataset=merged_train,
        test_dataset=merged_test,
        run_name="merge",
        output_dir=output_dir,
        epochs=args.merge_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        device=device,
    )
    result_rows.extend(evaluate_checkpoint_on_all(merge_ckpt, "merge", datasets, args.batch_size, device))

    soup_state = average_state_dicts(list(expert_checkpoints.values()), device)
    soup_model = SimpleCNN().to(device)
    soup_model.load_state_dict(soup_state)
    ce_loss = nn.CrossEntropyLoss()
    for test_name, (_, test_dataset) in datasets.items():
        test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)
        test_loss, test_accuracy = evaluate(soup_model, test_loader, ce_loss, device)
        result_rows.append({"method": "soup", "test_dataset": test_name, "test_loss": test_loss, "test_accuracy": test_accuracy})

    experts = []
    for checkpoint in expert_checkpoints.values():
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
        experts.append(model)
    moe = SimpleMoE(experts).to(device)
    result_rows.extend(train_model(moe, merged_train, datasets, "moe", output_dir, args.moe_epochs, args.batch_size, 1e-3, args.seed, device))

    base_model = SimpleCNN().to(device)
    base_model.load_state_dict(torch.load(merge_ckpt, map_location=device)["model_state_dict"])
    lora_model = LoRASimpleCNN(base_model, rank=args.lora_rank).to(device)
    result_rows.extend(train_model(lora_model, merged_train, datasets, "lora", output_dir, args.lora_epochs, args.batch_size, 1e-3, args.seed, device))

    gap_df = compute_gap_matrix({name: train for name, (train, _) in datasets.items()}, args.gap_sample_size, args.num_projections, args.distance_batch_size, device, output_dir)
    distilled_dataset = build_distilled_dataset(datasets, expert_checkpoints, gap_df, args.distill_source_sample_size, args.distill_ipc, args.batch_size, device)
    torch.save({"images": distilled_dataset.images, "hard_labels": distilled_dataset.hard_labels, "soft_labels": distilled_dataset.soft_labels}, output_dir / "sotdd_distilled_dataset.pt")
    result_rows.extend(train_student_on_soft(distilled_dataset, datasets, output_dir, args.distill_epochs, args.batch_size, args.learning_rate, args.temperature, args.alpha, args.seed, device))

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_dir / "method_results.csv", index=False)
    summary_df = result_df.groupby("method")["test_accuracy"].agg(["mean", "min", "max"]).reset_index()
    summary_df.to_csv(output_dir / "method_summary.csv", index=False)
    plot_heatmap(result_df, "test_accuracy", "method", "test_dataset", "Method Accuracy Heatmap", output_dir / "method_accuracy_heatmap.png", as_percentage=True)
    plot_method_bars(result_df, output_dir / "method_accuracy_bar.png")
    with open(output_dir / "benchmark_config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified benchmark for merge, soup, MoE, LoRA and s-OTDD-guided distillation.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--individual-epochs", type=int, default=12)
    parser.add_argument("--merge-epochs", type=int, default=12)
    parser.add_argument("--moe-epochs", type=int, default=8)
    parser.add_argument("--lora-epochs", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--gap-sample-size", type=int, default=300)
    parser.add_argument("--num-projections", type=int, default=80)
    parser.add_argument("--distance-batch-size", type=int, default=64)
    parser.add_argument("--distill-source-sample-size", type=int, default=1000)
    parser.add_argument("--distill-ipc", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
