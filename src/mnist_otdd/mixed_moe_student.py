from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from kmnist_oss.model import SimpleCNN
from mnist_otdd.cross_dataset_study import build_loader, ensure_dir, evaluate, load_dataset_pair, resolve_device, set_seed
from mnist_otdd.resnet50_sodd_hybrid_distill import split_integer_evenly


class SimpleMoE(nn.Module):
    def __init__(self, experts: list[SimpleCNN]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
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


class LabelTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index])


def reconstruct_domain_ids(summary: dict, datasets: list[str]) -> torch.Tensor:
    domain_budgets = summary["domain_budgets"]
    domain_ids = []
    for label in range(10):
        for domain_index, domain_name in enumerate(datasets):
            count = split_integer_evenly(int(domain_budgets[domain_name]), 10)[label]
            domain_ids.extend([domain_index] * count)
    return torch.tensor(domain_ids, dtype=torch.long)


def train_classifier(
    model: nn.Module,
    train_dataset: Dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    tag: str,
) -> nn.Module:
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"[{tag}] epoch={epoch}/{epochs} train_acc={total_correct / max(total,1):.4f}")
    return model


def train_gate(
    moe: SimpleMoE,
    train_dataset: Dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> SimpleMoE:
    for expert in moe.experts:
        expert.eval()
        for p in expert.parameters():
            p.requires_grad = False
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = optim.Adam(moe.gate.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        moe.train()
        total_correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = moe(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"[mixed-gate] epoch={epoch}/{epochs} train_acc={total_correct / max(total,1):.4f}")
    return moe


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))

    payload = torch.load(args.distilled_dataset, map_location="cpu")
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    synth_images = payload["images"].float()
    synth_labels = payload["hard_labels"].long()
    datasets = args.datasets
    domain_ids = reconstruct_domain_ids(summary, datasets)
    if len(domain_ids) != len(synth_labels):
        raise RuntimeError(f"Domain id count mismatch: {len(domain_ids)} vs {len(synth_labels)}")

    real_train = {}
    test_datasets = {}
    for name in datasets:
        train_dataset, test_dataset = load_dataset_pair(name, Path(args.data_root))
        real_train[name] = train_dataset
        test_datasets[name] = test_dataset

    experts = []
    expert_counts = []
    mixed_gate_parts = []
    for domain_index, domain_name in enumerate(datasets):
        synth_idx = torch.nonzero(domain_ids == domain_index, as_tuple=False).squeeze(1)
        synth_subset = LabelTensorDataset(synth_images[synth_idx], synth_labels[synth_idx])
        mixed_train = ConcatDataset([real_train[domain_name], synth_subset])
        expert = SimpleCNN().to(device)
        train_classifier(
            expert,
            mixed_train,
            epochs=args.expert_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + domain_index,
            device=device,
            tag=f"mixed-expert-{domain_name}",
        )
        experts.append(expert)
        expert_counts.append({
            "domain": domain_name,
            "real_samples": len(real_train[domain_name]),
            "synthetic_samples": int(len(synth_idx)),
            "total_samples": len(real_train[domain_name]) + int(len(synth_idx)),
        })
        mixed_gate_parts.append(mixed_train)

    moe = SimpleMoE(experts).to(device)
    gate_train = ConcatDataset(mixed_gate_parts)
    train_gate(
        moe,
        gate_train,
        epochs=args.gate_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )

    ce_loss = nn.CrossEntropyLoss()
    rows = []
    for test_name, dataset_test in test_datasets.items():
        test_loader = build_loader(dataset_test, batch_size=args.batch_size, shuffle=False, seed=args.seed)
        test_loss, test_accuracy = evaluate(moe, test_loader, ce_loss, device)
        rows.append({"train_dataset": "MixedRealSyntheticMoE", "test_dataset": test_name, "test_loss": test_loss, "test_accuracy": test_accuracy})

    cross_eval = pd.DataFrame(rows)
    cross_eval.to_csv(output_dir / "mixed_moe_cross_eval.csv", index=False)
    pd.DataFrame(expert_counts).to_csv(output_dir / "expert_counts.csv", index=False)
    torch.save({"model_state_dict": moe.state_dict()}, output_dir / "mixed_moe_last.pt")
    result_summary = {
        "mean_cross_accuracy": float(cross_eval["test_accuracy"].mean()),
        "min_cross_accuracy": float(cross_eval["test_accuracy"].min()),
        "max_cross_accuracy": float(cross_eval["test_accuracy"].max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(result_summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SimpleMoE on all real data plus domain-partitioned synthetic data.")
    parser.add_argument("--distilled-dataset", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--expert-epochs", type=int, default=8)
    parser.add_argument("--gate-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
