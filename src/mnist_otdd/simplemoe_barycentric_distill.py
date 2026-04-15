from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from kmnist_oss.model import SimpleCNN
from mnist_otdd.cross_dataset_study import (
    TensorDigitDataset,
    build_loader,
    compute_distilled_gaps,
    denormalize_tensor,
    ensure_dir,
    evaluate,
    evaluate_model_on_all,
    load_dataset_pair,
    materialize_dataset,
    plot_distilled_cross_eval,
    plot_distilled_gap,
    resolve_device,
    save_figure,
    set_seed,
)
from mnist_otdd.resnet50_sodd_hybrid_distill import (
    SoftLabelDataset,
    build_budget_plan,
    pick_anchor_pool,
    initialize_synthetic_images,
)


class SimpleMoETeacher(nn.Module):
    def __init__(self, experts: list[SimpleCNN]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, len(experts)),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=1)
        expert_features = [expert(x, return_features=True) for expert in self.experts]
        expert_logits = torch.stack([feat["logits"] for feat in expert_features], dim=1)
        expert_embeddings = torch.stack([feat["embedding"] for feat in expert_features], dim=1)
        logits = (expert_logits * gate_weights.unsqueeze(-1)).sum(dim=1)
        embedding = (expert_embeddings * gate_weights.unsqueeze(-1)).sum(dim=1)
        if return_features:
            return {"logits": logits, "embedding": embedding, "gate_weights": gate_weights}
        return logits


def load_simplemoe_teacher(benchmark_dir: Path, datasets: list[str], device: torch.device) -> SimpleMoETeacher:
    experts = []
    for name in datasets:
        ckpt = benchmark_dir / "experts" / f"expert_{name}_best_test.pt"
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        model.eval()
        experts.append(model)
    teacher = SimpleMoETeacher(experts).to(device)
    teacher.load_state_dict(torch.load(benchmark_dir / "moe.pt", map_location=device)["model_state_dict"], strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def collect_teacher_bank(
    teacher: SimpleMoETeacher,
    train_datasets: dict[str, Dataset],
    sample_size_per_dataset: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    bank = {}
    for idx, (name, dataset) in enumerate(train_datasets.items()):
        images, labels = materialize_dataset(dataset)
        if len(labels) > sample_size_per_dataset:
            chosen = torch.randperm(len(labels), generator=torch.Generator().manual_seed(100 + idx))[:sample_size_per_dataset]
            images = images[chosen]
            labels = labels[chosen]
        embeds = []
        probs = []
        for start in range(0, len(labels), batch_size):
            batch = images[start : start + batch_size].to(device)
            outputs = teacher(batch, return_features=True)
            embeds.append(outputs["embedding"].cpu())
            probs.append(F.softmax(outputs["logits"], dim=1).cpu())
        bank[name] = {
            "images": images.cpu(),
            "labels": labels.cpu(),
            "embeddings": torch.cat(embeds, dim=0),
            "probs": torch.cat(probs, dim=0),
        }
    return bank


def build_barycentric_domain_weights(
    bank: dict[str, dict[str, torch.Tensor]],
    datasets: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    global_centers = {}
    for label in range(10):
        pooled = []
        for name in datasets:
            mask = bank[name]["labels"] == label
            if mask.sum() > 0:
                pooled.append(bank[name]["embeddings"][mask])
        global_centers[label] = torch.cat(pooled, dim=0).mean(dim=0)

    dists = {}
    for name in datasets:
        accum = 0.0
        count = 0
        for label in range(10):
            mask = bank[name]["labels"] == label
            if mask.sum() == 0:
                continue
            local_center = bank[name]["embeddings"][mask].mean(dim=0)
            accum += float(F.mse_loss(local_center, global_centers[label]))
            count += 1
        dists[name] = accum / max(count, 1)
    total = sum(dists.values())
    weights = {name: dists[name] / total for name in datasets}
    return weights, dists


def build_class_targets(bank: dict[str, dict[str, torch.Tensor]], domain_weights: dict[str, float]) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    prototypes = {}
    spreads = {}
    for label in range(10):
        rows = []
        for name, payload in bank.items():
            mask = payload["labels"] == label
            if mask.sum() > 0:
                rows.append(payload["embeddings"][mask] * domain_weights[name])
        feats = torch.cat(rows, dim=0)
        prototypes[label] = feats.mean(dim=0)
        spreads[label] = feats.std(dim=0).clamp_min(1e-4)
    return prototypes, spreads


def optimize_dataset(
    teacher: SimpleMoETeacher,
    bank: dict[str, dict[str, torch.Tensor]],
    anchor_pool,
    prototypes,
    spreads,
    domain_class_budgets,
    steps: int,
    outer_lr: float,
    feature_weight: float,
    recon_weight: float,
    tv_weight: float,
    optimize_batch_size: int,
    seed: int,
    device: torch.device,
) -> SoftLabelDataset:
    raw_images, hard_labels, soft_targets = initialize_synthetic_images(anchor_pool, domain_class_budgets, seed, 0.98, 0.0025)
    raw = torch.atanh(raw_images.clamp(-0.999, 0.999)).to(device).detach().requires_grad_(True)
    hard_labels = hard_labels.to(device)
    soft_targets = soft_targets.to(device)
    optimizer = optim.Adam([raw], lr=outer_lr)
    step_generator = torch.Generator().manual_seed(seed + 2027)

    for step in range(steps):
        optimizer.zero_grad()
        full_synth = torch.tanh(raw)
        subset_indices = torch.randperm(len(hard_labels), generator=step_generator)[: min(optimize_batch_size, len(hard_labels))].to(device)
        synth = full_synth[subset_indices]
        subset_labels = hard_labels[subset_indices]
        subset_soft = soft_targets[subset_indices]
        outputs = teacher(synth, return_features=True)
        teacher_probs = F.softmax(outputs["logits"], dim=1)
        teacher_ce = F.cross_entropy(outputs["logits"], subset_labels)
        teacher_kd = F.kl_div(torch.log(teacher_probs.clamp_min(1e-8)), subset_soft, reduction="batchmean")
        feature_loss = torch.zeros((), device=device)
        recon_loss = torch.zeros((), device=device)
        label_count = 0
        for label in range(10):
            mask = subset_labels == label
            if mask.sum() == 0:
                continue
            synth_label = synth[mask]
            embed = outputs["embedding"][mask]
            feature_loss = feature_loss + F.mse_loss(embed.mean(dim=0), prototypes[label].to(device))
            feature_loss = feature_loss + 0.1 * F.mse_loss(embed.std(dim=0), spreads[label].to(device))
            anchor_images = torch.cat([payload["images"] for payload in anchor_pool[label].values()], dim=0).to(device)
            nearest = anchor_images[torch.cdist(synth_label.flatten(1), anchor_images.flatten(1)).argmin(dim=1)]
            recon_loss = recon_loss + F.mse_loss(synth_label, nearest)
            label_count += 1
        if label_count > 0:
            feature_loss = feature_loss / label_count
            recon_loss = recon_loss / label_count
        tv_loss = (
            (synth[:, :, 1:, :] - synth[:, :, :-1, :]).pow(2).mean()
            + (synth[:, :, :, 1:] - synth[:, :, :, :-1]).pow(2).mean()
        )
        loss = teacher_ce + teacher_kd + feature_weight * feature_loss + recon_weight * recon_loss + tv_weight * tv_loss
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"[simplemoe-synth] step={step+1}/{steps} ce={teacher_ce.item():.4f} kd={teacher_kd.item():.4f} feat={feature_loss.item():.4f} recon={recon_loss.item():.4f}")

    return SoftLabelDataset(torch.tanh(raw).detach().cpu(), hard_labels.cpu(), soft_targets.cpu())


def train_student(dataset: SoftLabelDataset, test_datasets: dict[str, Dataset], epochs: int, batch_size: int, learning_rate: float, alpha: float, temperature: float, seed: int, device: torch.device, output_dir: Path):
    student = SimpleCNN().to(device)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for epoch in range(1, epochs + 1):
        student.train()
        total = 0
        correct = 0
        for images, hard_labels, soft_labels in loader:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)
            optimizer.zero_grad()
            logits = student(images)
            kd = kl_loss(F.log_softmax(logits / temperature, dim=1), soft_labels) * (temperature ** 2)
            ce = ce_loss(logits, hard_labels)
            loss = alpha * kd + (1 - alpha) * ce
            loss.backward()
            optimizer.step()
            total += hard_labels.size(0)
            correct += (logits.argmax(dim=1) == hard_labels).sum().item()
        print(f"[simplemoe-student] epoch={epoch}/{epochs} train_acc={correct / max(total,1):.4f}")
    rows = pd.DataFrame(evaluate_model_on_all(student, "SimpleMoEBaryStudent", test_datasets, batch_size=batch_size, device=device))
    rows.to_csv(output_dir / "student_cross_eval.csv", index=False)
    torch.save({"model_state_dict": student.state_dict()}, output_dir / "student_last.pt")
    return rows


def save_preview(images: torch.Tensor, labels: torch.Tensor, output_path: Path) -> None:
    count = min(100, len(labels))
    preview = denormalize_tensor(images[:count]).numpy()
    preview_labels = labels[:count].numpy()
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < count:
            ax.imshow(preview[idx, 0], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(str(int(preview_labels[idx])), fontsize=8)
        ax.axis("off")
    save_figure(fig, output_path)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir))
    datasets = args.datasets
    teacher = load_simplemoe_teacher(Path(args.benchmark_dir), datasets, device)
    train_datasets = {}
    test_datasets = {}
    for name in datasets:
        tr, te = load_dataset_pair(name, Path(args.data_root))
        train_datasets[name] = tr
        test_datasets[name] = te
    bank = collect_teacher_bank(teacher, train_datasets, args.prototype_sample_size, args.batch_size, device)
    domain_weights, center_distances = build_barycentric_domain_weights(bank, datasets)
    budget_plan = build_budget_plan(domain_weights, train_datasets, args.synth_total_budget, 20, False)
    prototypes, spreads = build_class_targets(bank, domain_weights)
    anchor_pool = pick_anchor_pool(bank, domain_weights, prototypes, args.anchor_pool_per_class, args.confidence_weight)
    distilled = optimize_dataset(
        teacher, bank, anchor_pool, prototypes, spreads, budget_plan["domain_class_budgets"],
        steps=args.optimize_steps, outer_lr=args.optimize_lr, feature_weight=args.feature_weight,
        recon_weight=args.recon_weight, tv_weight=args.tv_weight, optimize_batch_size=args.optimize_batch_size,
        seed=args.seed, device=device,
    )
    torch.save({"images": distilled.images, "hard_labels": distilled.hard_labels, "soft_labels": distilled.soft_labels}, output_dir / "hybrid_distilled_dataset.pt")
    save_preview(distilled.images, distilled.hard_labels, output_dir / "hybrid_preview.png")
    cross_eval = train_student(distilled, test_datasets, args.student_epochs, args.batch_size, args.student_learning_rate, args.alpha, args.temperature, args.seed, device, output_dir)
    gap_to_real = compute_distilled_gaps(TensorDigitDataset(distilled.images, distilled.hard_labels, "SimpleMoEBaryDistilled"), train_datasets, args.gap_sample_size, args.num_projections, args.distance_batch_size, device, output_dir)
    plot_distilled_gap(gap_to_real, output_dir / "hybrid_gap_bar.png")
    plot_distilled_cross_eval(cross_eval.to_dict("records"), output_dir / "hybrid_cross_accuracy.png")
    summary = {
        "teacher_type": "SimpleMoE",
        "student_mean_cross_accuracy": float(cross_eval["test_accuracy"].mean()),
        "student_min_cross_accuracy": float(cross_eval["test_accuracy"].min()),
        "student_max_cross_accuracy": float(cross_eval["test_accuracy"].max()),
        "requested_total_budget": args.synth_total_budget,
        "effective_total_budget": budget_plan["effective_total_budget"],
        "center_distances": center_distances,
        "domain_budgets": budget_plan["domain_budgets"],
        "domain_weights": domain_weights,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barycentric distillation with SimpleMoE teacher.")
    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--student-epochs", type=int, default=12)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument("--prototype-sample-size", type=int, default=2500)
    parser.add_argument("--anchor-pool-per-class", type=int, default=80)
    parser.add_argument("--synth-total-budget", type=int, required=True)
    parser.add_argument("--confidence-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gap-sample-size", type=int, default=150)
    parser.add_argument("--num-projections", type=int, default=32)
    parser.add_argument("--distance-batch-size", type=int, default=64)
    parser.add_argument("--optimize-steps", type=int, default=20)
    parser.add_argument("--optimize-lr", type=float, default=0.05)
    parser.add_argument("--feature-weight", type=float, default=0.35)
    parser.add_argument("--recon-weight", type=float, default=5.0)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--optimize-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
