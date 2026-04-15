from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    compute_gap_matrix,
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
from mnist_otdd.resnet50_moe_proto_distill import load_teacher, to_resnet_input


class SoftLabelDataset(Dataset):
    def __init__(self, images: torch.Tensor, hard_labels: torch.Tensor, soft_labels: torch.Tensor):
        self.images = images.float()
        self.hard_labels = hard_labels.long()
        self.soft_labels = soft_labels.float()

    def __len__(self) -> int:
        return len(self.hard_labels)

    def __getitem__(self, index: int):
        return self.images[index], self.hard_labels[index], self.soft_labels[index]


def allocate_quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {name: weights[name] * total for name in weights}
    quotas = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(weights, key=lambda name: raw[name] - quotas[name], reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def split_integer_evenly(total: int, parts: int) -> list[int]:
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def build_budget_plan(
    domain_weights: dict[str, float],
    train_datasets: dict[str, Dataset],
    requested_total_budget: int | None,
    fallback_images_per_class: int,
    cap_to_max_train_size: bool,
    num_classes: int = 10,
) -> dict[str, object]:
    max_train_size = max(len(dataset) for dataset in train_datasets.values())
    if requested_total_budget is None:
        effective_total_budget = fallback_images_per_class * num_classes
    else:
        effective_total_budget = min(requested_total_budget, max_train_size) if cap_to_max_train_size else requested_total_budget
    domain_budgets = allocate_quotas(domain_weights, effective_total_budget)
    domain_class_budgets = {
        name: split_integer_evenly(domain_budgets[name], num_classes) for name in domain_weights
    }
    class_totals = {
        label: sum(domain_class_budgets[name][label] for name in domain_weights) for label in range(num_classes)
    }
    return {
        "max_train_size": max_train_size,
        "effective_total_budget": effective_total_budget,
        "domain_budgets": domain_budgets,
        "domain_class_budgets": domain_class_budgets,
        "class_totals": class_totals,
    }


@torch.no_grad()
def collect_teacher_bank(
    teacher,
    train_datasets: dict[str, Dataset],
    sample_size_per_dataset: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    bank: dict[str, dict[str, torch.Tensor]] = {}
    for idx, (name, dataset) in enumerate(train_datasets.items()):
        images, labels = materialize_dataset(dataset)
        if len(labels) > sample_size_per_dataset:
            chosen = torch.randperm(len(labels), generator=torch.Generator().manual_seed(100 + idx))[:sample_size_per_dataset]
            images = images[chosen]
            labels = labels[chosen]
        rows = []
        probs = []
        for start in range(0, len(labels), batch_size):
            batch_images = images[start : start + batch_size].to(device)
            outputs = teacher(to_resnet_input(batch_images), return_features=True)
            rows.append(outputs["embedding"].cpu())
            probs.append(F.softmax(outputs["logits"], dim=1).cpu())
        bank[name] = {
            "images": images.cpu(),
            "labels": labels.cpu(),
            "embeddings": torch.cat(rows, dim=0),
            "probs": torch.cat(probs, dim=0),
        }
    return bank


def build_domain_weights(gap_df: pd.DataFrame, datasets: list[str], mode: str) -> dict[str, float]:
    if mode == "uniform":
        weight = 1.0 / len(datasets)
        return {name: weight for name in datasets}
    if mode == "barycentric":
        raise ValueError("Barycentric weights require teacher-bank statistics.")
    if mode != "sodd":
        raise ValueError(f"Unsupported allocation mode: {mode}")
    mean_gaps = gap_df[gap_df["dataset_a"] != gap_df["dataset_b"]].groupby("dataset_a")["gap"].mean()
    inv_gap = (1.0 / mean_gaps).to_dict()
    total = sum(inv_gap.values())
    return {name: inv_gap[name] / total for name in datasets}


def build_barycentric_domain_weights(
    bank: dict[str, dict[str, torch.Tensor]],
    datasets: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    global_centers: dict[int, torch.Tensor] = {}
    for label in range(10):
        pooled = []
        for name in datasets:
            payload = bank[name]
            mask = payload["labels"] == label
            if mask.sum() > 0:
                pooled.append(payload["embeddings"][mask])
        if not pooled:
            raise RuntimeError(f"No embeddings found for label {label} when building barycentric center.")
        global_centers[label] = torch.cat(pooled, dim=0).mean(dim=0)

    center_distances: dict[str, float] = {}
    for name in datasets:
        payload = bank[name]
        accum = 0.0
        count = 0
        for label in range(10):
            mask = payload["labels"] == label
            if mask.sum() == 0:
                continue
            domain_center = payload["embeddings"][mask].mean(dim=0)
            global_center = global_centers[label]
            accum += float(F.mse_loss(domain_center, global_center, reduction="mean"))
            count += 1
        center_distances[name] = accum / max(count, 1)

    total = sum(center_distances.values())
    if total <= 0:
        weight = 1.0 / len(datasets)
        return {name: weight for name in datasets}, center_distances
    weights = {name: center_distances[name] / total for name in datasets}
    return weights, center_distances


def apply_domain_bonus(domain_weights: dict[str, float], bonus_spec: list[str] | None) -> dict[str, float]:
    if not bonus_spec:
        return domain_weights
    weighted = dict(domain_weights)
    for item in bonus_spec:
        if ":" not in item:
            raise ValueError(f"Invalid domain bonus spec: {item}")
        name, value = item.split(":", 1)
        if name not in weighted:
            raise ValueError(f"Unknown domain in bonus spec: {name}")
        weighted[name] *= float(value)
    total = sum(weighted.values())
    return {name: value / total for name, value in weighted.items()}


def build_class_targets(
    bank: dict[str, dict[str, torch.Tensor]],
    domain_weights: dict[str, float],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    prototypes = {}
    spreads = {}
    for label in range(10):
        weighted_rows = []
        for name, payload in bank.items():
            mask = payload["labels"] == label
            if mask.sum() == 0:
                continue
            weight = domain_weights[name]
            weighted_rows.append(payload["embeddings"][mask] * weight)
        features = torch.cat(weighted_rows, dim=0)
        prototypes[label] = features.mean(dim=0)
        spreads[label] = features.std(dim=0).clamp_min(1e-4)
    return prototypes, spreads


def pick_anchor_pool(
    bank: dict[str, dict[str, torch.Tensor]],
    domain_weights: dict[str, float],
    prototypes: dict[int, torch.Tensor],
    anchors_per_class: int,
    confidence_weight: float,
) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    anchor_pool: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    quotas = allocate_quotas(domain_weights, anchors_per_class)
    for label in range(10):
        anchor_pool[label] = {}
        for domain_name, payload in bank.items():
            mask = payload["labels"] == label
            if mask.sum() == 0:
                continue
            domain_images = payload["images"][mask]
            domain_embeddings = payload["embeddings"][mask]
            domain_probs = payload["probs"][mask]
            class_confidence = domain_probs[:, label]
            distances = (domain_embeddings - prototypes[label]).pow(2).mean(dim=1)
            scores = -distances + confidence_weight * class_confidence
            quota = min(quotas[domain_name], len(domain_images))
            if quota <= 0:
                continue
            chosen = torch.topk(scores, k=quota).indices
            anchor_pool[label][domain_name] = {
                "images": domain_images[chosen],
                "soft_labels": domain_probs[chosen],
            }
    return anchor_pool


def initialize_synthetic_images(
    anchor_pool: dict[int, dict[str, dict[str, torch.Tensor]]],
    domain_class_budgets: dict[str, list[int]],
    seed: int,
    primary_weight: float,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    synth_images = []
    synth_labels = []
    synth_soft_labels = []
    for label in range(10):
        local_images = []
        local_soft = []
        for domain_name, label_budgets in domain_class_budgets.items():
            count_needed = label_budgets[label]
            if count_needed <= 0 or domain_name not in anchor_pool[label]:
                continue
            anchors = anchor_pool[label][domain_name]["images"]
            soft_labels = anchor_pool[label][domain_name]["soft_labels"]
            count = len(anchors)
            for idx in range(count_needed):
                first = (2 * idx) % count
                second = (2 * idx + 1) % count
                lam = primary_weight if first != second else 1.0
                base = lam * anchors[first] + (1.0 - lam) * anchors[second]
                noise = noise_scale * torch.randn_like(base)
                local_images.append((base + noise).clamp(-1.0, 1.0))
                soft = (lam * soft_labels[first] + (1.0 - lam) * soft_labels[second]).clamp_min(1e-6)
                local_soft.append(soft / soft.sum())
        if not local_images:
            raise RuntimeError(f"No synthetic images initialized for label {label}.")
        count_for_label = len(local_images)
        synth_images.append(torch.stack(local_images))
        synth_labels.append(torch.full((count_for_label,), label, dtype=torch.long))
        synth_soft_labels.append(torch.stack(local_soft))
    return torch.cat(synth_images, dim=0), torch.cat(synth_labels, dim=0), torch.cat(synth_soft_labels, dim=0)


def flatten_grads(grads: list[torch.Tensor | None]) -> torch.Tensor:
    parts = []
    for grad in grads:
        if grad is None:
            continue
        parts.append(grad.reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(1)


def gradient_match_loss(
    student: SimpleCNN,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    synth_images: torch.Tensor,
    synth_labels: torch.Tensor,
) -> torch.Tensor:
    real_logits = student(real_images)
    real_loss = F.cross_entropy(real_logits, real_labels)
    real_grads = torch.autograd.grad(real_loss, student.parameters(), retain_graph=False, create_graph=False, allow_unused=True)
    real_vec = flatten_grads(list(real_grads)).detach()

    synth_logits = student(synth_images)
    synth_loss = F.cross_entropy(synth_logits, synth_labels)
    synth_grads = torch.autograd.grad(synth_loss, student.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
    synth_vec = flatten_grads(list(synth_grads))

    cosine = 1.0 - F.cosine_similarity(synth_vec.unsqueeze(0), real_vec.unsqueeze(0), dim=1).mean()
    mse = F.mse_loss(synth_vec, real_vec)
    return cosine + 0.1 * mse


def sample_real_batch(
    bank: dict[int, dict[str, torch.Tensor]],
    label: int,
    batch_size: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = bank[label]["images"]
    if len(images) <= batch_size:
        chosen = torch.arange(len(images))
    else:
        generator = torch.Generator().manual_seed(1234 + 97 * label + step)
        chosen = torch.randperm(len(images), generator=generator)[:batch_size]
    selected = images[chosen]
    labels = torch.full((len(selected),), label, dtype=torch.long)
    return selected, labels


def optimize_synthetic_dataset(
    teacher,
    teacher_bank: dict[str, dict[str, torch.Tensor]],
    anchor_pool: dict[int, dict[str, dict[str, torch.Tensor]]],
    prototypes: dict[int, torch.Tensor],
    spreads: dict[int, torch.Tensor],
    domain_class_budgets: dict[str, list[int]],
    steps: int,
    outer_lr: float,
    gm_weight: float,
    feature_weight: float,
    recon_weight: float,
    tv_weight: float,
    batch_size: int,
    optimize_batch_size: int,
    init_primary_weight: float,
    init_noise_scale: float,
    seed: int,
    device: torch.device,
) -> SoftLabelDataset:
    raw_images, hard_labels, soft_targets = initialize_synthetic_images(
        anchor_pool,
        domain_class_budgets,
        seed,
        init_primary_weight,
        init_noise_scale,
    )
    raw = torch.atanh(raw_images.clamp(-0.999, 0.999)).to(device).detach().requires_grad_(True)
    hard_labels = hard_labels.to(device)
    soft_targets = soft_targets.to(device)
    optimizer = optim.Adam([raw], lr=outer_lr)

    real_bank = {}
    for label in range(10):
        class_images = []
        for payload in teacher_bank.values():
            mask = payload["labels"] == label
            if mask.sum() > 0:
                class_images.append(payload["images"][mask])
        real_bank[label] = {"images": torch.cat(class_images, dim=0)}

    teacher.eval()
    step_generator = torch.Generator().manual_seed(seed + 2026)
    for step in range(steps):
        optimizer.zero_grad()
        full_synth = torch.tanh(raw)
        if len(hard_labels) <= optimize_batch_size:
            subset_indices = torch.arange(len(hard_labels), device=device)
        else:
            subset_indices = torch.randperm(len(hard_labels), generator=step_generator)[:optimize_batch_size].to(device)
        synth = full_synth[subset_indices]
        subset_labels = hard_labels[subset_indices]
        subset_soft_targets = soft_targets[subset_indices]

        teacher_outputs = teacher(to_resnet_input(synth), return_features=True)
        teacher_probs = F.softmax(teacher_outputs["logits"], dim=1)

        teacher_ce = F.cross_entropy(teacher_outputs["logits"], subset_labels)
        teacher_kd = F.kl_div(torch.log(teacher_probs.clamp_min(1e-8)), subset_soft_targets, reduction="batchmean")
        feature_loss = torch.zeros((), device=device)
        recon_loss = torch.zeros((), device=device)
        gm_loss = torch.zeros((), device=device)
        label_count = 0

        for label in range(10):
            mask = subset_labels == label
            if mask.sum() == 0:
                continue
            synth_label = synth[mask]
            outputs_label = teacher_outputs["embedding"][mask]
            feature_loss = feature_loss + F.mse_loss(outputs_label.mean(dim=0), prototypes[label].to(device))
            feature_loss = feature_loss + 0.1 * F.mse_loss(outputs_label.std(dim=0), spreads[label].to(device))

            anchor_images = torch.cat([payload["images"] for payload in anchor_pool[label].values()], dim=0).to(device)
            pairwise = torch.cdist(synth_label.flatten(1), anchor_images.flatten(1))
            nearest = anchor_images[pairwise.argmin(dim=1)]
            recon_loss = recon_loss + F.mse_loss(synth_label, nearest)

            student = SimpleCNN().to(device)
            real_images, real_labels = sample_real_batch(real_bank, label, batch_size, step)
            gm_loss = gm_loss + gradient_match_loss(
                student,
                real_images.to(device),
                real_labels.to(device),
                synth_label,
                torch.full((len(synth_label),), label, dtype=torch.long, device=device),
            )
            label_count += 1

        if label_count > 0:
            feature_loss = feature_loss / label_count
            recon_loss = recon_loss / label_count
            gm_loss = gm_loss / label_count

        tv_loss = (
            (synth[:, :, 1:, :] - synth[:, :, :-1, :]).pow(2).mean()
            + (synth[:, :, :, 1:] - synth[:, :, :, :-1]).pow(2).mean()
        )
        loss = teacher_ce + teacher_kd + feature_weight * feature_loss + recon_weight * recon_loss + gm_weight * gm_loss + tv_weight * tv_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % max(1, steps // 10) == 0:
            print(
                f"[hybrid-synth] step={step + 1}/{steps} "
                f"teacher_ce={teacher_ce.item():.4f} teacher_kd={teacher_kd.item():.4f} "
                f"feature={feature_loss.item():.4f} recon={recon_loss.item():.4f} gm={gm_loss.item():.4f}"
            )

    final_images = torch.tanh(raw).detach().cpu()
    final_labels = hard_labels.detach().cpu()
    final_soft = soft_targets.detach().cpu()
    return SoftLabelDataset(final_images, final_labels, final_soft)


def train_student(
    dataset: SoftLabelDataset,
    test_datasets: dict[str, Dataset],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    alpha: float,
    temperature: float,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    student = SimpleCNN().to(device)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    rows = []
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for images, hard_labels, soft_labels in train_loader:
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
            batch_size_local = hard_labels.size(0)
            total_loss += loss.item() * batch_size_local
            total_correct += (logits.argmax(dim=1) == hard_labels).sum().item()
            total_examples += batch_size_local
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
        }
        for test_name, dataset_test in test_datasets.items():
            test_loader = build_loader(dataset_test, batch_size=batch_size, shuffle=False, seed=seed)
            _, accuracy = evaluate(student, test_loader, ce_loss, device)
            row[f"test_accuracy_{test_name}"] = accuracy
        rows.append(row)
        print(f"[hybrid-student] epoch={epoch}/{epochs} train_acc={row['train_accuracy']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "student_metrics.csv", index=False)
    cross_eval_df = pd.DataFrame(evaluate_model_on_all(student, "SODDHybridStudent", test_datasets, batch_size=batch_size, device=device))
    cross_eval_df.to_csv(output_dir / "student_cross_eval.csv", index=False)
    torch.save({"model_state_dict": student.state_dict()}, output_dir / "student_last.pt")
    return metrics_df, cross_eval_df


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
    teacher = load_teacher(Path(args.teacher_checkpoint), device)

    train_datasets = {}
    test_datasets = {}
    for name in args.datasets:
        train_dataset, test_dataset = load_dataset_pair(name, Path(args.data_root))
        train_datasets[name] = train_dataset
        test_datasets[name] = test_dataset

    bank = collect_teacher_bank(teacher, train_datasets, args.prototype_sample_size, args.batch_size, device)
    center_distances = None
    if args.allocation_mode == "barycentric":
        domain_weights, center_distances = build_barycentric_domain_weights(bank, args.datasets)
    else:
        gap_df = compute_gap_matrix(train_datasets, args.gap_sample_size, args.num_projections, args.distance_batch_size, device, output_dir)
        domain_weights = build_domain_weights(gap_df, args.datasets, args.allocation_mode)
    domain_weights = apply_domain_bonus(domain_weights, args.domain_bonus)
    budget_plan = build_budget_plan(
        domain_weights,
        train_datasets,
        args.synth_total_budget,
        args.images_per_class,
        args.cap_to_max_train_size,
    )

    prototypes, spreads = build_class_targets(bank, domain_weights)
    anchor_pool = pick_anchor_pool(bank, domain_weights, prototypes, args.anchor_pool_per_class, args.confidence_weight)

    distilled = optimize_synthetic_dataset(
        teacher=teacher,
        teacher_bank=bank,
        anchor_pool=anchor_pool,
        prototypes=prototypes,
        spreads=spreads,
        domain_class_budgets=budget_plan["domain_class_budgets"],
        steps=args.optimize_steps,
        outer_lr=args.optimize_lr,
        gm_weight=args.gm_weight,
        feature_weight=args.feature_weight,
        recon_weight=args.recon_weight,
        tv_weight=args.tv_weight,
        batch_size=args.gm_real_batch_size,
        optimize_batch_size=args.optimize_batch_size,
        init_primary_weight=args.init_primary_weight,
        init_noise_scale=args.init_noise_scale,
        seed=args.seed,
        device=device,
    )

    torch.save(
        {"images": distilled.images, "hard_labels": distilled.hard_labels, "soft_labels": distilled.soft_labels},
        output_dir / "hybrid_distilled_dataset.pt",
    )
    save_preview(distilled.images, distilled.hard_labels, output_dir / "hybrid_preview.png")

    _, cross_eval = train_student(
        dataset=distilled,
        test_datasets=test_datasets,
        epochs=args.student_epochs,
        batch_size=args.batch_size,
        learning_rate=args.student_learning_rate,
        alpha=args.alpha,
        temperature=args.temperature,
        seed=args.seed,
        device=device,
        output_dir=output_dir,
    )

    gap_to_real = compute_distilled_gaps(
        TensorDigitDataset(distilled.images, distilled.hard_labels, "SODDHybridDistilled"),
        train_datasets,
        args.gap_sample_size,
        args.num_projections,
        args.distance_batch_size,
        device,
        output_dir,
    )
    plot_distilled_gap(gap_to_real, output_dir / "hybrid_gap_bar.png")
    plot_distilled_cross_eval(cross_eval.to_dict("records"), output_dir / "hybrid_cross_accuracy.png")

    summary = {
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "student_mean_cross_accuracy": float(cross_eval["test_accuracy"].mean()),
        "student_min_cross_accuracy": float(cross_eval["test_accuracy"].min()),
        "student_max_cross_accuracy": float(cross_eval["test_accuracy"].max()),
        "images_per_class": args.images_per_class,
        "requested_total_budget": args.synth_total_budget,
        "effective_total_budget": budget_plan["effective_total_budget"],
        "max_train_size": budget_plan["max_train_size"],
        "allocation_mode": args.allocation_mode,
        "domain_bonus": args.domain_bonus,
        "center_distances": center_distances,
        "domain_budgets": budget_plan["domain_budgets"],
        "class_totals": budget_plan["class_totals"],
        "domain_weights": domain_weights,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid s-ODD guided dataset distillation for SimpleCNN.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "FashionMNIST", "EMNISTDigits", "USPS", "KMNIST"])
    parser.add_argument("--allocation-mode", choices=["sodd", "uniform", "barycentric"], default="sodd")
    parser.add_argument("--domain-bonus", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument("--prototype-sample-size", type=int, default=5000)
    parser.add_argument("--anchor-pool-per-class", type=int, default=40)
    parser.add_argument("--images-per-class", type=int, default=20)
    parser.add_argument("--synth-total-budget", type=int, default=None)
    parser.add_argument("--cap-to-max-train-size", action="store_true")
    parser.add_argument("--confidence-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gap-sample-size", type=int, default=300)
    parser.add_argument("--num-projections", type=int, default=80)
    parser.add_argument("--distance-batch-size", type=int, default=64)
    parser.add_argument("--optimize-steps", type=int, default=200)
    parser.add_argument("--optimize-lr", type=float, default=0.05)
    parser.add_argument("--gm-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=2.0)
    parser.add_argument("--tv-weight", type=float, default=1e-3)
    parser.add_argument("--gm-real-batch-size", type=int, default=32)
    parser.add_argument("--optimize-batch-size", type=int, default=512)
    parser.add_argument("--init-primary-weight", type=float, default=0.9)
    parser.add_argument("--init-noise-scale", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
