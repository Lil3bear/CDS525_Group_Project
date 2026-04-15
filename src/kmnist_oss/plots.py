from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def read_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def save_figure(fig, path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")


def plot_baseline():
    baseline = read_metrics(RESULTS_DIR / "baseline_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(baseline["epoch"], baseline["train_loss"], marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(baseline["epoch"], baseline["train_accuracy"] * 100, marker="o")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(baseline["epoch"], baseline["test_accuracy"] * 100, marker="o")
    axes[2].set_title("Test Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, alpha=0.3)

    save_figure(fig, FIGURES_DIR / "baseline_training_curves.png")


def plot_loss_comparison():
    baseline = read_metrics(RESULTS_DIR / "baseline_metrics.csv")
    alternate = read_metrics(RESULTS_DIR / "loss_label_smoothing_metrics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(baseline["epoch"], baseline["train_loss"], marker="o", label="Cross Entropy")
    axes[0].plot(alternate["epoch"], alternate["train_loss"], marker="o", label="Alternate Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        baseline["epoch"],
        baseline["train_accuracy"] * 100,
        marker="o",
        label="Cross Entropy",
    )
    axes[1].plot(
        alternate["epoch"],
        alternate["train_accuracy"] * 100,
        marker="o",
        label="Alternate Loss",
    )
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        baseline["epoch"],
        baseline["test_accuracy"] * 100,
        marker="o",
        label="Cross Entropy",
    )
    axes[2].plot(
        alternate["epoch"],
        alternate["test_accuracy"] * 100,
        marker="o",
        label="Alternate Loss",
    )
    axes[2].set_title("Test Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    save_figure(fig, FIGURES_DIR / "loss_comparison.png")


def plot_sweep(summary_file: str, value_column: str, prefix: str, value_label: str):
    summary = pd.read_csv(require_file(RESULTS_DIR / summary_file))

    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    acc_fig, acc_axes = plt.subplots(1, 2, figsize=(12, 4))

    for _, row in summary.sort_values(value_column).iterrows():
        metrics = pd.read_csv(require_file(Path(row["metrics_path"])))
        label = f"{value_label}={row[value_column]}"

        loss_ax.plot(metrics["epoch"], metrics["train_loss"], marker="o", label=label)
        acc_axes[0].plot(
            metrics["epoch"],
            metrics["train_accuracy"] * 100,
            marker="o",
            label=label,
        )
        acc_axes[1].plot(
            metrics["epoch"],
            metrics["test_accuracy"] * 100,
            marker="o",
            label=label,
        )

    loss_ax.set_title(f"{prefix.upper()} Sweep Training Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    acc_axes[0].set_title("Training Accuracy")
    acc_axes[0].set_xlabel("Epoch")
    acc_axes[0].set_ylabel("Accuracy (%)")
    acc_axes[0].grid(True, alpha=0.3)
    acc_axes[0].legend()

    acc_axes[1].set_title("Test Accuracy")
    acc_axes[1].set_xlabel("Epoch")
    acc_axes[1].set_ylabel("Accuracy (%)")
    acc_axes[1].grid(True, alpha=0.3)
    acc_axes[1].legend()

    save_figure(loss_fig, FIGURES_DIR / f"{prefix}_comparison_loss.png")
    save_figure(acc_fig, FIGURES_DIR / f"{prefix}_comparison_accuracy.png")


def main():
    plot_baseline()
    plot_loss_comparison()
    plot_sweep("lr_sweep_summary.csv", "learning_rate", "lr", "lr")
    plot_sweep("batch_sweep_summary.csv", "batch_size", "batch", "batch")


if __name__ == "__main__":
    main()
