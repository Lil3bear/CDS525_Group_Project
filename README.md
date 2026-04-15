# MNIST Digit Recognition OSS

Open-source repository for two related tracks:

- `kmnist_oss`: a clean PyTorch baseline for KMNIST handwriting recognition
- `mnist_otdd`: advanced MNIST-style domain-distance and distillation experiments

## Repository Layout

```text
src/
  kmnist_oss/      # baseline training, sweeps, plots, prediction visualization
  mnist_otdd/      # OTDD-based cross-domain and distillation experiments
scripts/           # shell and Slurm launchers
docs/              # legacy project notes kept for reference
artifacts/         # generated during runs; ignored by git
data/              # local datasets; ignored by git
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For the OTDD experiments, you may also need the `otdd` package from the related `s-OTDD` codebase if it is not available in your environment.

## Baseline KMNIST

Train:

```bash
python -m kmnist_oss.train --run-name baseline --lr 0.001 --batch-size 64 --epochs 12 --loss cross_entropy
```

Run sweeps:

```bash
python -m kmnist_oss.experiments --epochs 12
```

Generate plots:

```bash
python -m kmnist_oss.plots
```

Visualize predictions:

```bash
python -m kmnist_oss.visualize_predictions --checkpoint artifacts/checkpoints/best_baseline.pt
```

## OTDD Experiments

Single distillation study:

```bash
python -m mnist_otdd.experiment --data-root data --output-dir artifacts/mnist_otdd/results --subset-sizes 200 500 --epochs 3 --domain-sample-size 2000
```

Cross-dataset study:

```bash
python -m mnist_otdd.cross_dataset_study --data-root data --output-dir artifacts/mnist_otdd/cross_dataset_results --epochs 4 --distill-epochs 6
```

## Notes

- Generated outputs are written under `artifacts/`.
- Raw datasets are expected under `data/`.
- Legacy project notes and the original course readme are preserved in `docs/`.
