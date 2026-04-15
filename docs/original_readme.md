# KMNIST One-Day Project

Simple PyTorch CNN for KMNIST handwriting recognition with baseline training, required experiments, plots, and prediction visualization.

## Recommended Setup

The current machine has Python 3.14 as the default `python`, but PyTorch support is more reliable on Python 3.10. Use the Python 3.10 launcher:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Commands

Baseline training:

```powershell
python train.py --run-name baseline --lr 0.001 --batch-size 64 --epochs 12 --loss cross_entropy
```

All required experiments:

```powershell
python experiments.py --epochs 12
```

Generate all plots:

```powershell
python plots.py
```

Visualize the first 100 test predictions:

```powershell
python visualize_predictions.py --checkpoint checkpoints/best_baseline.pt
```

MNIST domain distance + data distillation experiment:

```powershell
python -m mnist_otdd.experiment --data-root ../s-OTDD/data --output-dir mnist_otdd/results --subset-sizes 200 500 --epochs 3 --domain-sample-size 2000
```

## Output Locations

- `results/`: per-run metrics CSVs, config JSONs, sweep summaries
- `figures/`: all PNG charts and the first-100 prediction grid
- `checkpoints/`: saved best model weights
- `mnist_otdd/results/`: MNIST domain distance and distillation outputs
