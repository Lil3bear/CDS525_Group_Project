# MNIST OTDD Distillation

This module extends the original project with a data distillation study driven by dataset distance.

## Research Question

Can `s-OTDD` distance between a full MNIST domain and a distilled subset predict downstream classification quality?

## Domains

- `clean`
- `rotate15`
- `rotate30`
- `noise`
- `invert`

## Distillation Strategies

- `random`
- `balanced_random`
- `prototype`
- `hard`

## Run

Create an environment with the packages in `mnist_otdd/requirements.txt`, then run from the project root:

```bash
python -m mnist_otdd.experiment --data-root ../s-OTDD/data --output-dir mnist_otdd/results --subset-sizes 200 500 --epochs 3 --domain-sample-size 2000
```

Outputs:

- `domain_distances.csv`: pairwise distances between MNIST domains
- `distillation_results.csv`: subset distance and classification metrics
- `domain_distance_heatmap.png`: heatmap for domain distances
- `distance_vs_accuracy.png`: scatter plot for distillation quality

## Cross-Dataset Study

This repo also includes a multi-dataset experiment for `MNIST`, `FashionMNIST`, `EMNISTDigits`, `USPS`, and `KMNIST`.

It does four things in one run:

- trains the same `SimpleCNN` on each dataset
- evaluates each trained model on the other datasets
- computes pairwise `s-OTDD` gaps between datasets
- builds a gap-aware distilled dataset by class-wise prototype fusion

Run:

```bash
python -m mnist_otdd.cross_dataset_study --data-root ../s-OTDD/data --output-dir mnist_otdd/cross_dataset_results --epochs 4 --distill-epochs 6
```
