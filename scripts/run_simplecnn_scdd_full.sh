#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-data}"
TEACHER_ROOT="${TEACHER_ROOT:-artifacts/mnist_otdd/cross_dataset_full5/real_dataset_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/mnist_otdd/simplecnn_scdd_full_gpu}"
DEVICE="${DEVICE:-cuda}"

python -m mnist_otdd.simplecnn_scdd \
  --data-root "$DATA_ROOT" \
  --teacher-root "$TEACHER_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --datasets MNIST FashionMNIST EMNISTDigits USPS KMNIST \
  --epochs 30 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --temperature 4.0 \
  --alpha 0.8 \
  --gap-sample-size 500 \
  --num-projections 120 \
  --distance-batch-size 128 \
  --teacher-stat-sample-size 4000 \
  --ipc 20 \
  --recover-steps 400 \
  --recover-lr 0.03 \
  --feature-weight 1.0 \
  --entropy-weight 0.02 \
  --gap-weight-ratio 0.5 \
  --device "$DEVICE"
