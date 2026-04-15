#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/mnist_otdd/resnet50_teacher_distill_run1}"
DEVICE="${DEVICE:-cuda}"

python -m mnist_otdd.resnet50_teacher_distill_resume \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --datasets MNIST FashionMNIST EMNISTDigits USPS KMNIST \
  --teacher-epochs 5 \
  --teacher-batch-size 16 \
  --teacher-learning-rate 0.0001 \
  --student-epochs 20 \
  --student-batch-size 128 \
  --student-learning-rate 0.001 \
  --val-split 0.1 \
  --gap-sample-size 300 \
  --num-projections 80 \
  --distance-batch-size 64 \
  --teacher-stat-sample-size 2000 \
  --ipc 20 \
  --recover-steps 200 \
  --recover-lr 0.03 \
  --feature-weight 1.0 \
  --entropy-weight 0.02 \
  --gap-weight-ratio 0.5 \
  --temperature 4.0 \
  --alpha 0.8 \
  --device "$DEVICE" \
  --seed 42
