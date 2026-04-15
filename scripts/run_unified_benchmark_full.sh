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
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/mnist_otdd/unified_benchmark_full_run1}"
DEVICE="${DEVICE:-cuda}"

python -m mnist_otdd.unified_benchmark \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --datasets MNIST FashionMNIST EMNISTDigits USPS KMNIST \
  --individual-epochs 12 \
  --merge-epochs 12 \
  --moe-epochs 12 \
  --lora-epochs 12 \
  --distill-epochs 30 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --val-split 0.1 \
  --gap-sample-size 500 \
  --num-projections 120 \
  --distance-batch-size 128 \
  --distill-source-sample-size 2000 \
  --distill-ipc 30 \
  --temperature 4.0 \
  --alpha 0.8 \
  --lora-rank 4 \
  --device "$DEVICE" \
  --seed 42
