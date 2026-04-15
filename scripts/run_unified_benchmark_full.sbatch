#!/usr/bin/env bash
#SBATCH --job-name=mnist-unified
#SBATCH --output=slurm-unified-benchmark-%j.out
#SBATCH --error=slurm-unified-benchmark-%j.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export DATA_ROOT="data"
export OUTPUT_DIR="artifacts/mnist_otdd/unified_benchmark_full_run1"
export DEVICE="cuda"

bash scripts/run_unified_benchmark_full.sh
