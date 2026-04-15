#!/usr/bin/env bash
#SBATCH --job-name=simplecnn-scdd
#SBATCH --output=slurm-simplecnn-scdd-%j.out
#SBATCH --error=slurm-simplecnn-scdd-%j.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

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
export TEACHER_ROOT="artifacts/mnist_otdd/cross_dataset_full5/real_dataset_runs"
export OUTPUT_DIR="artifacts/mnist_otdd/simplecnn_scdd_full_gpu"
export DEVICE="cuda"

bash scripts/run_simplecnn_scdd_full.sh
