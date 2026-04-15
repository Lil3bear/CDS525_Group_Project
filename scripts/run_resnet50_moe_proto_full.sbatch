#!/usr/bin/env bash
#SBATCH --job-name=res50-moe-proto
#SBATCH --output=slurm-res50-moe-proto-%j.out
#SBATCH --error=slurm-res50-moe-proto-%j.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
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
export OUTPUT_DIR="artifacts/mnist_otdd/resnet50_moe_proto_full_run1"
export DEVICE="cuda"

bash scripts/run_resnet50_moe_proto_full.sh
