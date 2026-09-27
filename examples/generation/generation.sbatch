#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=daft-qwen38-27b
#SBATCH --array=0
#SBATCH --output=/e/project1/reformo/enrico/qwen-sglang/logs/%x_%A_%a.out

set -euo pipefail

export WANDB_MODE=offline

PROJECT_DIR="$SLURM_SUBMIT_DIR"
VENV="$PROJECT_DIR/.venv-sglang"
PY="$VENV/bin/python"
SCRIPT="$PROJECT_DIR/examples/generation/generation.py"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

"$PY" -u "$SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" "$@"