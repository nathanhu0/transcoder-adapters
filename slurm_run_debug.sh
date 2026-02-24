#!/bin/bash
#SBATCH --account=nlp
#SBATCH --gres=gpu:4
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --partition=jag-hi
#SBATCH --job-name=gemma2_2b
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── Usage ────────────────────────────────────────────────────────────
#   Run ./slurm_batch. Or you can do `sbatch slurm_run_debug.sh`, but
#   will need to set up env variables yourself in that case.
# ─────────────────────────────────────────────────────────────────────

# ── Logs & Outputs ──────────────────────────────────────────────────
#
# SLURM stdout/stderr:
#   logs/<job_id>.out   and   logs/<job_id>.err
#   (location from repo root)
#
#   To find your job ID after submitting:
#     squeue --me
#
#   To tail logs of a running job:
#     tail -f logs/<job_id>.out
#
# Training checkpoints & wandb artifacts:
#   Written to the output directory configured in the YAML config
#   (defaults are printed at the start of training in stdout).
# ─────────────────────────────────────────────────────────────────────

mkdir -p logs

echo "[Slurm] Setting up (uv sync)..."

uv sync

echo "[Slurm] Running Python..."

# uv run python -m training.train --config training/configs/r1_distil_1.5b_debug.yaml
uv run python -m training.train --config training/configs/gemma2_2b.yaml
# uv run python -m training.train --config training/configs/gemma2_2b.yaml

echo "[Slurm] Job finished!"
