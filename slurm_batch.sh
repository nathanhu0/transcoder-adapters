#!/bin/bash

# Setup environment variables. sbatch passes all current env variables to the job.
# This runs `export HF_TOKEN=...`
export HF_TOKEN=$(cat ~/.shell/secrets/hf_token)

# Submit batch
sbatch ./slurm_run_debug.sh "$@"
