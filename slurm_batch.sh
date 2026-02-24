#!/bin/bash

# Setup environment variables. sbatch passes all current env variables to the job.
# This runs `export HF_TOKEN=...`
. ~/.shell/secrets/export_hf_token.sh

# Submit batch
sbatch ./slurm_run_debug.sh
