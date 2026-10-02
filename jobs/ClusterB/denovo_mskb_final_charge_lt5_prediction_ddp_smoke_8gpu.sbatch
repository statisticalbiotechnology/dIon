#!/usr/bin/env bash
# Actual two-node DDP smoke for distributed mzTab prediction writing.
# It runs two test batches per rank from the charge-limited canonical MSKB-final test split.

#SBATCH --job-name=denovo-predict-ddp-smoke
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=/path/to/runs/slurm/%x-%j.out
#SBATCH --error=/path/to/runs/slurm/%x-%j.err

set -euo pipefail
export PREDICTION_SOURCE=mskb
export PREDICTION_TIER=priority
export PREDICTION_NUM_NODES=2
export PREDICTION_NUM_DEVICES=4
export PREDICTION_LIMIT_TEST_BATCHES=2
source /path/to/dIon-runs/denovo-mskb-final/jobs/ClusterB/denovo_mskb_final_charge_lt5_prediction_common.sh
