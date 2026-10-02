#!/usr/bin/env bash
# One-GPU timing smoke for distributed-prediction equivalence and throughput estimation.
# Runs 100 real test batches at max_peaks=1000, then validates mzTab confidence output.

#SBATCH --job-name=denovo-predict-1g
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/runs/slurm/%x-%j.out
#SBATCH --error=/path/to/runs/slurm/%x-%j.err

set -euo pipefail
export PREDICTION_SOURCE=mskb
export PREDICTION_TIER=priority
export PREDICTION_NUM_NODES=1
export PREDICTION_NUM_DEVICES=1
export PREDICTION_LIMIT_TEST_BATCHES=100
source /path/to/dIon-runs/denovo-mskb-final/jobs/ClusterB/denovo_mskb_final_charge_lt5_prediction_common.sh
