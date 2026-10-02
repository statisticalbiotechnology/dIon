#!/usr/bin/env bash
# High-memory one-GPU prediction for the MSKB-final 1000-peak Hybrid-300 model.
# The 1000-peak path exceeded the default 23-GB CPU allocation.
# Submit with: sbatch jobs/ClusterB/denovo_mskb_final_charge_lt5_prediction_maxpeaks1000_1gpu.sbatch

#SBATCH --job-name=denovo-mskb-pred-p1000
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err

set -euo pipefail
export PREDICTION_SOURCE=mskb
export PREDICTION_TIER=priority
export SLURM_ARRAY_TASK_ID=0
export PREDICTION_NUM_NODES=1
export PREDICTION_NUM_DEVICES=1
export PREDICTION_NUM_WORKERS=1
source /path/to/dIon-runs/denovo-mskb-final/jobs/ClusterB/denovo_mskb_final_charge_lt5_prediction_common.sh
