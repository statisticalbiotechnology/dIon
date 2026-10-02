#!/usr/bin/env bash
# One-GPU score-ranked prediction on the canonical charge-<5 MSKB-final test set.
# Source=v5; lower-priority tier.
# Submit with: sbatch jobs/ClusterB/denovo_dnlv1_charge_lt5_prediction_lower_1gpu_array.sbatch

#SBATCH --job-name=denovo-v5-pred-1g-low
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err

set -euo pipefail
export PREDICTION_SOURCE=v5
export PREDICTION_TIER=lower
export PREDICTION_NUM_NODES=1
export PREDICTION_NUM_DEVICES=1
source /path/to/dIon-runs/denovo-mskb-final/jobs/ClusterB/denovo_mskb_final_charge_lt5_prediction_common.sh
