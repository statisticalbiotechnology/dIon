#!/usr/bin/env bash
# One-GPU score-ranked prediction on the canonical charge-<5 MSKB-final test set.
# Source=v5; priority tier=priority. The 1000-peak recovery uses batch 25 and an 8 h limit.
# Submit with: sbatch jobs/ClusterB/denovo_dnlv1_charge_lt5_prediction_priority_1gpu_array.sbatch

#SBATCH --job-name=denovo-v5-pred-1g-prio
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err

set -euo pipefail
export PREDICTION_SOURCE=v5
export PREDICTION_TIER=priority
export PREDICTION_NUM_NODES=1
export PREDICTION_NUM_DEVICES=1
source /path/to/dIon-runs/denovo-mskb-final/jobs/ClusterB/denovo_mskb_final_charge_lt5_prediction_common.sh
