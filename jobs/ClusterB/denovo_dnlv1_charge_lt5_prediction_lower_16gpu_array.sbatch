#!/usr/bin/env bash
# Score-ranked prediction on the canonical charge-<5 MSKB-final held-out test set.
# Source=v5; priority tier=lower.
# Submit with: sbatch jobs/ClusterB/denovo_dnlv1_charge_lt5_prediction_lower_16gpu_array.sbatch

#SBATCH --job-name=denovo-v5-predict-low
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err

set -euo pipefail
export PREDICTION_SOURCE=v5
export PREDICTION_TIER=lower
source /path/to/dIon-runs/denovo-mskb-final/jobs/ClusterB/denovo_mskb_final_charge_lt5_prediction_common.sh
