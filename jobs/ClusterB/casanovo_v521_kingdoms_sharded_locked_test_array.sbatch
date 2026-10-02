#!/usr/bin/env bash
# Submit with: sbatch --array=0-15%16 jobs/ClusterB/casanovo_v521_kingdoms_sharded_locked_test_array.sbatch
# Site paths/environment are deliberately supplied through exported variables.
#SBATCH --job-name=cas-v521-king-shard
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err
set -euo pipefail
: "${DION_ROOT:?Export the ClusterB dIon checkout path.}"
: "${CASANOVO_V521_ENV:?Export the isolated Casanovo v5.2.1 environment.}"
export ALLOW_LOCKED_TEST=1
exec bash "$DION_ROOT/jobs/portable/casanovo_v521_kingdoms_shard.sh"
