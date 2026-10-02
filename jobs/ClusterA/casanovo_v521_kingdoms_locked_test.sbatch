#!/usr/bin/env bash
# Stock Casanovo v5.2.1 on the full-denominator Kingdoms held-out test.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 16:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=cas_v521_kingdoms
#SBATCH --output=/path/to/dIon/slurm_outs/cas_v521_kingdoms-%j.out
#SBATCH --error=/path/to/dIon/slurm_outs/cas_v521_kingdoms-%j.err

set -eo pipefail
exec bash /path/to/dIon/jobs/portable/casanovo_v521_denovo_locked_test.sh 1
