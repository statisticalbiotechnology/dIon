#!/usr/bin/env bash
# Stock Casanovo v5.2.1 on the official MSKB-final charge-<5 held-out test.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=cas_v521_mskb
#SBATCH --output=/path/to/dIon/slurm_outs/cas_v521_mskb-%j.out
#SBATCH --error=/path/to/dIon/slurm_outs/cas_v521_mskb-%j.err

set -eo pipefail
exec bash /path/to/dIon/jobs/portable/casanovo_v521_denovo_locked_test.sh 0
