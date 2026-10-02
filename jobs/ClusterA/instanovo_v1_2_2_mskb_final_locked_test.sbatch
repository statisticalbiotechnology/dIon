#!/usr/bin/env bash
# InstaNovo v1.2.2 greedy decode on the official MSKB-final charge-<5 test.
# Smoke benchmark: 230.5 spectra/s end-to-end; one hour is conservative.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=instv122_mskb
#SBATCH --output=/path/to/dIon/slurm_outs/instv122_mskb-%j.out
#SBATCH --error=/path/to/dIon/slurm_outs/instv122_mskb-%j.err

set -eo pipefail
exec bash /path/to/dIon/jobs/portable/instanovo_v1_2_2_denovo_locked_test.sh 0
