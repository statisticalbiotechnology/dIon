#!/usr/bin/env bash
# InstaNovo v1.2.2 greedy decode on full-charge capped Kingdoms test.
# Smoke benchmark: 202.1 spectra/s end-to-end; nine hours includes I/O margin.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 09:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=instv122_kingdoms
#SBATCH --output=/path/to/dIon/slurm_outs/instv122_kingdoms-%j.out
#SBATCH --error=/path/to/dIon/slurm_outs/instv122_kingdoms-%j.err

set -eo pipefail
exec bash /path/to/dIon/jobs/portable/instanovo_v1_2_2_denovo_locked_test.sh 1
