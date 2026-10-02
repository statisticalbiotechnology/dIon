#!/usr/bin/env bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 01:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=instv122_king_shard
#SBATCH --output=/path/to/dIon/slurm_outs/instv122_king_shard-%A_%a.out
#SBATCH --error=/path/to/dIon/slurm_outs/instv122_king_shard-%A_%a.err
set -euo pipefail
exec bash /path/to/dIon/jobs/portable/instanovo_v122_kingdoms_shard.sh
