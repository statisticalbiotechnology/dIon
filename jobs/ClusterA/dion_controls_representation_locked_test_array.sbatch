#!/usr/bin/env bash
# Locked held-out representation evaluation for fixed dIon controls only.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=controls_repr_test
#SBATCH --output=/path/to/results/slurm_logs/controls_repr_test-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/controls_repr_test-%A_%a.err
set -eo pipefail
cd /path/to/dIon
if [[ "${ALLOW_LOCKED_TEST:-0}" != 1 ]]; then
  echo 'Set ALLOW_LOCKED_TEST=1.' >&2
  exit 2
fi
export SELECTION_FILE=configs/evaluation/locked_test_dion_controls.json
export RESULTS_ROOT=/path/to/results/locked_test_dion_controls_representation_${SLURM_ARRAY_JOB_ID}
exec bash jobs/ClusterA/representation_selected_locked_test_array.sbatch
