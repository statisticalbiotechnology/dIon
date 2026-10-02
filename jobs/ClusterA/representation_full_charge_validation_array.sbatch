#!/usr/bin/env bash
# Full-charge raw DINO plus selected SupCon representation validation.
# PY=/path/to/conda-envs/dIon-env/bin/python
# N="$($PY scripts/run_representation_benchmark.py --list-cells --cohort primary --split validation --model-set all | wc -l)"
# sbatch --array="0-$((N - 1))" jobs/ClusterA/representation_full_charge_validation_array.sbatch
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=repr_fullval
#SBATCH --output=slurm_outs/repr_fullval-%A_%a.out
#SBATCH --error=slurm_outs/repr_fullval-%A_%a.err

set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export COHORT=primary
export SPLIT=validation
export MODEL_SET="${MODEL_SET:-all}"
export RESULTS_ROOT="${RESULTS_ROOT:-/path/to/results/full_charge_validation_${SLURM_ARRAY_JOB_ID}}"
exec bash jobs/ClusterA/representation_validation_array.sbatch
