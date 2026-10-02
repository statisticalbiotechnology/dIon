#!/usr/bin/env bash
# Materialize locked charge-2--4 retrieval and static-pair artifacts for
# GLEAMS/Casanovo-v5.2 matched external-reference evaluation.
# Submit: sbatch jobs/ClusterA/materialize_charge2to4_locked_test_array.sbatch
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --job-name=charge2to4_test
#SBATCH --output=slurm_outs/charge2to4_test-%A_%a.out
#SBATCH --error=slurm_outs/charge2to4_test-%A_%a.err

set -eo pipefail

cd /path/to/dIon
mkdir -p slurm_outs
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
set -u

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    NAME=ninespecies_v2
    SOURCE_ROOT=/path/to/data/probing_datasets/ninespecies_v2_paper_test
    ;;
  1)
    NAME=bacterial
    SOURCE_ROOT=/path/to/data/probing_datasets/bacterial_paper_benchmarks/test_peptide_disjoint
    ;;
  2)
    NAME=kingdoms
    SOURCE_ROOT=/path/to/data/probing_datasets/kingdoms_paper_benchmarks/test
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="${SOURCE_ROOT}/charge2to4"
echo "Materializing ${NAME}: ${SOURCE_ROOT} -> ${OUTPUT_ROOT}"

srun --kill-on-bad-exit=1 python -u \
  scripts/data_prep/materialize_charge_filtered_embedding_benchmark.py \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --min-charge 2 \
  --max-charge 4 \
  --seed 20260817
