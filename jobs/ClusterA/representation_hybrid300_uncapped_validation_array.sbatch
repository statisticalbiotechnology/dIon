#!/usr/bin/env bash
# Hybrid-300-last raw backbone embeddings with uncapped peak arrays at inference.
# This is an isolated validation ablation: conditioned/null x three charge-2--4 corpora.
# Submit: sbatch --array=0-5 jobs/ClusterA/representation_hybrid300_uncapped_validation_array.sbatch
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5
#SBATCH --job-name=repr_h300_uncap
#SBATCH --output=/path/to/results/slurm_logs/repr_h300_uncap-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/repr_h300_uncap-%A_%a.err
set -eo pipefail

cd /path/to/dIon
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

PY=/path/to/conda-envs/dIon-env/bin/python
RESULTS_ROOT="${RESULTS_ROOT:-/path/to/results/hybrid300_uncapped_validation_${SLURM_ARRAY_JOB_ID}}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) CONDITIONING=conditioned; CORPUS=bacterial ;;
  1) CONDITIONING=conditioned; CORPUS=ninespecies_v2 ;;
  2) CONDITIONING=conditioned; CORPUS=kingdoms ;;
  3) CONDITIONING=null; CORPUS=bacterial ;;
  4) CONDITIONING=null; CORPUS=ninespecies_v2 ;;
  5) CONDITIONING=null; CORPUS=kingdoms ;;
  *) exit 2 ;;
esac

if [[ "${CONDITIONING}" == "null" ]]; then
  BATCH_SIZE="${UNCAPPED_EXTRACTION_BATCH_SIZE:-16}"
else
  BATCH_SIZE="${UNCAPPED_EXTRACTION_BATCH_SIZE:-128}"
fi

echo "start=$(date --iso-8601=seconds)"
echo "model=hybrid_300_last; conditioning=${CONDITIONING}; corpus=${CORPUS}; max_peaks=0"
echo "results_root=${RESULTS_ROOT}; extraction_batch_size=${BATCH_SIZE}"

srun --kill-on-bad-exit=1 "${PY}" -u scripts/run_representation_benchmark.py \
  --cohort charge2to4 \
  --split validation \
  --model hybrid_300_last \
  --conditioning "${CONDITIONING}" \
  --corpus "${CORPUS}" \
  --tasks both \
  --accelerator gpu \
  --extraction-batch-size "${BATCH_SIZE}" \
  --max-peaks-override 0 \
  --output-root "${RESULTS_ROOT}"

echo "done=$(date --iso-8601=seconds)"
