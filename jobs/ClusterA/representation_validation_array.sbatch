#!/usr/bin/env bash
# Registry-driven offline representation evaluation.
#
# The default is validation only. Locked test requires both SPLIT=locked_test
# and ALLOW_LOCKED_TEST=1, plus ready artifacts in the registry.
#
# Submit all current validation cells:
#   PY=/path/to/conda-envs/dIon-env/bin/python
#   N="$($PY scripts/run_representation_benchmark.py --list-cells \
#     --cohort charge2to4 --split validation --model-set all | wc -l)"
#   sbatch --array="0-$((N - 1))" jobs/ClusterA/representation_validation_array.sbatch
#
# Future raw-DINO additions need only a registry entry, then:
#   N="$($PY scripts/run_representation_benchmark.py --list-cells \
#     --cohort charge2to4 --split validation --model-set raw_dino | wc -l)"
#   sbatch --array="0-$((N - 1))" --export=ALL,MODEL_SET=raw_dino \
#     jobs/ClusterA/representation_validation_array.sbatch

#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=repr_eval
#SBATCH --output=slurm_outs/repr_eval-%A_%a.out
#SBATCH --error=slurm_outs/repr_eval-%A_%a.err

set -eo pipefail

cd /path/to/dIon
mkdir -p slurm_outs

# Mambaforge's shell hook may reference CONDA_DEFAULT_ENV.
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

PY="${PYTHON_BIN:-/path/to/conda-envs/dIon-env/bin/python}"
REGISTRY="${REGISTRY:-configs/evaluation/representation_benchmark_registry.yaml}"
COHORT="${COHORT:-charge2to4}"
SPLIT="${SPLIT:-validation}"
MODEL_SET="${MODEL_SET:-all}"
ALLOW_LOCKED_TEST="${ALLOW_LOCKED_TEST:-0}"

if [[ "${SPLIT}" == "locked_test" && "${ALLOW_LOCKED_TEST}" != "1" ]]; then
  echo "Refusing locked test without ALLOW_LOCKED_TEST=1" >&2
  exit 2
fi

LIST_ARGS=(--registry "${REGISTRY}" --list-cells --cohort "${COHORT}" --split "${SPLIT}" --model-set "${MODEL_SET}")
RUN_ARGS=(--registry "${REGISTRY}" --cohort "${COHORT}" --split "${SPLIT}" --accelerator gpu --extraction-batch-size auto)
if [[ "${SPLIT}" == "locked_test" ]]; then
  LIST_ARGS+=(--allow-locked-test)
  RUN_ARGS+=(--allow-locked-test)
fi

mapfile -t CELLS < <("${PY}" scripts/run_representation_benchmark.py "${LIST_ARGS[@]}")
if (( ${#CELLS[@]} == 0 )); then
  echo "Registry resolved zero cells." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CELLS[@]} )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} outside ${#CELLS[@]} registry cells." >&2
  exit 2
fi

IFS=$'\t' read -r MODEL CONDITIONING CORPUS <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
RESULTS_ROOT="${RESULTS_ROOT:-results/representation/dion_${COHORT}_${SPLIT}__slurm_${SLURM_ARRAY_JOB_ID}}"

echo "start=$(date --iso-8601=seconds)"
echo "model=${MODEL}; conditioning=${CONDITIONING}; corpus=${CORPUS}"
echo "cohort=${COHORT}; split=${SPLIT}; model_set=${MODEL_SET}"
echo "results_root=${RESULTS_ROOT}"

srun --kill-on-bad-exit=1 "${PY}" -u scripts/run_representation_benchmark.py \
  "${RUN_ARGS[@]}" \
  --model "${MODEL}" \
  --conditioning "${CONDITIONING}" \
  --corpus "${CORPUS}" \
  --tasks both \
  --output-root "${RESULTS_ROOT}"

echo "done=$(date --iso-8601=seconds)"
