#!/usr/bin/env bash
# Full-charge validation retrieval/pair evaluation for the complete prespecified 1%/10% SupCon cohort.
# Requires a committed selection manifest with status=frozen.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=metric_label_val
#SBATCH --output=/path/to/results/slurm_logs/metric_label_val-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/metric_label_val-%A_%a.err
set -eo pipefail
cd /path/to/dIon
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PY=/path/to/conda-envs/dIon-env/bin/python
SELECTION_FILE=${SELECTION_FILE:-configs/evaluation/metric_learning_label_efficiency_evaluation.json}
mapfile -t CELLS < <("$PY" scripts/locked_test_selection.py --selection "$SELECTION_FILE" --list-representation)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$CELL" ]]; then echo 'Invalid array index.' >&2; exit 2; fi
IFS=$'\t' read -r MODEL CONDITIONING CORPUS <<< "$CELL"
COHORT=${COHORT:-primary}
if [[ "${COHORT}" != "primary" && "${COHORT}" != "charge2to4" ]]; then
  echo "Validation cohort must be primary or charge2to4, got ${COHORT}." >&2
  exit 2
fi
RESULTS_ROOT=${RESULTS_ROOT:-/path/to/results/metric_learning_label_efficiency_${COHORT}_validation_${SLURM_ARRAY_JOB_ID}}
srun --kill-on-bad-exit=1 "$PY" -u scripts/run_representation_benchmark.py --registry configs/evaluation/representation_benchmark_registry.yaml --cohort "$COHORT" --split validation --model "$MODEL" --conditioning "$CONDITIONING" --corpus "$CORPUS" --tasks both --accelerator gpu --extraction-batch-size auto --hash-checkpoint --output-root "$RESULTS_ROOT"
