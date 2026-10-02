#!/usr/bin/env bash
# Locked-test retrieval/pair evaluation for validation-frozen registry candidates.
# Requires a committed selection manifest with status=frozen.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=repr_locked
#SBATCH --output=/path/to/results/slurm_logs/repr_locked-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/repr_locked-%A_%a.err
set -eo pipefail
cd /path/to/dIon
# An sbatch submitted from an interactive allocation can inherit a different
# Slurm memory-request variable. It conflicts with this job's GPU request when
# launching its inner srun; allocation limits remain enforced by Slurm itself.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
PY=/path/to/conda-envs/dIon-env/bin/python
if [[ "${ALLOW_LOCKED_TEST:-0}" != 1 ]]; then echo 'Set ALLOW_LOCKED_TEST=1.' >&2; exit 2; fi
SELECTION_FILE=${SELECTION_FILE:?Set SELECTION_FILE to a frozen JSON manifest.}
mapfile -t CELLS < <("$PY" scripts/locked_test_selection.py --selection "$SELECTION_FILE" --list-representation)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$CELL" ]]; then echo 'Invalid array index.' >&2; exit 2; fi
IFS=$'\t' read -r MODEL CONDITIONING CORPUS <<< "$CELL"
COHORT=${COHORT:-primary}
if [[ "$COHORT" != "primary" && "$COHORT" != "charge2to4" ]]; then echo "Unsupported cohort: $COHORT" >&2; exit 2; fi
RESULTS_ROOT=${RESULTS_ROOT:-/path/to/results/locked_test_representation_${COHORT}_${SLURM_ARRAY_JOB_ID}}
srun --kill-on-bad-exit=1 "$PY" -u scripts/run_representation_benchmark.py --registry configs/evaluation/representation_benchmark_registry.yaml --cohort "$COHORT" --split locked_test --allow-locked-test --model "$MODEL" --conditioning "$CONDITIONING" --corpus "$CORPUS" --tasks both --accelerator gpu --extraction-batch-size auto --hash-checkpoint --output-root "$RESULTS_ROOT"
