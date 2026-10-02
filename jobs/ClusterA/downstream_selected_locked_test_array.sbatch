#!/usr/bin/env bash
# Locked-test downstream evaluation selected solely by a frozen validation manifest.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=downstream_locked
#SBATCH --output=/path/to/results/slurm_logs/downstream_locked-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/downstream_locked-%A_%a.err
set -eo pipefail
cd /path/to/dIon
# An sbatch submitted from an interactive allocation can inherit a different
# Slurm memory-request variable. It conflicts with this job's GPU request when
# launching its inner srun; allocation limits remain enforced by Slurm itself.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
PY=/path/to/conda-envs/dIon-env/bin/python
if [[ "${ALLOW_LOCKED_TEST:-0}" != 1 ]]; then echo 'Set ALLOW_LOCKED_TEST=1.' >&2; exit 2; fi
SELECTION_FILE=${SELECTION_FILE:?Set SELECTION_FILE to a frozen JSON manifest.}
eval "$("$PY" scripts/locked_test_selection.py --selection "$SELECTION_FILE" --downstream-index "$SLURM_ARRAY_TASK_ID")"
# `not_used` and `not_applicable` are reporting labels; the CLI uses null.
if [[ "$PRECURSOR_CONDITIONING" == "not_used" || "$PRECURSOR_CONDITIONING" == "not_applicable" ]]; then PRECURSOR_CONDITIONING=null; fi
if [[ ! -f "$DOWNSTREAM_CHECKPOINT" ]]; then echo "Missing selected checkpoint: $DOWNSTREAM_CHECKPOINT" >&2; exit 3; fi
if [[ -n "${EXTERNAL_EMBEDDING_CACHE:-}" && ! -f "$EXTERNAL_EMBEDDING_CACHE/test.pt" ]]; then echo "Missing final-test external cache: $EXTERNAL_EMBEDDING_CACHE/test.pt" >&2; exit 3; fi
RUN=${TASK}_${MODEL_LABEL}_locked_test_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
RESULTS_PREFIX=${RESULTS_PREFIX:-locked_test_downstream}
OUT=${RESULTS_ROOT:-/path/to/results/${RESULTS_PREFIX}_${SLURM_ARRAY_JOB_ID}}/$RUN
ARGS=(--config "$MASTER_CONFIG" --downstream_config "$DOWNSTREAM_CONFIG" --downstream_weights "$DOWNSTREAM_CHECKPOINT" --freeze_encoder "$FREEZE_ENCODER" --precursor_conditioning "$PRECURSOR_CONDITIONING" --decoder_model "$DECODER_MODEL" --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --eval_only 1 --validate_on_end 0 --test_on_end 1 --limit_test_batches 1.0 --save_top_k 0 --save_last 0 --output_dir "$OUT" --log_dir "$OUT/logs" --wandb_project "$WANDB_PROJECT" --wandb_entity user --log_wandb 1)
if [[ -n "${ENCODER_CHECKPOINT:-}" ]]; then ARGS+=(--encoder_weights "$ENCODER_CHECKPOINT"); fi
if [[ -n "${EXTERNAL_EMBEDDING_CACHE:-}" ]]; then ARGS+=(--external_embedding_cache "$EXTERNAL_EMBEDDING_CACHE"); fi
export WANDB_NAME=$RUN WANDB_RUN_GROUP=paper_locked_test_selected
srun --kill-on-bad-exit=1 "$PY" -u -m src.main "${ARGS[@]}"
