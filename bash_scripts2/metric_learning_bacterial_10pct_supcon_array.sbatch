#!/usr/bin/env bash
# First supervised metric-learning calibration at the 10% peptide-label budget.
# Default array: scratch plus fine-tuned and frozen variants of the selected
# DINO epoch-219 and canonical epoch-100 encoders. Every variant uses identical
# data, sampler, metric head, and training schedule.
#
# Submit all five variants:
#   sbatch bash_scripts2/metric_learning_bacterial_10pct_supcon_array.sbatch
# Submit only the two frozen variants:
#   sbatch --array=0-1 --export=ALL,RUN_VARIANTS=frozen_only \
#     bash_scripts2/metric_learning_bacterial_10pct_supcon_array.sbatch

#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --job-name=metric10_supcon
#SBATCH --output=slurm_outs/metric10_supcon-%A_%a.out
#SBATCH --error=slurm_outs/metric10_supcon-%A_%a.err

set -euo pipefail

cd /path/to/dIon
mkdir -p slurm_outs

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

MASTER_CONFIG=configs/master_metric_learning_bacterial.yaml
PROBING_CONFIG=configs/probing_tasks/metric_learning_paper_validation.yaml
TRAIN_LANCE=/path/to/data/bacteria_PXD010000__PXD010613/annotated_regenerated_v3/metric_learning/bacterial_metric_train_peptides_10pct_seed20260908.lance
HYBRID_EPOCH219_CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/epoch=219-dinov2_val_loss_epoch=5.71.ckpt
HYBRID_EPOCH100_CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_maxpeaks200_bs128_64gpu_cluster_b_epoch100/last.ckpt

RUN_VARIANTS="${RUN_VARIANTS:-all}"
ENCODER_PATH=""
FREEZE_ENCODER=0

if [[ "${RUN_VARIANTS}" == "frozen_only" ]]; then
  case "${SLURM_ARRAY_TASK_ID}" in
    0)
      RUN_LABEL=hybrid300e219_frozen
      ENCODER_PATH="${HYBRID_EPOCH219_CKPT}"
      FREEZE_ENCODER=1
      ;;
    1)
      RUN_LABEL=hybrid100e_frozen
      ENCODER_PATH="${HYBRID_EPOCH100_CKPT}"
      FREEZE_ENCODER=1
      ;;
    *)
      echo "frozen_only requires --array=0-1; got task ${SLURM_ARRAY_TASK_ID}" >&2
      exit 2
      ;;
  esac
elif [[ "${RUN_VARIANTS}" == "all" ]]; then
  case "${SLURM_ARRAY_TASK_ID}" in
    0)
      RUN_LABEL=scratch
      ;;
    1)
      RUN_LABEL=hybrid300e219
      ENCODER_PATH="${HYBRID_EPOCH219_CKPT}"
      ;;
    2)
      RUN_LABEL=hybrid100e
      ENCODER_PATH="${HYBRID_EPOCH100_CKPT}"
      ;;
    3)
      RUN_LABEL=hybrid300e219_frozen
      ENCODER_PATH="${HYBRID_EPOCH219_CKPT}"
      FREEZE_ENCODER=1
      ;;
    4)
      RUN_LABEL=hybrid100e_frozen
      ENCODER_PATH="${HYBRID_EPOCH100_CKPT}"
      FREEZE_ENCODER=1
      ;;
    *)
      echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
      exit 2
      ;;
  esac
else
  echo "RUN_VARIANTS must be 'all' or 'frozen_only'; got ${RUN_VARIANTS}" >&2
  exit 2
fi

ENCODER_ARGS=(--freeze_encoder "${FREEZE_ENCODER}")
if [[ -n "${ENCODER_PATH}" ]]; then
  test -f "${ENCODER_PATH}"
  ENCODER_ARGS+=(--encoder_weights "${ENCODER_PATH}")
fi

test -d "${TRAIN_LANCE}"

RUN_NAME="metric_learning_bacterial_10pct_${RUN_LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT="/path/to/checkpoints/metric_learning/bacterial_10pct/${RUN_NAME}"
LOG_ROOT="/path/to/project/logs/metric_learning/bacterial_10pct/${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP=metric_learning_bacterial_10pct_supcon_v4_full_global_suite

MAX_STEPS=40000
LOGICAL_EPOCHS=80
EARLY_STOP_PATIENCE=20

echo "start=$(date)"
echo "run=${RUN_NAME}"
echo "run_variants=${RUN_VARIANTS}"
echo "freeze_encoder=${FREEZE_ENCODER}"
echo "train_lance=${TRAIN_LANCE}"
echo "max_steps=${MAX_STEPS}; logical_epochs=${LOGICAL_EPOCHS}; validation_round_steps=500; early_stop_patience=${EARLY_STOP_PATIENCE}"

srun --kill-on-bad-exit=1 python -u -m src.main \
  --config "${MASTER_CONFIG}" \
  --probing_config "${PROBING_CONFIG}" \
  --downstream_train_path "${TRAIN_LANCE}" \
  "${ENCODER_ARGS[@]}" \
  --accelerator gpu \
  --num_devices 1 \
  --num_nodes 1 \
  --strategy ddp \
  --precision bf16-mixed \
  --num_workers 8 \
  --pin_mem 1 \
  --max_steps "${MAX_STEPS}" \
  --epochs "${LOGICAL_EPOCHS}" \
  --early_stop "${EARLY_STOP_PATIENCE}" \
  --save_top_k 1 \
  --save_last 1 \
  --validate_every_n_epochs 1 \
  --test_on_end 0 \
  --output_dir "${RUN_ROOT}" \
  --log_dir "${LOG_ROOT}" \
  --wandb_project bacteria-metric \
  --wandb_entity user \
  --log_wandb 1

echo "done=$(date)"
echo "outputs=${RUN_ROOT}"
