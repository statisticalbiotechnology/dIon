#!/usr/bin/env bash
# Frozen retention-time probe ablation: random, selected hybrid, and binned encoders,
# each evaluated with both soft-ordinal and scalar-regression targets.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9
#SBATCH --job-name=retention_time
#SBATCH --output=slurm_outs/retention_time-%A_%a.out
#SBATCH --error=slurm_outs/retention_time-%A_%a.err

set -eo pipefail

cd /path/to/dIon
mkdir -p slurm_outs
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

HYBRID_CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/epoch=219-dinov2_val_loss_epoch=5.71.ckpt
MASTER_HYBRID=configs/master_retention_time_probe.yaml
MASTER_RANDOM=configs/master_retention_time_random.yaml
MASTER_BINNED=configs/master_retention_time_binned.yaml
ORDINAL_CONFIG=configs/downstream/retention_time_probe.yaml
REGRESSION_CONFIG=configs/downstream/retention_time_regression_probe.yaml

RUN_LABEL=
MASTER_CONFIG=
DOWNSTREAM_CONFIG=
DECODER_MODEL=
PRECURSOR_CONDITIONING=conditioned
USE_MASS=1
USE_CHARGE=1
ENCODER_ARGS=()

case "${SLURM_ARRAY_TASK_ID}" in
  0) RUN_LABEL=random_ordinal_frozen; MASTER_CONFIG="${MASTER_RANDOM}"; DOWNSTREAM_CONFIG="${ORDINAL_CONFIG}"; DECODER_MODEL=linear_ordinal_head ;;
  1) RUN_LABEL=random_regression_frozen; MASTER_CONFIG="${MASTER_RANDOM}"; DOWNSTREAM_CONFIG="${REGRESSION_CONFIG}"; DECODER_MODEL=linear_regression_head ;;
  2) RUN_LABEL=hybrid300e219_conditioned_ordinal_frozen; MASTER_CONFIG="${MASTER_HYBRID}"; DOWNSTREAM_CONFIG="${ORDINAL_CONFIG}"; DECODER_MODEL=linear_ordinal_head; ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}") ;;
  3) RUN_LABEL=hybrid300e219_conditioned_regression_frozen; MASTER_CONFIG="${MASTER_HYBRID}"; DOWNSTREAM_CONFIG="${REGRESSION_CONFIG}"; DECODER_MODEL=linear_regression_head; ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}") ;;
  4) RUN_LABEL=hybrid300e219_null_ordinal_frozen; MASTER_CONFIG="${MASTER_HYBRID}"; DOWNSTREAM_CONFIG="${ORDINAL_CONFIG}"; DECODER_MODEL=linear_ordinal_head; PRECURSOR_CONDITIONING=null; ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}") ;;
  5) RUN_LABEL=hybrid300e219_null_regression_frozen; MASTER_CONFIG="${MASTER_HYBRID}"; DOWNSTREAM_CONFIG="${REGRESSION_CONFIG}"; DECODER_MODEL=linear_regression_head; PRECURSOR_CONDITIONING=null; ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}") ;;
  6) RUN_LABEL=binned1024_precursor_metadata_ordinal_frozen; MASTER_CONFIG="${MASTER_BINNED}"; DOWNSTREAM_CONFIG="${ORDINAL_CONFIG}"; DECODER_MODEL=linear_ordinal_head ;;
  7) RUN_LABEL=binned1024_precursor_metadata_regression_frozen; MASTER_CONFIG="${MASTER_BINNED}"; DOWNSTREAM_CONFIG="${REGRESSION_CONFIG}"; DECODER_MODEL=linear_regression_head ;;
  8) RUN_LABEL=binned1024_spectrum_only_ordinal_frozen; MASTER_CONFIG="${MASTER_BINNED}"; DOWNSTREAM_CONFIG="${ORDINAL_CONFIG}"; DECODER_MODEL=linear_ordinal_head; USE_MASS=0; USE_CHARGE=0 ;;
  9) RUN_LABEL=binned1024_spectrum_only_regression_frozen; MASTER_CONFIG="${MASTER_BINNED}"; DOWNSTREAM_CONFIG="${REGRESSION_CONFIG}"; DECODER_MODEL=linear_regression_head; USE_MASS=0; USE_CHARGE=0 ;;
  *) echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

RUN_NAME="retention_time_${RUN_LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT="/path/to/checkpoints/retention_time/${RUN_NAME}"
LOG_ROOT="/path/to/project/logs/retention_time/${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP=retention_time_auxiliary_ablation_v1

# 20 full train epochs are 62.5k batches; cap all variants at 60k optimizer steps.
# Validation is a fixed 100-batch prefix every ~3.1k steps; final test is complete.
srun --kill-on-bad-exit=1 python -u -m src.main \
  --config "${MASTER_CONFIG}" \
  --downstream_config "${DOWNSTREAM_CONFIG}" \
  "${ENCODER_ARGS[@]}" \
  --decoder_model "${DECODER_MODEL}" \
  --freeze_encoder 1 \
  --precursor_conditioning "${PRECURSOR_CONDITIONING}" \
  --use_mass "${USE_MASS}" \
  --use_charge "${USE_CHARGE}" \
  --accelerator gpu \
  --num_devices 1 \
  --num_nodes 1 \
  --strategy ddp \
  --precision bf16-mixed \
  --num_workers 8 \
  --pin_mem 1 \
  --max_steps 60000 \
  --epochs 20 \
  --early_stop 12 \
  --limit_val_batches 100 \
  --validate_every_n_epochs 1 \
  --test_on_end 1 \
  --validate_on_end 0 \
  --save_top_k 1 \
  --save_last 1 \
  --output_dir "${RUN_ROOT}" \
  --log_dir "${LOG_ROOT}" \
  --wandb_project ms2-retention-time \
  --wandb_entity user \
  --log_wandb 1
