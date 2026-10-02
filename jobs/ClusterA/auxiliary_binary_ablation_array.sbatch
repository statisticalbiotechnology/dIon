#!/usr/bin/env bash
# Binary auxiliary-task ablation: scratch, selected hybrid, and two binned baselines.
# Submit chimericity:
#   sbatch --export=ALL,AUXILIARY_TASK=chimericity jobs/ClusterA/auxiliary_binary_ablation_array.sbatch
# Submit oxidized-Met:
#   sbatch --export=ALL,AUXILIARY_TASK=oxidized_met jobs/ClusterA/auxiliary_binary_ablation_array.sbatch
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-6
#SBATCH --job-name=aux_binary
#SBATCH --output=slurm_outs/aux_binary-%A_%a.out
#SBATCH --error=slurm_outs/aux_binary-%A_%a.err

set -eo pipefail

cd /path/to/dIon
mkdir -p slurm_outs
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

TASK="${AUXILIARY_TASK:?Set AUXILIARY_TASK to chimericity or oxidized_met.}"
HYBRID_CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/epoch=219-dinov2_val_loss_epoch=5.71.ckpt

case "${TASK}" in
  chimericity)
    MASTER_HYBRID=configs/master_chimericity_hye.yaml
    MASTER_RANDOM=configs/master_chimericity_hye_random.yaml
    MASTER_BINNED=configs/master_chimericity_hye_binned.yaml
    DOWNSTREAM_CONFIG=configs/downstream/chimericity_hye.yaml
    WANDB_PROJECT=ms2-chimericity
    EPOCHS=100
    VALIDATE_EVERY=5
    ;;
  oxidized_met)
    MASTER_HYBRID=configs/master_oxidized_met.yaml
    MASTER_RANDOM=configs/master_oxidized_met_random.yaml
    MASTER_BINNED=configs/master_oxidized_met_binned.yaml
    DOWNSTREAM_CONFIG=configs/downstream/oxidized_met.yaml
    WANDB_PROJECT=ms2-oxidized-met
    EPOCHS=60
    VALIDATE_EVERY=3
    ;;
  *)
    echo "Unknown AUXILIARY_TASK: ${TASK}" >&2
    exit 2
    ;;
esac

RUN_LABEL=
MASTER_CONFIG=
FREEZE_ENCODER=0
USE_MASS=1
USE_CHARGE=1
PRECURSOR_CONDITIONING=conditioned
ENCODER_ARGS=()

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_LABEL=scratch_conditioned_finetune
    MASTER_CONFIG="${MASTER_RANDOM}"
    ;;
  1)
    RUN_LABEL=hybrid300e219_conditioned_frozen
    MASTER_CONFIG="${MASTER_HYBRID}"
    FREEZE_ENCODER=1
    ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}")
    ;;
  2)
    RUN_LABEL=hybrid300e219_null_frozen
    MASTER_CONFIG="${MASTER_HYBRID}"
    FREEZE_ENCODER=1
    PRECURSOR_CONDITIONING=null
    ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}")
    ;;
  3)
    RUN_LABEL=hybrid300e219_conditioned_finetune
    MASTER_CONFIG="${MASTER_HYBRID}"
    ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}")
    ;;
  4)
    RUN_LABEL=hybrid300e219_null_finetune
    MASTER_CONFIG="${MASTER_HYBRID}"
    PRECURSOR_CONDITIONING=null
    ENCODER_ARGS=(--encoder_weights "${HYBRID_CKPT}")
    ;;
  5)
    RUN_LABEL=binned1024_precursor_metadata_frozen
    MASTER_CONFIG="${MASTER_BINNED}"
    FREEZE_ENCODER=1
    ;;
  6)
    RUN_LABEL=binned1024_spectrum_only_frozen
    MASTER_CONFIG="${MASTER_BINNED}"
    FREEZE_ENCODER=1
    USE_MASS=0
    USE_CHARGE=0
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN_NAME="${TASK}_${RUN_LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT="/path/to/checkpoints/${TASK}/${RUN_NAME}"
LOG_ROOT="/path/to/project/logs/${TASK}/${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP="${TASK}_auxiliary_ablation_v1"

# A shared 60k-step horizon. Validation rounds are roughly 3.1k steps apart;
# each uses a fixed 100-batch validation prefix, while test-on-end is complete.
MAX_STEPS=60000
EARLY_STOP_PATIENCE=12

echo "task=${TASK} run=${RUN_NAME} max_steps=${MAX_STEPS} epochs=${EPOCHS} validation_every=${VALIDATE_EVERY}"

srun --kill-on-bad-exit=1 python -u -m src.main \
  --config "${MASTER_CONFIG}" \
  --downstream_config "${DOWNSTREAM_CONFIG}" \
  "${ENCODER_ARGS[@]}" \
  --freeze_encoder "${FREEZE_ENCODER}" \
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
  --max_steps "${MAX_STEPS}" \
  --epochs "${EPOCHS}" \
  --early_stop "${EARLY_STOP_PATIENCE}" \
  --limit_val_batches 100 \
  --validate_every_n_epochs "${VALIDATE_EVERY}" \
  --test_on_end 1 \
  --validate_on_end 0 \
  --save_top_k 1 \
  --save_last 1 \
  --output_dir "${RUN_ROOT}" \
  --log_dir "${LOG_ROOT}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity user \
  --log_wandb 1
