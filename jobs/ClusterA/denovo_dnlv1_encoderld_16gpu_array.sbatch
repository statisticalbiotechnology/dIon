#!/usr/bin/env bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -C fat
#SBATCH -t 2-00:00:00
#SBATCH --nodes=2
#SBATCH --gpus=16
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --job-name=denovo_v5_ld
#SBATCH --output=slurm_outs/denovo_v5_ld-%A_%a.out
#SBATCH --error=slurm_outs/denovo_v5_ld-%A_%a.err

set -eo pipefail

cd /path/to/dIon
mkdir -p slurm_outs

export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
set -u

export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API:UserWarning"
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp

SOURCE_DATA_ROOT=/path/to/data/denovo_dnlv1/lance
HYBRID_100_CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_maxpeaks200_bs128_64gpu_cluster_b_epoch100/last.ckpt
HYBRID_300_CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt
LOCAL_STAGE_PARENT="${SLURM_TMPDIR:-/tmp}/dion_denovo_v5_${SLURM_JOB_ID}"
LOCAL_DATA_ROOT="$LOCAL_STAGE_PARENT/lance"
export SOURCE_DATA_ROOT LOCAL_STAGE_PARENT

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MASTER_CONFIG=configs/master_denovo_dnlv1_encoderld_scratch.yaml
    RUN_LABEL=scratch_conditioned
    ENCODER_ARGS=()
    ;;
  1)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_dnlv1.yaml
    RUN_LABEL=hybrid100_conditioned
    ENCODER_ARGS=(--encoder_weights "$HYBRID_100_CKPT")
    ;;
  2)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_dnlv1.yaml
    RUN_LABEL=hybrid300_conditioned
    ENCODER_ARGS=(--encoder_weights "$HYBRID_300_CKPT")
    ;;
  3)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_null_dnlv1.yaml
    RUN_LABEL=hybrid100_null
    ENCODER_ARGS=(--encoder_weights "$HYBRID_100_CKPT")
    ;;
  4)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_null_dnlv1.yaml
    RUN_LABEL=hybrid300_null
    ENCODER_ARGS=(--encoder_weights "$HYBRID_300_CKPT")
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN_NAME="denovo_dnlv1_encoderld_${RUN_LABEL}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT="/path/to/checkpoints/denovo/${RUN_NAME}"
LOG_ROOT="/path/to/project/logs/denovo/${RUN_NAME}"
export WANDB_NAME="$RUN_NAME"
export WANDB_RUN_GROUP=denovo_dnlv1_encoderld_ablation

for required in "$SOURCE_DATA_ROOT/train.lance" "$SOURCE_DATA_ROOT/val.lance" "$SOURCE_DATA_ROOT/test.lance"; do
  test -d "$required"
done
if [[ ${SLURM_ARRAY_TASK_ID} -eq 1 || ${SLURM_ARRAY_TASK_ID} -eq 3 ]]; then
  test -f "$HYBRID_100_CKPT"
elif [[ ${SLURM_ARRAY_TASK_ID} -ne 0 ]]; then
  test -f "$HYBRID_300_CKPT"
fi

# Stage only final, cleanup-audited V5 Lance splits on each node.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 --exact bash -lc '
  set -euo pipefail
  local_root="${LOCAL_STAGE_PARENT}/lance"
  source_kb=$(du -sk "${SOURCE_DATA_ROOT}" | awk "{print \$1}")
  free_kb=$(df -Pk "${SLURM_TMPDIR:-/tmp}" | awk "NR == 2 {print \$4}")
  required_kb=$((source_kb + 4 * 1024 * 1024))
  if [[ "${free_kb}" -lt "${required_kb}" ]]; then
    echo "Insufficient local scratch on $(hostname): ${free_kb} KiB available; ${required_kb} KiB required" >&2
    exit 1
  fi
  mkdir -p "${local_root}"
  rsync -a --delete "${SOURCE_DATA_ROOT}/" "${local_root}/"
  test -d "${local_root}/train.lance"
  test -d "${local_root}/val.lance"
  test -d "${local_root}/test.lance"
  echo "Staged on $(hostname): $(du -sh "${local_root}")"
'

echo "host: $(hostname)"
echo "start: $(date)"
echo "variant: ${RUN_LABEL}"
echo "master config: ${MASTER_CONFIG}"
echo "data root: ${LOCAL_DATA_ROOT}"
echo "primary validation: ${LOCAL_DATA_ROOT}/val.lance"
echo "batch size: 100/GPU; GPUs: 16; max steps: 149200"

srun --kill-on-bad-exit=1 python -u -m src.main \
  --config "$MASTER_CONFIG" \
  "${ENCODER_ARGS[@]}" \
  --downstream_root_dir "$LOCAL_DATA_ROOT" \
  --output_dir "$RUN_ROOT/checkpoint" \
  --log_dir "$LOG_ROOT" \
  --wandb_project denovo-dnlv1 \
  --wandb_entity user \
  --accelerator gpu \
  --num_devices 8 \
  --num_nodes 2 \
  --strategy ddp \
  --num_workers 4 \
  --pin_mem 1 \
  --batch_size 100 \
  --epochs 80 \
  --max_steps 149200 \
  --limit_train_batches 1.0 \
  --limit_val_batches 200 \
  --limit_test_batches 1.0 \
  --validate_every_n_epochs 1 \
  --save_top_k 5 \
  --save_last 1 \
  --test_on_end 1 \
  --log_wandb 1

echo "done: $(date)"
echo "outputs: ${RUN_ROOT}"
