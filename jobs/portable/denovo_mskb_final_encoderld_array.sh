#!/usr/bin/env bash
# Scheduler-neutral nine-variant launcher for standalone official MSKB-final.
# Activate dIon-env before invoking. Submit/launch one index per process:
#   bash jobs/portable/denovo_mskb_final_encoderld_array.sh 0
# or set SLURM_ARRAY_TASK_ID=0..8 in a site-specific array job.
#
# Do not scale LR, warmup, batch size, or max steps by GPU count. Override only
# launcher topology/path arguments through EXTRA_ARGS when the target cluster
# needs them (for example: "--num_devices 4 --num_nodes 1 --strategy ddp").

set -eo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

INDEX=${SLURM_ARRAY_TASK_ID:-${1:-}}
if [[ -z "$INDEX" ]]; then
  echo "Usage: $0 <0..8>, or set SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_BASE=${OUTPUT_BASE:-/path/to/checkpoints/denovo}
LOG_BASE=${LOG_BASE:-/path/to/project/logs/denovo}
RUN_SUFFIX=${RUN_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}
HYBRID_100_CKPT=${HYBRID_100_CKPT:-/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_maxpeaks200_bs128_64gpu_cluster_b_epoch100/last.ckpt}
HYBRID_300_CKPT=${HYBRID_300_CKPT:-/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt}

ENCODER_ARGS=()
FREEZE_ARGS=(--freeze_encoder 0)
case "$INDEX" in
  0)
    MASTER_CONFIG=configs/master_denovo_mskb_final_encoderld_scratch.yaml
    VARIANT=scratch_conditioned
    ;;
  1)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_mskb_final.yaml
    VARIANT=hybrid100_conditioned
    ENCODER_ARGS=(--encoder_weights "$HYBRID_100_CKPT")
    ;;
  2)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_mskb_final.yaml
    VARIANT=hybrid300_conditioned
    ENCODER_ARGS=(--encoder_weights "$HYBRID_300_CKPT")
    ;;
  3)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_null_mskb_final.yaml
    VARIANT=hybrid100_null
    ENCODER_ARGS=(--encoder_weights "$HYBRID_100_CKPT")
    ;;
  4)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_null_mskb_final.yaml
    VARIANT=hybrid300_null
    ENCODER_ARGS=(--encoder_weights "$HYBRID_300_CKPT")
    ;;
  5)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_mskb_final.yaml
    VARIANT=hybrid100_conditioned_frozen
    ENCODER_ARGS=(--encoder_weights "$HYBRID_100_CKPT")
    FREEZE_ARGS=(--freeze_encoder 1)
    ;;
  6)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_null_mskb_final.yaml
    VARIANT=hybrid100_null_frozen
    ENCODER_ARGS=(--encoder_weights "$HYBRID_100_CKPT")
    FREEZE_ARGS=(--freeze_encoder 1)
    ;;
  7)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_mskb_final.yaml
    VARIANT=hybrid300_conditioned_frozen
    ENCODER_ARGS=(--encoder_weights "$HYBRID_300_CKPT")
    FREEZE_ARGS=(--freeze_encoder 1)
    ;;
  8)
    MASTER_CONFIG=configs/master_denovo_dion_hybrid_null_mskb_final.yaml
    VARIANT=hybrid300_null_frozen
    ENCODER_ARGS=(--encoder_weights "$HYBRID_300_CKPT")
    FREEZE_ARGS=(--freeze_encoder 1)
    ;;
  *)
    echo "Unknown array index: $INDEX (expected 0..8)." >&2
    exit 2
    ;;
esac

for path in \
  /path/to/data/denovo_mskb_final/lance_peptidoform_val10k_seed42/train.lance \
  /path/to/data/denovo_mskb_final/lance_peptidoform_val10k_seed42/val.lance \
  /path/to/data/denovo_mskb_final/lance_peptidoform_val10k_seed42/test.lance; do
  test -d "$path"
done
if [[ ${#ENCODER_ARGS[@]} -gt 0 ]]; then
  test -f "${ENCODER_ARGS[1]}"
fi

RUN_NAME="denovo_mskb_final_encoderld_${VARIANT}_${RUN_SUFFIX}_${INDEX}"
export WANDB_NAME="$RUN_NAME"
export WANDB_RUN_GROUP=denovo_mskb_final_encoderld_ablation
mkdir -p "$OUTPUT_BASE/$RUN_NAME" "$LOG_BASE/$RUN_NAME"

echo "variant: $VARIANT"
echo "master config: $MASTER_CONFIG"
echo "run name: $RUN_NAME"
echo "extra args: ${EXTRA_ARGS:-<none>}"

# shellcheck disable=SC2086
"$PYTHON_BIN" -u -m src.main \
  --config "$MASTER_CONFIG" \
  "${ENCODER_ARGS[@]}" \
  "${FREEZE_ARGS[@]}" \
  --output_dir "$OUTPUT_BASE/$RUN_NAME" \
  --log_dir "$LOG_BASE/$RUN_NAME" \
  --batch_size 100 \
  --max_steps 149200 \
  --limit_val_batches 100 \
  --log_wandb 1 \
  ${EXTRA_ARGS:-}
