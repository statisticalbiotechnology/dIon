#!/usr/bin/env bash
# Recover complete validation for Hybrid-300-last runs whose training completed
# before the validation-array checkpoint lookup was corrected. Submit with
# SOURCE_JOB_ID set to the original training array ID.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --job-name=reval_h300last
#SBATCH --output=slurm_outs/reval_h300last-%A_%a.out
#SBATCH --error=slurm_outs/reval_h300last-%A_%a.err
set -eo pipefail

if [[ -z "${SOURCE_JOB_ID:-}" ]]; then
  echo "SOURCE_JOB_ID must name the original training array." >&2
  exit 2
fi
if [[ -z "${TASK_FAMILY:-}" ]]; then
  echo "TASK_FAMILY must be one of: sqa, chimericity, oxidized_met, retention_time." >&2
  exit 2
fi

cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
PY=/path/to/conda-envs/dIon-env/bin/python
CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt
INDEX=$SLURM_ARRAY_TASK_ID

case "$TASK_FAMILY" in
  sqa)
    case "$INDEX" in
      0) LABEL=hybrid300last_conditioned_frozen; MASTER=configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10.yaml; DOWN=configs/downstream/sqa_casanovo_19pxd_frozen.yaml; FREEZE=1; COND=conditioned ;;
      1) LABEL=hybrid300last_null_frozen; MASTER=configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10_null.yaml; DOWN=configs/downstream/sqa_casanovo_19pxd_frozen.yaml; FREEZE=1; COND=null ;;
      2) LABEL=hybrid300last_conditioned_finetune; MASTER=configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10.yaml; DOWN=configs/downstream/sqa_casanovo_19pxd_finetune.yaml; FREEZE=0; COND=conditioned ;;
      3) LABEL=hybrid300last_null_finetune; MASTER=configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10_null.yaml; DOWN=configs/downstream/sqa_casanovo_19pxd_finetune.yaml; FREEZE=0; COND=null ;;
      *) exit 2 ;;
    esac
    RUN=sqa_${LABEL}_${SOURCE_JOB_ID}_${INDEX}
    PROJECT=ms2-sqa
    ;;
  chimericity|oxidized_met)
    if [[ "$TASK_FAMILY" == chimericity ]]; then
      MASTER=configs/master_chimericity_hye.yaml; DOWN=configs/downstream/chimericity_hye.yaml; PROJECT=ms2-chimericity
    else
      MASTER=configs/master_oxidized_met.yaml; DOWN=configs/downstream/oxidized_met.yaml; PROJECT=ms2-oxidized-met
    fi
    case "$INDEX" in
      0) LABEL=hybrid300last_conditioned_frozen; FREEZE=1; COND=conditioned ;;
      1) LABEL=hybrid300last_null_frozen; FREEZE=1; COND=null ;;
      2) LABEL=hybrid300last_conditioned_finetune; FREEZE=0; COND=conditioned ;;
      3) LABEL=hybrid300last_null_finetune; FREEZE=0; COND=null ;;
      *) exit 2 ;;
    esac
    if [[ "$TASK_FAMILY" == oxidized_met ]]; then INDEX=$((INDEX + 4)); fi
    RUN=${TASK_FAMILY}_${LABEL}_${SOURCE_JOB_ID}_${INDEX}
    ;;
  retention_time)
    case "$INDEX" in
      0) LABEL=hybrid300last_conditioned_ordinal_frozen; MASTER=configs/master_retention_time_probe.yaml; DOWN=configs/downstream/retention_time_probe.yaml; DECODER=linear_ordinal_head; COND=conditioned ;;
      1) LABEL=hybrid300last_conditioned_regression_frozen; MASTER=configs/master_retention_time_regression_probe.yaml; DOWN=configs/downstream/retention_time_regression_probe.yaml; DECODER=linear_regression_head; COND=conditioned ;;
      2) LABEL=hybrid300last_null_ordinal_frozen; MASTER=configs/master_retention_time_probe.yaml; DOWN=configs/downstream/retention_time_probe.yaml; DECODER=linear_ordinal_head; COND=null ;;
      3) LABEL=hybrid300last_null_regression_frozen; MASTER=configs/master_retention_time_regression_probe.yaml; DOWN=configs/downstream/retention_time_regression_probe.yaml; DECODER=linear_regression_head; COND=null ;;
      *) exit 2 ;;
    esac
    FREEZE=1
    RUN=retention_time_${LABEL}_${SOURCE_JOB_ID}_${INDEX}
    PROJECT=ms2-retention-time
    ;;
  *) echo "Unknown TASK_FAMILY: $TASK_FAMILY" >&2; exit 2 ;;
esac

ROOT=/path/to/checkpoints/$TASK_FAMILY/$RUN
find_selected_checkpoint() {
  local attempt candidate
  for attempt in {1..12}; do
    candidate=$(find "$(dirname "$ROOT")" -maxdepth 2 -type f -name 'epoch=*.ckpt' \
      -path "${ROOT}*/*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    sleep 5
  done
  return 1
}
BEST=$(find_selected_checkpoint) || { echo "No selected checkpoint under timestamped root $ROOT" >&2; exit 3; }
echo "Selected checkpoint: $BEST"

EXTRA=()
if [[ "$TASK_FAMILY" == retention_time ]]; then EXTRA=(--decoder_model "$DECODER"); fi
export WANDB_NAME=${RUN}_full_validation WANDB_RUN_GROUP=${TASK_FAMILY}_hybrid300last_validation
srun --kill-on-bad-exit=1 "$PY" -u -m src.main --config "$MASTER" --downstream_config "$DOWN" --encoder_weights "$CKPT" --downstream_weights "$BEST" "${EXTRA[@]}" --freeze_encoder "$FREEZE" --precursor_conditioning "$COND" --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --eval_only 1 --validate_on_end 1 --test_on_end 0 --save_top_k 0 --save_last 0 --output_dir "$ROOT/full_validation" --log_dir /path/to/project/logs/$TASK_FAMILY/$RUN/full_validation --wandb_project "$PROJECT" --wandb_entity user --log_wandb 1
