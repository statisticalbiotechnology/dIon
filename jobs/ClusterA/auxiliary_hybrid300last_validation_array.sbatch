#!/usr/bin/env bash
# Hybrid-300-last auxiliary validation: HYE chimericity and oxidized-Met,
# each with conditioned/null x frozen/full fine-tune, followed by full validation.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --job-name=aux_h300last
#SBATCH --output=slurm_outs/aux_h300last-%A_%a.out
#SBATCH --error=slurm_outs/aux_h300last-%A_%a.err
set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
PY=/path/to/conda-envs/dIon-env/bin/python
CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt
INDEX=$SLURM_ARRAY_TASK_ID
if (( INDEX < 4 )); then TASK=chimericity; MASTER=configs/master_chimericity_hye.yaml; DOWN=configs/downstream/chimericity_hye.yaml; PROJECT=ms2-chimericity; EPOCHS=100; VALIDATE=5; LOCAL=$INDEX; else TASK=oxidized_met; MASTER=configs/master_oxidized_met.yaml; DOWN=configs/downstream/oxidized_met.yaml; PROJECT=ms2-oxidized-met; EPOCHS=60; VALIDATE=3; LOCAL=$((INDEX - 4)); fi
case "$LOCAL" in
  0) LABEL=hybrid300last_conditioned_frozen; FREEZE=1; COND=conditioned ;;
  1) LABEL=hybrid300last_null_frozen; FREEZE=1; COND=null ;;
  2) LABEL=hybrid300last_conditioned_finetune; FREEZE=0; COND=conditioned ;;
  3) LABEL=hybrid300last_null_finetune; FREEZE=0; COND=null ;;
  *) exit 2 ;;
esac
RUN=${TASK}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
ROOT=/path/to/checkpoints/$TASK/$RUN
export WANDB_NAME=$RUN WANDB_RUN_GROUP=${TASK}_hybrid300last_validation
COMMON=(--config "$MASTER" --downstream_config "$DOWN" --encoder_weights "$CKPT" --freeze_encoder "$FREEZE" --precursor_conditioning "$COND" --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --max_steps 60000 --epochs "$EPOCHS" --early_stop 12 --limit_val_batches 100 --validate_every_n_epochs "$VALIDATE" --test_on_end 0 --validate_on_end 0 --output_dir "$ROOT" --log_dir /path/to/project/logs/$TASK/$RUN --wandb_project "$PROJECT" --wandb_entity user --log_wandb 1)
srun --kill-on-bad-exit=1 "$PY" -u -m src.main "${COMMON[@]}"
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
srun --kill-on-bad-exit=1 "$PY" -u -m src.main --config "$MASTER" --downstream_config "$DOWN" --encoder_weights "$CKPT" --downstream_weights "$BEST" --freeze_encoder "$FREEZE" --precursor_conditioning "$COND" --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --eval_only 1 --validate_on_end 1 --test_on_end 0 --save_top_k 0 --save_last 0 --output_dir "$ROOT/full_validation" --log_dir /path/to/project/logs/$TASK/$RUN/full_validation --wandb_project "$PROJECT" --wandb_entity user --log_wandb 1
