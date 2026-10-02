#!/usr/bin/env bash
# Frozen released InstaNovo-FM mean-pool heads on validation only. Submit after the cache array completes.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --job-name=instanovo_fm_aux
#SBATCH --output=slurm_outs/instanovo_fm_aux-%A_%a.out
#SBATCH --error=slurm_outs/instanovo_fm_aux-%A_%a.err
set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
PY=/path/to/conda-envs/dIon-env/bin/python
CACHE_ROOT=${CACHE_ROOT:-/path/to/work/instanovo_fm_downstream_validation_cache}
case "$SLURM_ARRAY_TASK_ID" in
  0) TASK=sqa; LABEL=instanovo_fm_meanpool_frozen; MASTER=configs/master_sqa_instanovo_fm_frozen.yaml; DOWN=configs/downstream/sqa_casanovo_19pxd_frozen.yaml; PROJECT=ms2-sqa; EXTRA=() ;;
  1) TASK=chimericity; LABEL=instanovo_fm_meanpool_frozen; MASTER=configs/master_chimericity_hye_instanovo_fm_frozen.yaml; DOWN=configs/downstream/chimericity_hye.yaml; PROJECT=ms2-chimericity; EXTRA=(--max_steps 60000 --epochs 100 --early_stop 12 --limit_val_batches 100 --validate_every_n_epochs 5) ;;
  2) TASK=oxidized_met; LABEL=instanovo_fm_meanpool_frozen; MASTER=configs/master_oxidized_met_instanovo_fm_frozen.yaml; DOWN=configs/downstream/oxidized_met.yaml; PROJECT=ms2-oxidized-met; EXTRA=(--max_steps 60000 --epochs 60 --early_stop 12 --limit_val_batches 100 --validate_every_n_epochs 3) ;;
  3) TASK=retention_time; LABEL=instanovo_fm_meanpool_ordinal_frozen; MASTER=configs/master_retention_time_instanovo_fm_ordinal_frozen.yaml; DOWN=configs/downstream/retention_time_probe.yaml; PROJECT=ms2-retention-time; EXTRA=(--decoder_model linear_ordinal_head --max_steps 60000 --epochs 20 --early_stop 12 --limit_val_batches 100) ;;
  4) TASK=retention_time; LABEL=instanovo_fm_meanpool_regression_frozen; MASTER=configs/master_retention_time_instanovo_fm_regression_frozen.yaml; DOWN=configs/downstream/retention_time_regression_probe.yaml; PROJECT=ms2-retention-time; EXTRA=(--decoder_model linear_regression_head --max_steps 60000 --epochs 20 --early_stop 12 --limit_val_batches 100) ;;
  *) exit 2 ;;
esac
CACHE="$CACHE_ROOT/$TASK/cache"
if [[ ! -f "$CACHE/train.pt" || ! -f "$CACHE/val.pt" ]]; then echo "Missing train/val cache: $CACHE" >&2; exit 3; fi
RUN=${TASK}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
ROOT=/path/to/checkpoints/$TASK/$RUN
export WANDB_NAME=$RUN WANDB_RUN_GROUP=${TASK}_instanovo_fm_validation
srun --kill-on-bad-exit=1 "$PY" -u -m src.main --config "$MASTER" --downstream_config "$DOWN" --external_embedding_cache "$CACHE" --freeze_encoder 1 --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 "${EXTRA[@]}" --test_on_end 0 --validate_on_end 0 --output_dir "$ROOT" --log_dir /path/to/project/logs/$TASK/$RUN --wandb_project "$PROJECT" --wandb_entity user --log_wandb 1
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
BEST=$(find_selected_checkpoint) || { echo "No selected checkpoint under timestamped root $ROOT" >&2; exit 4; }
echo "Selected checkpoint: $BEST"
srun --kill-on-bad-exit=1 "$PY" -u -m src.main --config "$MASTER" --downstream_config "$DOWN" --external_embedding_cache "$CACHE" --downstream_weights "$BEST" --freeze_encoder 1 --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --eval_only 1 --validate_on_end 1 --test_on_end 0 --save_top_k 0 --save_last 0 --output_dir "$ROOT/full_validation" --log_dir /path/to/project/logs/$TASK/$RUN/full_validation --wandb_project "$PROJECT" --wandb_entity user --log_wandb 1
