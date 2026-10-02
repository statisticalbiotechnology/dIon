#!/usr/bin/env bash
# Hybrid-300-last frozen RT probes: conditioned/null x ordinal/scalar regression.
# Each run selects on the fixed 100-batch monitor, then reports complete validation.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --job-name=rt_h300last
#SBATCH --output=slurm_outs/rt_h300last-%A_%a.out
#SBATCH --error=slurm_outs/rt_h300last-%A_%a.err
set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
PY=/path/to/conda-envs/dIon-env/bin/python
CKPT=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt
case "$SLURM_ARRAY_TASK_ID" in
  0) LABEL=hybrid300last_conditioned_ordinal_frozen; MASTER=configs/master_retention_time_probe.yaml; DOWN=configs/downstream/retention_time_probe.yaml; DECODER=linear_ordinal_head; COND=conditioned ;;
  1) LABEL=hybrid300last_conditioned_regression_frozen; MASTER=configs/master_retention_time_regression_probe.yaml; DOWN=configs/downstream/retention_time_regression_probe.yaml; DECODER=linear_regression_head; COND=conditioned ;;
  2) LABEL=hybrid300last_null_ordinal_frozen; MASTER=configs/master_retention_time_probe.yaml; DOWN=configs/downstream/retention_time_probe.yaml; DECODER=linear_ordinal_head; COND=null ;;
  3) LABEL=hybrid300last_null_regression_frozen; MASTER=configs/master_retention_time_regression_probe.yaml; DOWN=configs/downstream/retention_time_regression_probe.yaml; DECODER=linear_regression_head; COND=null ;;
  *) exit 2 ;;
esac
RUN=retention_time_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
ROOT=/path/to/checkpoints/retention_time/$RUN
export WANDB_NAME=$RUN WANDB_RUN_GROUP=retention_time_hybrid300last_validation
srun --kill-on-bad-exit=1 "$PY" -u -m src.main --config "$MASTER" --downstream_config "$DOWN" --encoder_weights "$CKPT" --decoder_model "$DECODER" --freeze_encoder 1 --precursor_conditioning "$COND" --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --max_steps 60000 --epochs 20 --early_stop 12 --limit_val_batches 100 --validate_every_n_epochs 1 --test_on_end 0 --validate_on_end 0 --output_dir "$ROOT" --log_dir /path/to/project/logs/retention_time/$RUN --wandb_project ms2-retention-time --wandb_entity user --log_wandb 1
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
srun --kill-on-bad-exit=1 "$PY" -u -m src.main --config "$MASTER" --downstream_config "$DOWN" --encoder_weights "$CKPT" --downstream_weights "$BEST" --decoder_model "$DECODER" --freeze_encoder 1 --precursor_conditioning "$COND" --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --eval_only 1 --validate_on_end 1 --test_on_end 0 --save_top_k 0 --save_last 0 --output_dir "$ROOT/full_validation" --log_dir /path/to/project/logs/retention_time/$RUN/full_validation --wandb_project ms2-retention-time --wandb_entity user --log_wandb 1
