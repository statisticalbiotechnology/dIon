#!/usr/bin/env bash
# Canonical 1% bacterial SupCon label-efficiency cohort.
# 0=scratch, 1=Hybrid-300-last full fine-tune, 2=Hybrid-300-last frozen.
# Model selection uses the fixed seeded 100-batch validation SupCon draw and
# metric_learning_val_loss_epoch (minimum). The restored best checkpoint then
# traverses every row of annotated_regenerated_v3/test.lance without a cap.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2
#SBATCH --job-name=metric_1pct
#SBATCH --output=slurm_outs/metric_1pct-%A_%a.out
#SBATCH --error=slurm_outs/metric_1pct-%A_%a.err

set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env

PY=/path/to/conda-envs/dIon-env/bin/python
MASTER=configs/master_metric_learning_bacterial_1pct.yaml
DOWNSTREAM=configs/downstream/metric_learning_bacterial_1pct.yaml
HYBRID_300=/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt

case "${SLURM_ARRAY_TASK_ID}" in
  0) LABEL=scratch; FREEZE=0; ENCODER_ARGS=() ;;
  1) LABEL=hybrid300last_conditioned; FREEZE=0; ENCODER_ARGS=(--encoder_weights "$HYBRID_300") ;;
  2) LABEL=hybrid300last_conditioned_frozen; FREEZE=1; ENCODER_ARGS=(--encoder_weights "$HYBRID_300") ;;
  *) echo "Unknown array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

RUN="metric_learning_bacterial_1pct_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export WANDB_NAME="$RUN"
export WANDB_RUN_GROUP=metric_learning_bacterial_1pct

# Do not override max_steps, LR, scheduler, validation sampler, or batch
# limits here. The strict 1% config copy preserves the 10% recipe: 40k steps,
# 100 fixed validation batches, and limit_test_batches: 1.0.
srun --kill-on-bad-exit=1 "$PY" -u -m src.main \
  --config "$MASTER" \
  --downstream_config "$DOWNSTREAM" \
  --freeze_encoder "$FREEZE" \
  --accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp \
  --precision bf16-mixed --num_workers 8 --pin_mem 1 \
  --output_dir "/path/to/checkpoints/metric_learning/bacterial_1pct/$RUN" \
  --log_dir "/path/to/project/logs/metric_learning/bacterial_1pct/$RUN" \
  --wandb_project bacteria-metric --wandb_entity user --log_wandb 1 \
  "${ENCODER_ARGS[@]}"
