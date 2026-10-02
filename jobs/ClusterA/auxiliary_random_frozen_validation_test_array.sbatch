#!/usr/bin/env bash
# Train validation-selected frozen random-transformer heads and evaluate each
# selected checkpoint on its full held-out test split. The native main runner
# reloads ckpt_path="best" before testing.
#
# Submit:
#   sbatch --array=0-1 jobs/ClusterA/auxiliary_random_frozen_validation_test_array.sbatch
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=aux_random_frozen
#SBATCH --output=/path/to/results/slurm_logs/aux_random_frozen-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/aux_random_frozen-%A_%a.err

set -eo pipefail
cd /path/to/dIon
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
PY=/path/to/conda-envs/dIon-env/bin/python

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TASK=chimericity
    MASTER=configs/master_chimericity_hye_random.yaml
    DOWN=configs/downstream/chimericity_hye.yaml
    PROJECT=ms2-chimericity
    EPOCHS=100
    VALIDATE_EVERY=5
    ;;
  1)
    TASK=oxidized_met
    MASTER=configs/master_oxidized_met_random.yaml
    DOWN=configs/downstream/oxidized_met.yaml
    PROJECT=ms2-oxidized-met
    EPOCHS=60
    VALIDATE_EVERY=3
    ;;
  *) echo "Unknown array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

RUN=${TASK}_random_conditioned_frozen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
OUT=/path/to/results/auxiliary_random_frozen/${RUN}
export WANDB_NAME=${RUN}
export WANDB_RUN_GROUP=paper_auxiliary_random_frozen

srun --kill-on-bad-exit=1 "$PY" -u -m src.main \
  --config "$MASTER" \
  --downstream_config "$DOWN" \
  --freeze_encoder 1 \
  --precursor_conditioning conditioned \
  --use_mass 1 \
  --use_charge 1 \
  --accelerator gpu \
  --num_devices 1 \
  --num_nodes 1 \
  --strategy ddp \
  --precision bf16-mixed \
  --num_workers 8 \
  --pin_mem 1 \
  --max_steps 60000 \
  --epochs "$EPOCHS" \
  --early_stop 12 \
  --limit_val_batches 100 \
  --limit_test_batches 1.0 \
  --validate_every_n_epochs "$VALIDATE_EVERY" \
  --test_on_end 1 \
  --validate_on_end 0 \
  --save_top_k 1 \
  --save_last 1 \
  --output_dir "$OUT/checkpoints" \
  --log_dir "$OUT/logs" \
  --wandb_project "$PROJECT" \
  --wandb_entity user \
  --log_wandb 1
