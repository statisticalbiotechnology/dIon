#!/usr/bin/env bash
# Source Git revision: ca3161e7414f504618c4a17f74feb7cbeaddd975
# Archived Slurm array: 17425290 (tasks 0-7).
# Historical caveat: the checkpoint monitor keys were at the wrong YAML level.
# Only the three rows with identical validation-loss and validation-AUROC optima
# were retained in the final result record.
#SBATCH --array=0-7
#SBATCH --output=slurm_outs/sqa_hybrid_ablation-%A_%a.out
#SBATCH --error=slurm_outs/sqa_hybrid_ablation-%A_%a.err

# Historical job specification preserved with the canonical paper result.
# This file records the job topology and command that produced the archived
# result. To reproduce byte-for-byte, use the Git revision named below and the
# configuration snapshots under results/sqa/casanovo_19pxd_auc_selected__20260908T160857Z/configs/.
#
# Do not treat this as a current production submission recipe.

#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -C fat
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sqa_hybrid_ablation

set -euo pipefail

cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
conda activate dIon-env

RUN_NAMES=(
  hybrid_conditioned_frozen
  hybrid_null_frozen
  hybrid_conditioned_finetune
  hybrid_null_finetune
  random_conditioned_frozen
  random_conditioned_finetune
  binned1024_conditioned_frozen
  binned1024_spectrum_only_frozen
)
MASTER_CONFIGS=(
  configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10.yaml
  configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10_null.yaml
  configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10.yaml
  configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10_null.yaml
  configs/master_sqa_random_casanovo_19pxd_balanced_charge1_10.yaml
  configs/master_sqa_random_casanovo_19pxd_balanced_charge1_10.yaml
  configs/master_sqa_binned_casanovo_19pxd_balanced_charge1_10.yaml
  configs/master_sqa_binned_casanovo_19pxd_balanced_charge1_10.yaml
)
DOWNSTREAM_CONFIGS=(
  configs/downstream/sqa_casanovo_19pxd_frozen.yaml
  configs/downstream/sqa_casanovo_19pxd_frozen.yaml
  configs/downstream/sqa_casanovo_19pxd_finetune.yaml
  configs/downstream/sqa_casanovo_19pxd_finetune.yaml
  configs/downstream/sqa_casanovo_19pxd_frozen.yaml
  configs/downstream/sqa_casanovo_19pxd_finetune.yaml
  configs/downstream/sqa_casanovo_19pxd_frozen.yaml
  configs/downstream/sqa_casanovo_19pxd_frozen.yaml
)
FREEZE_ENCODERS=(1 1 0 0 1 0 1 1)
EARLY_STOP_PATIENCE=(2 2 3 3 2 3 2 2)
USE_MASS=(1 1 1 1 1 1 1 0)
USE_CHARGE=(1 1 1 1 1 1 1 0)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
export WANDB_NAME="$RUN_NAME"

export WANDB_RUN_GROUP="sqa_19pxd_auc_selected_ablation"

srun --kill-on-bad-exit=1 python -u -m src.main \
  --config "${MASTER_CONFIGS[$TASK_ID]}" \
  --downstream_config "${DOWNSTREAM_CONFIGS[$TASK_ID]}" \
  --freeze_encoder "${FREEZE_ENCODERS[$TASK_ID]}" \
  --use_mass "${USE_MASS[$TASK_ID]}" \
  --use_charge "${USE_CHARGE[$TASK_ID]}" \
  --early_stop "${EARLY_STOP_PATIENCE[$TASK_ID]}" \
  --test_on_end 1 \
  --validate_on_end 0 \
  --output_dir "/path/to/checkpoints/sqa/${RUN_NAME}" \
  --log_dir "/path/to/project/logs/sqa/${RUN_NAME}" \
  --wandb_project ms2-sqa \
  --wandb_entity user \
  --log_wandb 1
