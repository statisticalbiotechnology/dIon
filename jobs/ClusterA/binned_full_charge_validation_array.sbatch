#!/usr/bin/env bash
# Uncapped peak-only binned/spectral-angle baseline for full-charge validation.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2
#SBATCH --job-name=binned_fullval
#SBATCH --output=slurm_outs/binned_fullval-%A_%a.out
#SBATCH --error=slurm_outs/binned_fullval-%A_%a.err

set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
CORPORA=(ninespecies_v2 bacterial kingdoms)
CORPUS="${CORPORA[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "$CORPUS" ]]; then echo "Invalid array index" >&2; exit 2; fi
case "$CORPUS" in
  ninespecies_v2) PREFIX=ninespecies_v2_paper_validation ;;
  bacterial) PREFIX=bacterial_paper_validation ;;
  kingdoms) PREFIX=kingdoms_paper_validation ;;
esac
RESULTS="${RESULTS_ROOT:-/path/to/results/full_charge_validation_${SLURM_ARRAY_JOB_ID}}/representation/binned_spectrum/${CORPUS}"
srun --kill-on-bad-exit=1 python -u scripts/run_binned_full_charge_baseline.py \
  --retrieval-config "configs/probing_tasks/${PREFIX}_retrieval.yaml" \
  --pair-config "configs/probing_tasks/${PREFIX}_pairs.yaml" \
  --results-dir "$RESULTS" --accelerator gpu --batch-size 128 --max-peaks 0
