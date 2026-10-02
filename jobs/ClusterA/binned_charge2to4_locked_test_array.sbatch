#!/usr/bin/env bash
# Uncapped binned spectral-angle reference on the GLEAMS-matched held-out split.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_CPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-2
#SBATCH --job-name=binned_c24_test
#SBATCH --output=/path/to/results/slurm_logs/binned_c24_test-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/binned_c24_test-%A_%a.err
set -eo pipefail
cd /path/to/dIon
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
if [[ "${ALLOW_LOCKED_TEST:-0}" != 1 ]]; then echo 'Set ALLOW_LOCKED_TEST=1.' >&2; exit 2; fi
CORPORA=(ninespecies_v2 bacterial kingdoms)
CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]:-}
if [[ -z "$CORPUS" ]]; then exit 2; fi
case "$CORPUS" in
  ninespecies_v2) PREFIX=ninespecies_v2_paper_test; DATA=/path/to/data/probing_datasets/ninespecies_v2_paper_test/charge2to4 ;;
  bacterial) PREFIX=bacterial_paper_test; DATA=/path/to/data/probing_datasets/bacterial_paper_benchmarks/test_peptide_disjoint/charge2to4 ;;
  kingdoms) PREFIX=kingdoms_paper_test; DATA=/path/to/data/probing_datasets/kingdoms_paper_benchmarks/test/charge2to4 ;;
esac
RESULTS=${RESULTS_ROOT:-/path/to/results/locked_test_binned_charge2to4_${SLURM_ARRAY_JOB_ID}}/representation/binned_spectrum/${CORPUS}_heldout_test_charge2to4
srun --kill-on-bad-exit=1 python -u scripts/run_binned_full_charge_baseline.py \
  --retrieval-config configs/probing_tasks/${PREFIX}_retrieval.yaml \
  --pair-config configs/probing_tasks/${PREFIX}_pairs.yaml \
  --dataset-root "$DATA" --results-dir "$RESULTS" \
  --accelerator cpu --batch-size 512 --max-peaks 0
