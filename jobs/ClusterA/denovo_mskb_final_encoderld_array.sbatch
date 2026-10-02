#!/usr/bin/env bash
# ClusterA wrapper for the standalone official MSKB-final de novo cohort.
# Required high-priority jobs are 0-4; optional frozen controls are 5-8.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-8
#SBATCH --job-name=denovo_mskb
#SBATCH --output=slurm_outs/denovo_mskb-%A_%a.out
#SBATCH --error=slurm_outs/denovo_mskb-%A_%a.err
set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
export PYTHON_BIN=/path/to/conda-envs/dIon-env/bin/python
export OUTPUT_BASE=/path/to/checkpoints/denovo_mskb_final
export LOG_BASE=/path/to/project/logs/denovo_mskb_final
export EXTRA_ARGS='--accelerator gpu --num_devices 1 --num_nodes 1 --strategy ddp --precision bf16-mixed --num_workers 8 --pin_mem 1 --test_on_end 0 --validate_on_end 0'
exec bash jobs/portable/denovo_mskb_final_encoderld_array.sh "$SLURM_ARRAY_TASK_ID"
