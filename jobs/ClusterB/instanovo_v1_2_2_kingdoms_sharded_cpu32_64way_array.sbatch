#!/usr/bin/env bash
#SBATCH -A YOUR_CPU_ACCOUNT
#SBATCH -p cpu
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --array=0-63%64
#SBATCH --job-name=instv122_king64
#SBATCH --output=/path/to/runs/slurm_outs/instv122_king64-%A_%a.out
#SBATCH --error=/path/to/runs/slurm_outs/instv122_king64-%A_%a.err
set -euo pipefail

# Locked production mode, measured on ClusterB CPU: 32 cores, batch 32.
# The 64 deterministic subshards target same-day completion. Keep one process
# per subshard to avoid the predictor's all-rank prediction gather.
export DION_ROOT=/path/to/dIon
export INSTANOVO_V122_ENV=/path/to/work/conda_envs/instanovo_1_2_2_cpu
export CASANOVO_V521_PY="$INSTANOVO_V122_ENV/bin/python"
export LD_LIBRARY_PATH="$INSTANOVO_V122_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export INSTANOVO_V120_CKPT=/path/to/checkpoints/instanovo_v1_2_0/instanovo-v1.2.0.ckpt
export INSTANOVO_KINGDOMS_SHARDS=/path/to/data/denovo_kingdoms/run_disjoint_v1_species_cap100k/instanovo_v1_2_2_charge1to10/shards_64
export RESULTS_BASE=/path/to/runs/results/instanovo_v1_2_2/kingdoms_species_cap100k_full_charge_sharded_cpu32_batch32_shards64
export ALLOW_LOCKED_TEST=1
export INSTANOVO_FORCE_CPU=true
export INSTANOVO_FP16=false
export INSTANOVO_BATCH_SIZE=32
export INSTANOVO_NUM_WORKERS=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec bash /path/to/dIon/jobs/portable/instanovo_v122_kingdoms_shard.sh
