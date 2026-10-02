#!/usr/bin/env bash
# Production 128-GPU local-60 Gram refinement with a 50,715-step linear 0-to-2 ramp.
# Submit with: sbatch jobs/ClusterB/dion_hybrid_local60_gram_refinement_linear_ramp_128gpu.sbatch

#SBATCH --job-name=dino-gram-lin
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/runs/slurm/%x-%j.out
#SBATCH --error=/path/to/runs/slurm/%x-%j.err

set -euo pipefail

readonly PERSONAL=/path/to/work
readonly REPO="${PERSONAL}/repos/dion-gram-anchor"
readonly DATA_ROOT=/path/to/data
readonly SIF="${PERSONAL}/containers/dion-ngc-26.06-slingshot.sif"

export APPTAINER_CACHEDIR="${PERSONAL}/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/${USER}-apptainer-${SLURM_JOB_ID}"
export APPTAINERENV_NCCL_NET_PLUGIN=ofi
export APPTAINERENV_FI_PROVIDER=cxi
export APPTAINERENV_FI_MR_CACHE_MONITOR=userfaultfd
export APPTAINERENV_FI_CXI_DISABLE_HOST_REGISTER=1
export APPTAINERENV_FI_CXI_DEFAULT_CQ_SIZE=131072
export APPTAINERENV_FI_CXI_RDZV_PROTO=alt_read
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_DEBUG_SUBSYS=INIT,NET
export APPTAINERENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export SLURM_MPI_TYPE=pmix
export WANDB_NAME="dinov2_hybrid_distractor_null_local60_gram_refinement_linear_ramp_bs64_128gpu_cluster_b_${SLURM_JOB_ID}"
export WANDB_RUN_GROUP="bacteria_dinov2_gram_refinement_linear_ramp_cluster_b"
unset SINGULARITY_CACHEDIR

readonly RESUME_CKPT="/path/to/checkpoints/checkpoint_12_59_01_073022__03_09_26/last.ckpt"

mkdir -p "${APPTAINER_TMPDIR}" "${PERSONAL}/runs/dion/slurm"
cd "${REPO}"

echo "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"

srun --mpi=pmix --kill-on-bad-exit=1 \
    apptainer exec --nv \
        --bind "${PERSONAL}:${PERSONAL}" \
        --bind "${DATA_ROOT}:${DATA_ROOT}" \
        "${SIF}" \
        python -m src.main \
            --config config_cluster_b/master_dion_hybrid_distractor_null_local_60pct_gram_refinement_linear_ramp.yaml \
            --encoder_weights "${RESUME_CKPT}" \
            --resume 1 \
            --accelerator gpu \
            --strategy ddp \
            --num_devices 4 \
            --num_nodes 32 \
            --batch_size 64 \
            --max_peaks 200 \
            --epochs 340 \
            --wandb_project dion-pretraining \
            --wandb_entity user \
            --log_wandb 1
