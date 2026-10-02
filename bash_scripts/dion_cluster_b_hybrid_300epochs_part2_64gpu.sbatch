#!/usr/bin/env bash
# Phase 2 resume of the 300-epoch hybrid distractor/null-local DINO run on 64 ClusterB GH200 GPUs.
# Submit with: sbatch bash_scripts/dion_cluster_b_hybrid_300epochs_part2_64gpu.sbatch

#SBATCH --job-name=dino-hyb-3p2
#SBATCH --exclude=n206
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/runs/slurm/%x-%j.out
#SBATCH --error=/path/to/runs/slurm/%x-%j.err

set -euo pipefail

readonly PERSONAL=/path/to/work
readonly REPO="${PERSONAL}/repos/dion"
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
export WANDB_NAME="dinov2_hybrid_distractor_null_local_300epochs_part2_maxpeaks200_bs128_64gpu_cluster_b"
export WANDB_RUN_GROUP="bacteria_dinov2_hybrid_300epochs_cluster_b"
unset SINGULARITY_CACHEDIR

: "${RESUME_CKPT:?Set RESUME_CKPT to the phase-one segment_end.ckpt path before submitting.}"

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
            --config config_cluster_b/master_dion_hybrid_distractor_null_local_300epochs.yaml \
            --encoder_weights "${RESUME_CKPT}" \
            --resume 1 \
            --accelerator gpu \
            --strategy ddp \
            --num_devices 4 \
            --num_nodes 16 \
            --batch_size 128 \
            --max_peaks 200 \
            --limit_train_batches 1.0 \
            --limit_val_batches 100 \
            --wandb_project dion-pretraining \
            --wandb_entity user \
            --log_wandb 1
