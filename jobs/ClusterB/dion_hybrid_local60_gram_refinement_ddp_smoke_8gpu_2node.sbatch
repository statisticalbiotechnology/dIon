#!/usr/bin/env bash
# Eight-GPU, two-node distributed smoke of late local-60 DINO resumed with Gram anchoring.
# Submit with: sbatch jobs/ClusterB/dion_hybrid_local60_gram_refinement_ddp_smoke_8gpu_2node.sbatch

#SBATCH --job-name=dino-gram-sm
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
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
export WANDB_NAME="gram_anchor_ddp_smoke_8gpu_2node_${SLURM_JOB_ID}"
export WANDB_RUN_GROUP="gram_anchor_ddp_smoke"
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
            --config config_cluster_b/master_dion_hybrid_distractor_null_local_60pct_gram_refinement_ddp_smoke.yaml \
            --encoder_weights "${RESUME_CKPT}" \
            --resume 1 \
            --accelerator gpu \
            --strategy ddp \
            --num_devices 4 \
            --num_nodes 2 \
            --batch_size 64 \
            --max_peaks 200 \
            --epochs 282 \
            --save_top_k 1 \
            --save_last 1 \
            --every_n_epochs 1 \
            --limit_train_batches 1 \
            --limit_val_batches 1 \
            --wandb_project dion-pretraining \
            --wandb_entity user \
            --log_wandb 1
