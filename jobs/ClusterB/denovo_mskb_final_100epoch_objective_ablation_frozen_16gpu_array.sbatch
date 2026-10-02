#!/usr/bin/env bash
# Validation-only frozen-encoder transfer diagnostic on official MSKB-final.
# Indices 0-2 use matched 100-epoch pretrained encoders; index 3 is the frozen random-encoder floor.
# Every encoder stays in eval/no-grad mode; only a fresh Casanovo decoder trains.
# Checkpoints select maximum validation peptide precision and retain terminal last.ckpt.
# Submit only the new random control with: sbatch --array=3 jobs/ClusterB/denovo_mskb_final_100epoch_objective_ablation_frozen_16gpu_array.sbatch

#SBATCH --job-name=denovo-obj100-val
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err

set -euo pipefail

readonly PERSONAL=/path/to/work
readonly REPO="${PERSONAL}/repos/dion-runs/denovo-mskb-final"
readonly DATA_ROOT=/path/to/data
readonly SOURCE_DATA_ROOT="${DATA_ROOT}/denovo_mskb_final/lance_peptidoform_val10k_seed42"
readonly SIF="${PERSONAL}/containers/dion-ngc-26.06-slingshot.sif"
readonly PURE_MIXTURE_CKPT="${PERSONAL}/runs/dion/checkpoints/checkpoint_17_16_33_941556__20_08_26/last.ckpt"
readonly MIXTURE_FREE_CKPT="${PERSONAL}/runs/dion/checkpoints/checkpoint_20_00_00_019308__19_08_26/last.ckpt"
readonly DUAL_OBJECTIVE_CKPT="${PERSONAL}/runs/dion/checkpoints/checkpoint_20_39_05_791780__18_08_26/last.ckpt"
readonly LOCAL_STAGE_PARENT="${SLURM_TMPDIR:-/tmp}/dion_denovo_mskb_final_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly LOCAL_DATA_ROOT="${LOCAL_STAGE_PARENT}/lance"

export APPTAINER_CACHEDIR="${PERSONAL}/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/${USER}-apptainer-${SLURM_JOB_ID}"
export APPTAINERENV_NCCL_NET_PLUGIN=ofi
export APPTAINERENV_FI_PROVIDER=cxi
export APPTAINERENV_FI_MR_CACHE_MONITOR=userfaultfd
export APPTAINERENV_FI_CXI_DISABLE_HOST_REGISTER=1
export APPTAINERENV_FI_CXI_DEFAULT_CQ_SIZE=131072
export APPTAINERENV_FI_CXI_RDZV_PROTO=alt_read
export APPTAINERENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export SLURM_MPI_TYPE=pmix
export OMP_NUM_THREADS=1
unset SINGULARITY_CACHEDIR

case "${SLURM_ARRAY_TASK_ID}" in
  0) MASTER_CONFIG=config_cluster_b/master_denovo_dion_hybrid_mskb_final.yaml; RUN_LABEL=pure_mixture; ENCODER_ARGS=(--encoder_weights "${PURE_MIXTURE_CKPT}");;
  1) MASTER_CONFIG=config_cluster_b/master_denovo_dion_hybrid_mskb_final.yaml; RUN_LABEL=mixture_free_control; ENCODER_ARGS=(--encoder_weights "${MIXTURE_FREE_CKPT}");;
  2) MASTER_CONFIG=config_cluster_b/master_denovo_dion_hybrid_mskb_final.yaml; RUN_LABEL=dual_objective; ENCODER_ARGS=(--encoder_weights "${DUAL_OBJECTIVE_CKPT}");;
  3) MASTER_CONFIG=config_cluster_b/master_denovo_mskb_final_encoderld_scratch.yaml; RUN_LABEL=random_frozen_control; ENCODER_ARGS=();;
  *) echo "Unknown array index ${SLURM_ARRAY_TASK_ID}; expected 0-3." >&2; exit 2;;
esac

for split in train.lance val.lance; do
  test -d "${SOURCE_DATA_ROOT}/${split}"
done
test -f "${SOURCE_DATA_ROOT}/manifest.json"
if [[ ${#ENCODER_ARGS[@]} -gt 0 ]]; then
  test -f "${ENCODER_ARGS[1]}"
fi

readonly RUN_NAME="denovo_mskb_final_encoderld_${RUN_LABEL}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly RUN_ROOT="${PERSONAL}/runs/dion/checkpoints/denovo_mskb_final_100epoch_objective_ablation/${RUN_NAME}"
readonly LOG_ROOT="${PERSONAL}/runs/dion/logs/denovo_mskb_final/${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_GROUP=denovo_mskb_final_100epoch_objective_ablation_validation_only
mkdir -p "${APPTAINER_TMPDIR}" "${PERSONAL}/runs/dion/slurm" "${RUN_ROOT}" "${LOG_ROOT}"
cd "${REPO}"

# Each training node gets a private copy to keep concurrent array elements off shared Lance storage.
export SOURCE_DATA_ROOT LOCAL_DATA_ROOT
srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" \
  --ntasks-per-node=1 --cpus-per-task=1 --exact bash -lc '''
    set -euo pipefail
    source_kb=$(du -sk "${SOURCE_DATA_ROOT}" | awk "{print \$1}")
    free_kb=$(df -Pk "${SLURM_TMPDIR:-/tmp}" | awk "NR == 2 {print \$4}")
    required_kb=$((source_kb + 4 * 1024 * 1024))
    if [[ "${free_kb}" -lt "${required_kb}" ]]; then
      echo "Insufficient local scratch on $(hostname): ${free_kb} KiB available; ${required_kb} KiB required" >&2
      exit 1
    fi
    mkdir -p "${LOCAL_DATA_ROOT}"
    rsync -a --delete "${SOURCE_DATA_ROOT}/" "${LOCAL_DATA_ROOT}/"
    test -d "${LOCAL_DATA_ROOT}/train.lance"
    test -d "${LOCAL_DATA_ROOT}/val.lance"
  '''

echo "job=${SLURM_JOB_ID} array_index=${SLURM_ARRAY_TASK_ID} nodes=${SLURM_JOB_NUM_NODES}"
echo "variant=${RUN_LABEL} config=${MASTER_CONFIG} data=${LOCAL_DATA_ROOT}"
scontrol show hostnames "${SLURM_JOB_NODELIST}"

srun --mpi=pmix --kill-on-bad-exit=1 \
  apptainer exec --nv \
    --bind "${PERSONAL}:${PERSONAL}" \
    --bind "${DATA_ROOT}:${DATA_ROOT}" \
    "${SIF}" \
    python -u -m src.main \
      --config "${MASTER_CONFIG}" \
      "${ENCODER_ARGS[@]}" \
      --freeze_encoder 1 \
            --downstream_config config_cluster_b/downstream/denovo_mskb_final_100epoch_objective_ablation.yaml \
      --downstream_root_dir "${LOCAL_DATA_ROOT}" \
      --output_dir "${RUN_ROOT}/checkpoint" \
      --log_dir "${LOG_ROOT}" \
      --accelerator gpu \
      --num_devices 4 \
      --num_nodes 4 \
      --strategy ddp \
      --num_workers 4 \
      --pin_mem 1 \
      --batch_size 100 \
      --epochs 80 \
      --max_steps 149200 \
      --limit_train_batches 1.0 \
      --limit_val_batches 100 \
      --limit_test_batches 1.0 \
      --validate_every_n_epochs 1 \
      --save_top_k 5 \
      --save_last 1 \
      --test_on_end 0 \
      --log_wandb 1
