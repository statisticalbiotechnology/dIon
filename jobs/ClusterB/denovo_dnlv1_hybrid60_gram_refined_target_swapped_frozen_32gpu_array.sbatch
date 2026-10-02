#!/usr/bin/env bash
# Frozen Gram-refined dIon-de-novo-labeled-v1 (DNLv1) de novo training on matched DINO A+B mixtures.
# Each 95%-A global crop is mixed by the original pretraining policy and appears
# twice: `(M, precursor(A)) -> peptide(A)` and `(M, precursor(B)) -> peptide(B)`.
# Native clean V5 validation peptide precision selects checkpoints; no held-out
# transfer test is run from this experimental training job.
# Submit with: sbatch jobs/ClusterB/denovo_dnlv1_hybrid60_gram_refined_target_swapped_frozen_32gpu_array.sbatch
#SBATCH --job-name=denovo-v5-gram-mix
#SBATCH --account=YOUR_GPU_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --array=0
#SBATCH --output=/path/to/runs/slurm/%x-%A_%a.out
#SBATCH --error=/path/to/runs/slurm/%x-%A_%a.err

set -euo pipefail

readonly PERSONAL=/path/to/work
readonly REPO="${PERSONAL}/repos/dion-runs/gram-dense-dia"
readonly DATA_ROOT=/path/to/data
readonly SOURCE_DATA_ROOT="${DATA_ROOT}/denovo_dnlv1/lance"
readonly SIF="${PERSONAL}/containers/dion-ngc-26.06-slingshot.sif"
readonly GRAM_REFINED_LOCAL60_CKPT="${PERSONAL}/runs/dion/checkpoints/checkpoint_02_51_05_597141__13_09_26/epoch=298-dinov2_val_loss_epoch=6.43.ckpt"
readonly LOCAL_STAGE_PARENT="${SLURM_TMPDIR:-/tmp}/dion_denovo_v5_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly LOCAL_DATA_ROOT="${LOCAL_STAGE_PARENT}/v5_lance"

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

MASTER_CONFIG=config_cluster_b/master_denovo_dion_hybrid_dnlv1.yaml
RUN_LABEL=hybrid60_gram_refined_target_swapped_frozen
ENCODER_ARGS=(--encoder_weights "${GRAM_REFINED_LOCAL60_CKPT}")

for split in train.lance val.lance test.lance; do
  test -d "${SOURCE_DATA_ROOT}/${split}"
done
if [[ ${#ENCODER_ARGS[@]} -gt 0 && "${ENCODER_ARGS[0]}" == "--encoder_weights" ]]; then
  test -f "${ENCODER_ARGS[1]}"
fi

readonly RUN_NAME="denovo_dnlv1_encoderld_${RUN_LABEL}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly RUN_ROOT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1_gram_refined_target_swapped_frozen/${RUN_NAME}"
readonly LOG_ROOT="${PERSONAL}/runs/dion/logs/denovo_dnlv1/${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_ID="v5${SLURM_JOB_ID: -5}${SLURM_ARRAY_TASK_ID}"
export WANDB_RESUME=allow
export WANDB_RUN_GROUP=denovo_dnlv1_gram_refined_target_swapped_frozen
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
    test -d "${LOCAL_DATA_ROOT}/test.lance"
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
      --downstream_config config_cluster_b/downstream/denovo_dnlv1_dion_hybrid_gram_target_swapped_frozen.yaml \
      "${ENCODER_ARGS[@]}" \
      --freeze_encoder 1 \
      --downstream_root_dir "${LOCAL_DATA_ROOT}" \
      --output_dir "${RUN_ROOT}/checkpoint" \
      --log_dir "${LOG_ROOT}" \
      --accelerator gpu \
      --num_devices 4 \
      --num_nodes 8 \
      --strategy ddp \
      --num_workers 4 \
      --pin_mem 1 \
      --batch_size 25 \
      --epochs 80 \
      --max_steps 527680 \
      --limit_train_batches 1.0 \
      --limit_val_batches 200 \
      --limit_test_batches 0 \
      --validate_every_n_epochs 1 \
      --save_top_k 5 \
      --save_last 1 \
      --validate_on_end 1 \
      --test_on_end 0 \
      --log_wandb 1

# Resolve the selection result from ModelCheckpoint metadata. This is reported
# for provenance only; clean native V5 validation remains the sole selector.
mapfile -t LAST_CKPTS < <(find "${RUN_ROOT}" -type f -name last.ckpt -print)
if [[ ${#LAST_CKPTS[@]} -ne 1 ]]; then
  echo "Expected exactly one terminal last.ckpt under ${RUN_ROOT}; found ${#LAST_CKPTS[@]}." >&2
  printf "%s\n" "${LAST_CKPTS[@]}" >&2
  exit 1
fi
readonly LAST_CKPT="${LAST_CKPTS[0]}"
BEST_CKPT="$(apptainer exec --nv --bind "${PERSONAL}:${PERSONAL}" "${SIF}" python -c 'import sys, torch; checkpoint=torch.load(sys.argv[1], map_location="cpu", weights_only=False); states=checkpoint.get("callbacks", {}); paths=[state.get("best_model_path") for state in states.values() if isinstance(state, dict) and state.get("monitor") == "denovo_tf_val_pep_prec"]; assert len(paths) == 1 and paths[0], paths; print(paths[0])' "${LAST_CKPT}")"
test -f "${BEST_CKPT}"
echo "selected_native_v5_val_pep_prec_checkpoint=${BEST_CKPT}"
echo "done: $(date)"
echo "outputs: ${RUN_ROOT}"
