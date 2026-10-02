#!/usr/bin/env bash
# Shared implementation for score-ranked prediction on the charge-limited MSKB-final test set.
set -euo pipefail

: "${PREDICTION_SOURCE:?Set PREDICTION_SOURCE to mskb or v5.}"
: "${PREDICTION_TIER:?Set PREDICTION_TIER to priority, lower, or lowest.}"
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

readonly PERSONAL=/path/to/work
readonly REPO="${PERSONAL}/repos/dion-runs/denovo-mskb-final"
readonly DATA_ROOT=/path/to/data
readonly SOURCE_ROOT="${DATA_ROOT}/denovo_mskb_final/lance_peptidoform_val10k_seed42"
readonly SOURCE_TEST_ROOT="${SOURCE_ROOT}/test_charge_lt5_for_casanovo_v_gt_5_0"
readonly SIF="${PERSONAL}/containers/dion-ngc-26.06-slingshot.sif"
readonly LOCAL_STAGE_PARENT="${SLURM_TMPDIR:-/tmp}/dion_charge_lt5_prediction_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly PREDICTION_NUM_NODES="${PREDICTION_NUM_NODES:-4}"
readonly PREDICTION_NUM_DEVICES="${PREDICTION_NUM_DEVICES:-4}"
readonly PREDICTION_LIMIT_TEST_BATCHES="${PREDICTION_LIMIT_TEST_BATCHES:-1.0}"
readonly PREDICTION_NUM_WORKERS="${PREDICTION_NUM_WORKERS:-4}"
readonly LOCAL_DATA_ROOT="${LOCAL_STAGE_PARENT}/lance"

case "${PREDICTION_SOURCE}:${PREDICTION_TIER}:${SLURM_ARRAY_TASK_ID}" in
  mskb:priority:0)
    MODEL_LABEL=hybrid300_maxpeaks1000; MAX_PEAKS=1000; BATCH_SIZE=25
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_mskb_final_maxpeaks1000/denovo_mskb_final_encoderld_hybrid300_conditioned_maxpeaks1000_2359879_0/checkpoint_20_04_28_378965__13_09_26/epoch=77-denovo_tf_val_pep_prec=0.82.ckpt" ;;
  mskb:priority:1)
    MODEL_LABEL=hybrid300; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_mskb_final/denovo_mskb_final_encoderld_hybrid300_conditioned_2344884_2/checkpoint_02_03_48_526498__13_09_26/epoch=72-denovo_tf_val_pep_prec=0.80.ckpt" ;;
  mskb:priority:2)
    MODEL_LABEL=scratch_encoderld; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_mskb_final/denovo_mskb_final_encoderld_scratch_conditioned_2344805_0/checkpoint_01_47_09_792668__13_09_26/epoch=78-denovo_tf_val_pep_prec=0.74.ckpt" ;;
  mskb:priority:3)
    MODEL_LABEL=scratch_encoderld_maxpeaks1000; MAX_PEAKS=1000; BATCH_SIZE=25
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_mskb_final_maxpeaks1000/denovo_mskb_final_encoderld_scratch_encoderld_conditioned_maxpeaks1000_2465043_0/checkpoint_03_37_19_137171__15_09_26/epoch=79-denovo_tf_val_pep_prec=0.76.ckpt" ;;
  mskb:lower:0)
    MODEL_LABEL=hybrid300_frozen; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_mskb_final_frozen/denovo_mskb_final_encoderld_hybrid300_conditioned_frozen_2344517_0/checkpoint_06_53_54_873542__13_09_26/epoch=74-denovo_tf_val_pep_prec=0.51.ckpt" ;;
  mskb:lower:1)
    MODEL_LABEL=local60_gram_refined_frozen; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_mskb_final_gram_refined_frozen/denovo_mskb_final_encoderld_hybrid60_gram_refined_conditioned_frozen_2360329_0/checkpoint_17_54_00_299663__13_09_26/epoch=76-denovo_tf_val_pep_prec=0.67.ckpt" ;;
  v5:priority:0)
    MODEL_LABEL=hybrid300_maxpeaks1000; MAX_PEAKS=1000; BATCH_SIZE=25
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1_maxpeaks1000/denovo_dnlv1_encoderld_hybrid300_conditioned_maxpeaks1000_2359880_0/checkpoint_01_33_53_276019__14_09_26/epoch=27-denovo_tf_val_pep_prec=0.62.ckpt" ;;
  v5:priority:1)
    MODEL_LABEL=hybrid300; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1/denovo_dnlv1_encoderld_hybrid300_conditioned_2344260_2/checkpoint_06_55_38_836668__13_09_26/epoch=37-denovo_tf_val_pep_prec=0.60.ckpt" ;;
  v5:priority:2)
    MODEL_LABEL=scratch_encoderld; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1/denovo_dnlv1_encoderld_scratch_encoderld_conditioned_2346267_0/checkpoint_04_27_32_377495__13_09_26/epoch=41-denovo_tf_val_pep_prec=0.57.ckpt" ;;
  v5:priority:3)
    MODEL_LABEL=scratch_encoderld_maxpeaks1000; MAX_PEAKS=1000; BATCH_SIZE=25
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1_maxpeaks1000/denovo_dnlv1_encoderld_scratch_encoderld_conditioned_maxpeaks1000_2465044_0/checkpoint_03_48_58_284008__15_09_26/epoch=26-denovo_tf_val_pep_prec=0.59.ckpt" ;;
  v5:lower:0)
    MODEL_LABEL=local60_gram_refined_frozen; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1_gram_refined_frozen/denovo_dnlv1_encoderld_hybrid60_gram_refined_conditioned_frozen_2360330_0/checkpoint_17_55_52_479329__13_09_26/epoch=79-denovo_tf_val_pep_prec=0.47.ckpt" ;;
  v5:lower:1)
    MODEL_LABEL=hybrid300_frozen_best; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1_frozen/denovo_dnlv1_encoderld_hybrid300_conditioned_frozen_2344518_0/checkpoint_06_56_55_277799__13_09_26/epoch=39-denovo_tf_val_pep_prec=0.35.ckpt" ;;
  v5:lower:2)
    MODEL_LABEL=hybrid300_frozen_late; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1_frozen/denovo_dnlv1_encoderld_hybrid300_conditioned_frozen_2344518_0/checkpoint_06_56_55_277799__13_09_26/last.ckpt" ;;
  v5:lowest:0)
    MODEL_LABEL=local60_gram_refined; MAX_PEAKS=200; BATCH_SIZE=100
    DOWNSTREAM_CKPT="${PERSONAL}/runs/dion/checkpoints/denovo_dnlv1_gram_refined/denovo_dnlv1_encoderld_hybrid60_gram_refined_conditioned_2356464_0/checkpoint_15_18_10_789472__13_09_26/epoch=41-denovo_tf_val_pep_prec=0.60.ckpt" ;;
  *) echo "Unknown prediction selection: ${PREDICTION_SOURCE}:${PREDICTION_TIER}:${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

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

test -f "${DOWNSTREAM_CKPT}"
test -f "${SOURCE_TEST_ROOT}/manifest.json"
test -d "${SOURCE_TEST_ROOT}/test.lance"

readonly RUN_NAME="mskb_final_charge_lt5_${PREDICTION_SOURCE}_${PREDICTION_TIER}_${MODEL_LABEL}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly RUN_ROOT="${PERSONAL}/runs/dion/predictions/mskb_final_charge_lt5/${PREDICTION_SOURCE}/${RUN_NAME}"
readonly LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${APPTAINER_TMPDIR}" "${PERSONAL}/runs/dion/slurm" "${RUN_ROOT}" "${LOG_ROOT}"
cd "${REPO}"

# Only the immutable charge-limited test package is staged; eval_only never opens train or validation.
export SOURCE_TEST_ROOT LOCAL_DATA_ROOT
srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpus-per-task=1 --exact bash -lc '
  set -euo pipefail
  source_kb=$(du -sk "${SOURCE_TEST_ROOT}" | awk "{print \$1}")
  free_kb=$(df -Pk "${SLURM_TMPDIR:-/tmp}" | awk "NR == 2 {print \$4}")
  required_kb=$((source_kb + 2 * 1024 * 1024))
  if [[ "${free_kb}" -lt "${required_kb}" ]]; then
    echo "Insufficient local scratch on $(hostname): ${free_kb} KiB available; ${required_kb} KiB required" >&2
    exit 1
  fi
  mkdir -p "${LOCAL_DATA_ROOT}"
  rsync -a --delete "${SOURCE_TEST_ROOT}/" "${LOCAL_DATA_ROOT}/test_charge_lt5_for_casanovo_v_gt_5_0/"
  test -d "${LOCAL_DATA_ROOT}/test_charge_lt5_for_casanovo_v_gt_5_0/test.lance"
'

echo "job=${SLURM_JOB_ID} index=${SLURM_ARRAY_TASK_ID} source=${PREDICTION_SOURCE} tier=${PREDICTION_TIER} model=${MODEL_LABEL} max_peaks=${MAX_PEAKS}"
echo "checkpoint=${DOWNSTREAM_CKPT}"

# The 1000-peak decoder path OOMed with the default cuDNN SDP backend.
SDP_ARGS=()
if (( MAX_PEAKS == 1000 )); then
  SDP_ARGS=(--disable_cudnn_sdp 1)
fi

srun --mpi=pmix --kill-on-bad-exit=1 apptainer exec --nv --bind "${PERSONAL}:${PERSONAL}" --bind "${DATA_ROOT}:${DATA_ROOT}" "${SIF}" python -u -m src.main \
  --config config_cluster_b/master_denovo_dion_hybrid_mskb_final.yaml \
  --downstream_config config_cluster_b/downstream/denovo_mskb_final_charge_lt5_prediction.yaml \
  --downstream_weights "${DOWNSTREAM_CKPT}" \
  --downstream_root_dir "${LOCAL_DATA_ROOT}" \
  --output_dir "${RUN_ROOT}/checkpoint" --log_dir "${LOG_ROOT}" \
  --accelerator gpu --num_devices "${PREDICTION_NUM_DEVICES}" --num_nodes "${PREDICTION_NUM_NODES}" --strategy ddp \
  --num_workers "${PREDICTION_NUM_WORKERS}" --pin_mem 1 --batch_size "${BATCH_SIZE}" --max_peaks "${MAX_PEAKS}" "${SDP_ARGS[@]}" \
  --eval_only 1 --validate_on_end 0 --test_on_end 1 --limit_test_batches "${PREDICTION_LIMIT_TEST_BATCHES}" \
  --save_top_k 0 --save_last 0 --log_wandb 0

MZTAB=$(find "${LOG_ROOT}" -type f -name predictions_table.mzTab -print -quit)
test -n "${MZTAB}"
ACTUAL_ROWS=$(awk -F "\t" "\$1 == \"PSM\" { count++ } END { print count + 0 }" "${MZTAB}")
NUMERIC_SCORES=$(awk -F "\t" "\$1 == \"PSM\" && \$9 ~ /^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ { count++ } END { print count + 0 }" "${MZTAB}")
EXPECTED_ROWS=$(srun --nodes=1 --ntasks=1 --exact apptainer exec --bind "${PERSONAL}:${PERSONAL}" --bind "${DATA_ROOT}:${DATA_ROOT}" "${SIF}" python -c 'import lance, sys; print(lance.dataset(sys.argv[1]).count_rows())' "${LOCAL_DATA_ROOT}/test_charge_lt5_for_casanovo_v_gt_5_0/test.lance")
if [[ "${PREDICTION_LIMIT_TEST_BATCHES}" == "1.0" ]]; then
  if [[ "${ACTUAL_ROWS}" -ne "${EXPECTED_ROWS}" || "${NUMERIC_SCORES}" -ne "${EXPECTED_ROWS}" ]]; then
    echo "Invalid prediction output: expected=${EXPECTED_ROWS} rows=${ACTUAL_ROWS} numeric_scores=${NUMERIC_SCORES}" >&2
    exit 1
  fi
else
  if [[ "${ACTUAL_ROWS}" -le 0 || "${NUMERIC_SCORES}" -ne "${ACTUAL_ROWS}" || "${ACTUAL_ROWS}" -gt "${EXPECTED_ROWS}" ]]; then
    echo "Invalid limited prediction output: expected_at_most=${EXPECTED_ROWS} rows=${ACTUAL_ROWS} numeric_scores=${NUMERIC_SCORES}" >&2
    exit 1
  fi
fi
echo "validated_predictions=${MZTAB} rows=${ACTUAL_ROWS} expected_full_rows=${EXPECTED_ROWS}"
