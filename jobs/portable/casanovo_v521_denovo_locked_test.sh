#!/usr/bin/env bash
# Shared stock Casanovo v5.2.1 held-out decoder and scorer.
set -eo pipefail

if [[ "${ALLOW_LOCKED_TEST:-0}" != "1" ]]; then
  echo "Refusing held-out test inference without ALLOW_LOCKED_TEST=1." >&2
  exit 2
fi

DION_ROOT="${DION_ROOT:-/path/to/dIon}"
CASANOVO_ENV="${CASANOVO_V521_ENV:-/path/to/conda-envs/casanovo_5_2_1}"
CASANOVO_BIN="${CASANOVO_ENV}/bin/casanovo"
CASANOVO_PY="${CASANOVO_ENV}/bin/python"
CASANOVO_CKPT="${CASANOVO_V521_CKPT:-${HOME}/.cache/casanovo/casanovo_orbitrap_v5-2-0.ckpt}"
RESULTS_BASE="${RESULTS_BASE:-/path/to/results/denovo_eval/casanovo_v5_2_1}"
CASANOVO_BATCH_SIZE="${CASANOVO_BATCH_SIZE:-2048}"
STOCK_CONFIG="${CASANOVO_ENV}/lib/python3.10/site-packages/casanovo/config.yaml"

if [[ ! -x "${CASANOVO_BIN}" || ! -x "${CASANOVO_PY}" ]]; then
  echo "Casanovo v5.2.1 environment is unavailable: ${CASANOVO_ENV}" >&2
  exit 2
fi
if [[ ! -f "${CASANOVO_CKPT}" ]]; then
  echo "Casanovo checkpoint is unavailable: ${CASANOVO_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${STOCK_CONFIG}" || ! "${CASANOVO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Stock Casanovo config or positive batch size is unavailable." >&2
  exit 2
fi

case "${1:?Pass task 0 (MSKB) or 1 (Kingdoms)}" in
  0)
    COHORT="mskb_final_charge_lt5_raw_mgf"
    MGF="/path/to/data/casanovo_official/v5_mskb_final/mgf_data/mskb_final/charge1to4_raw/official_test.mgf"
    EXPECTED_INPUTS=196979
    SCORE_ARGS=()
    ;;
  1)
    COHORT="kingdoms_species_cap100k_full_denominator"
    KINGDOMS_ROOT="/path/to/data/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/casanovo_v5_2_1_charge1to4"
    MGF="${KINGDOMS_ROOT}/test_charge1to4_for_casanovo_v5_2_1.mgf"
    EXPECTED_INPUTS=4912728
    SCORE_ARGS=(
      --full-denominator-count 4926232
      --supported-to-full-index "${KINGDOMS_ROOT}/supported_to_full_test_index.npy"
    )
    ;;
  *)
    echo "Only task 0 (MSKB charge <5) and task 1 (Kingdoms) are valid." >&2
    exit 2
    ;;
esac

OUT="${RESULTS_BASE}/${COHORT}"
MZTAB="${OUT}/predictions.mztab"
METRICS="${OUT}/metrics.json"
CURVE="${OUT}/precision_coverage.csv"

if [[ ! -f "${MGF}" ]]; then
  echo "Input MGF is unavailable: ${MGF}" >&2
  exit 2
fi
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite an existing result directory: ${OUT}" >&2
  exit 2
fi
mkdir -p "${RESULTS_BASE}"

cd "${DION_ROOT}"
CONFIG_OVERRIDE="${SLURM_TMPDIR:-/tmp}/casanovo_v521_${SLURM_JOB_ID}_${1}.yaml"
trap 'rm -f "${CONFIG_OVERRIDE}"' EXIT
sed "s/^predict_batch_size: .*/predict_batch_size: ${CASANOVO_BATCH_SIZE}/" "${STOCK_CONFIG}" > "${CONFIG_OVERRIDE}"
echo "start=$(date --iso-8601=seconds)"
echo "cohort=${COHORT}; input=${MGF}; output=${OUT}; predict_batch_size=${CASANOVO_BATCH_SIZE}"
"${CASANOVO_BIN}" sequence \
  --model "${CASANOVO_CKPT}" \
  --config "${CONFIG_OVERRIDE}" \
  --output_dir "${OUT}" \
  --output_root predictions \
  --force_overwrite \
  "${MGF}"

if [[ ! -s "${MZTAB}" ]]; then
  echo "Casanovo produced no mzTab: ${MZTAB}" >&2
  exit 1
fi
ACTUAL_PSMS=$(awk -F '\t' '$1 == "PSM" { count += 1 } END { print count + 0 }' "${MZTAB}")
if (( ACTUAL_PSMS < 1 || ACTUAL_PSMS > EXPECTED_INPUTS )); then
  echo "Expected between 1 and ${EXPECTED_INPUTS} PSM rows but found ${ACTUAL_PSMS}." >&2
  exit 1
fi
echo "Casanovo emitted ${ACTUAL_PSMS}/${EXPECTED_INPUTS} PSM rows; omitted inputs are explicit no-prediction errors."

"${CASANOVO_PY}" -u scripts/evaluate_released_casanovo_mztab.py \
  --mgf "${MGF}" \
  --mztab "${MZTAB}" \
  --checkpoint "${CASANOVO_CKPT}" \
  --allow-missing-predictions \
  --output "${METRICS}" \
  --precision-coverage-output "${CURVE}" \
  "${SCORE_ARGS[@]}"

echo "done=$(date --iso-8601=seconds)"
