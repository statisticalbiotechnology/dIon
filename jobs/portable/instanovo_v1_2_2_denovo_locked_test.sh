#!/usr/bin/env bash
# Shared InstaNovo v1.2.2 held-out decoder and scorer.
set -eo pipefail

if [[ "${ALLOW_LOCKED_TEST:-0}" != "1" ]]; then
  echo "Refusing held-out test inference without ALLOW_LOCKED_TEST=1." >&2
  exit 2
fi

DION_ROOT="${DION_ROOT:-/path/to/dIon}"
INSTANOVO_ENV="${INSTANOVO_V122_ENV:-/path/to/conda-envs/instanovo_1_2_2}"
INSTANOVO_BIN="${INSTANOVO_ENV}/bin/instanovo"
CASANOVO_PY="${CASANOVO_V521_PY:-/path/to/conda-envs/casanovo_5_2_1/bin/python}"
CASANOVO_CKPT="${CASANOVO_V521_CKPT:-${HOME}/.cache/casanovo/casanovo_orbitrap_v5-2-0.ckpt}"
INSTANOVO_CKPT="${INSTANOVO_V120_CKPT:-/path/to/work/checkpoints/instanovo_v1_2_0/instanovo-v1.2.0.ckpt}"
RESULTS_BASE="${RESULTS_BASE:-/path/to/results/denovo_eval/instanovo_v1_2_2}"
INSTANOVO_BATCH_SIZE="${INSTANOVO_BATCH_SIZE:-1024}"
INSTANOVO_NUM_WORKERS="${INSTANOVO_NUM_WORKERS:-8}"

if [[ ! -x "${INSTANOVO_BIN}" || ! -x "${CASANOVO_PY}" ]]; then
  echo "Required InstaNovo or Casanovo executable is unavailable." >&2
  exit 2
fi
if [[ ! -f "${INSTANOVO_CKPT}" || ! -f "${CASANOVO_CKPT}" ]]; then
  echo "Required InstaNovo or Casanovo checkpoint is unavailable." >&2
  exit 2
fi
if [[ ! "${INSTANOVO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ || ! "${INSTANOVO_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "InstaNovo batch size must be positive and worker count non-negative." >&2
  exit 2
fi

case "${1:?Pass task 0 (MSKB) or 1 (Kingdoms)}" in
  0)
    COHORT="mskb_final_charge_lt5_raw_mgf"
    MGF="/path/to/data/casanovo_official/v5_mskb_final/mgf_data/mskb_final/charge1to4_raw/official_test.mgf"
    EXPECTED_INPUTS=196979
    ;;
  1)
    COHORT="kingdoms_species_cap100k_full_charge"
    KINGDOMS_ROOT="/path/to/data/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/instanovo_v1_2_2_charge1to10"
    MGF="${KINGDOMS_ROOT}/test_charge1to10_for_instanovo_v1_2_2.mgf"
    EXPECTED_INPUTS=4926232
    ;;
  *)
    echo "Only task 0 (MSKB charge <5) and task 1 (full-charge Kingdoms) are valid." >&2
    exit 2
    ;;
esac

OUT="${RESULTS_BASE}/${COHORT}"
PREDICTIONS_PARTIAL="${OUT}/predictions.csv.partial"
PREDICTIONS="${OUT}/predictions.csv"
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
mkdir -p "${OUT}"

# InstaNovo uses multiprocessing temporaries. Keep them off project storage;
# results themselves remain under the persistent OUT directory above.
export TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"
cd "${DION_ROOT}"
echo "start=$(date --iso-8601=seconds)"
echo "cohort=${COHORT}; input=${MGF}; output=${OUT}; batch_size=${INSTANOVO_BATCH_SIZE}"

"${INSTANOVO_BIN}" transformer predict \
  --data-path "${MGF}" \
  --output-path "${PREDICTIONS_PARTIAL}" \
  --instanovo-model "${INSTANOVO_CKPT}" \
  --denovo \
  num_beams=1 \
  use_knapsack=false \
  batch_size="${INSTANOVO_BATCH_SIZE}" \
  num_workers="${INSTANOVO_NUM_WORKERS}" \
  fp16=true

if [[ ! -s "${PREDICTIONS_PARTIAL}" ]]; then
  echo "InstaNovo produced no prediction CSV: ${PREDICTIONS_PARTIAL}" >&2
  exit 1
fi

"${CASANOVO_PY}" - "${PREDICTIONS_PARTIAL}" "${EXPECTED_INPUTS}" <<'PY'
import csv
import sys

path, expected = sys.argv[1], int(sys.argv[2])
seen = set()
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        index = int(row["prediction_id"])
        if index < 0 or index >= expected or index in seen:
            raise SystemExit(f"Invalid or duplicate prediction_id: {index}")
        seen.add(index)
if not seen:
    raise SystemExit("No emitted InstaNovo predictions.")
print(f"InstaNovo emitted {len(seen)}/{expected} valid prediction rows.")
PY

# A successful run has a complete, persistent CSV before downstream scoring.
mv "${PREDICTIONS_PARTIAL}" "${PREDICTIONS}"

"${CASANOVO_PY}" -u scripts/evaluate_released_instanovo_csv.py \
  --mgf "${MGF}" \
  --predictions-csv "${PREDICTIONS}" \
  --mass-checkpoint "${CASANOVO_CKPT}" \
  --output "${METRICS}" \
  --precision-coverage-output "${CURVE}"

echo "done=$(date --iso-8601=seconds)"
