#!/usr/bin/env bash
# Run released supervised-contrastive GLEAMS v0.3 embeddings on the fixed
# charge-2--4 locked-test matched cohorts.
#
# Explicit locked test only:
#   sbatch --array=0-2 --export=ALL,ALLOW_LOCKED_TEST=1 jobs/ClusterA/gleams_v03_locked_test_array.sbatch

#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gleams_locked
#SBATCH --output=/path/to/results/slurm_logs/gleams_locked-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/gleams_locked-%A_%a.err

set -eo pipefail

DION_ROOT="${DION_ROOT:-/path/to/dIon}"
cd "${DION_ROOT}"

DION_PY="${DION_PY:-/path/to/conda-envs/dIon-env/bin/python}"
GLEAMS_ENV="${GLEAMS_ENV:-/path/to/conda-envs/gleams39}"
GLEAMS_PY="${GLEAMS_PY:-${GLEAMS_ENV}/bin/python}"
export LD_LIBRARY_PATH="${GLEAMS_ENV}/lib:${LD_LIBRARY_PATH:-}"
SEED="${SEED:-42}"
GLEAMS_BATCH_SIZE="${GLEAMS_BATCH_SIZE:-4096}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
PREPROCESS_CHUNK_SIZE="${PREPROCESS_CHUNK_SIZE:-256}"

if [[ ! -x "${DION_PY}" || ! -x "${GLEAMS_PY}" ]]; then
  echo "dIon or GLEAMS Python interpreter is unavailable." >&2
  exit 2
fi
if [[ "${ALLOW_LOCKED_TEST:-0}" != "1" ]]; then
  echo "Refusing locked test without ALLOW_LOCKED_TEST=1." >&2
  exit 2
fi
BENCHMARK_SPLIT="heldout_test"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CORPUS="bacterial"
    if false; then
      SOURCE="/path/to/data/probing_datasets/bacterial_paper_benchmarks/validation/charge2to4"
      RETRIEVAL_CONFIG="configs/probing_tasks/bacterial_paper_validation_retrieval.yaml"
      PAIRS_CONFIG="configs/probing_tasks/bacterial_paper_validation_pairs.yaml"
    else
      SOURCE="/path/to/data/probing_datasets/bacterial_paper_benchmarks/test_peptide_disjoint/charge2to4"
      RETRIEVAL_CONFIG="configs/probing_tasks/bacterial_paper_test_retrieval.yaml"
      PAIRS_CONFIG="configs/probing_tasks/bacterial_paper_test_pairs.yaml"
    fi
    ;;
  1)
    CORPUS="ninespecies_v2"
    if false; then
      SOURCE="/path/to/data/probing_datasets/ninespecies_v2_paper_validation/charge2to4"
      RETRIEVAL_CONFIG="configs/probing_tasks/ninespecies_v2_paper_validation_retrieval.yaml"
      PAIRS_CONFIG="configs/probing_tasks/ninespecies_v2_paper_validation_pairs.yaml"
    else
      SOURCE="/path/to/data/probing_datasets/ninespecies_v2_paper_test/charge2to4"
      RETRIEVAL_CONFIG="configs/probing_tasks/ninespecies_v2_paper_test_retrieval.yaml"
      PAIRS_CONFIG="configs/probing_tasks/ninespecies_v2_paper_test_pairs.yaml"
    fi
    ;;
  2)
    CORPUS="kingdoms"
    if false; then
      SOURCE="/path/to/data/probing_datasets/kingdoms_paper_benchmarks/validation/charge2to4"
      RETRIEVAL_CONFIG="configs/probing_tasks/kingdoms_paper_validation_retrieval.yaml"
      PAIRS_CONFIG="configs/probing_tasks/kingdoms_paper_validation_pairs.yaml"
    else
      SOURCE="/path/to/data/probing_datasets/kingdoms_paper_benchmarks/test/charge2to4"
      RETRIEVAL_CONFIG="configs/probing_tasks/kingdoms_paper_test_retrieval.yaml"
      PAIRS_CONFIG="configs/probing_tasks/kingdoms_paper_test_pairs.yaml"
    fi
    ;;
  *)
    echo "Only array indices 0 (bacterial), 1 (ninespecies_v2), and 2 (kingdoms) are valid." >&2
    exit 2
    ;;
esac

WORK="/path/to/data/probing_datasets/external_matched/gleams_v03/${CORPUS}_${BENCHMARK_SPLIT}_charge2to4"
if [[ "${BENCHMARK_SPLIT}" == "heldout_test" ]]; then
  DEFAULT_RESULTS_ROOT="/path/to/results/gleams_v03_${SLURM_ARRAY_JOB_ID}"
else
  DEFAULT_RESULTS_ROOT="${DION_ROOT}/results"
fi
RESULTS_ROOT="${RESULTS_ROOT:-${DEFAULT_RESULTS_ROOT}}"
RESULTS="${RESULTS_ROOT}/representation/external/gleams_v03/${CORPUS}_${BENCHMARK_SPLIT}_charge2to4"
mkdir -p "${WORK}" "${RESULTS}"

echo "start=$(date --iso-8601=seconds)"
echo "corpus=${CORPUS}; split=${BENCHMARK_SPLIT}; source=${SOURCE}"
echo "work=${WORK}; results=${RESULTS}"

"${DION_PY}" -u scripts/export_external_embedding_input.py \
  --input-parquet "${SOURCE}/retrieval.parquet" \
  --output-npz "${WORK}/retrieval_input.npz" \
  --peptide-id-column peptide_id \
  --partition-column species \
  --seed "${SEED}" \
  --metadata-columns peptide_id species source_row_index

"${GLEAMS_PY}" -u scripts/embed_gleams_benchmark.py \
  --input-npz "${WORK}/retrieval_input.npz" \
  --output-npz "${WORK}/retrieval_gleams.npz" \
  --seed "${SEED}" --batch-size "${GLEAMS_BATCH_SIZE}" \
  --num-preprocess-workers "${PREPROCESS_WORKERS}" \
  --preprocess-chunk-size "${PREPROCESS_CHUNK_SIZE}"

"${DION_PY}" -u scripts/materialize_external_matched_retrieval.py \
  --source-parquet "${SOURCE}/retrieval.parquet" \
  --external-embeddings "${WORK}/retrieval_gleams.npz" \
  --output-parquet "${WORK}/retrieval_matched.parquet"

"${DION_PY}" -u scripts/evaluate_external_embeddings.py \
  --embeddings-npz "${WORK}/retrieval_gleams.npz" \
  --evaluation-config "${RETRIEVAL_CONFIG}" \
  --dataset-parquet "${WORK}/retrieval_matched.parquet" \
  --output-report "${RESULTS}/gleams_retrieval.json" \
  --metrics cosine euclidean \
  --device cuda

"${DION_PY}" -u scripts/export_external_pair_input.py \
  --spectra-path "${SOURCE}/pair_discrimination/spectra.parquet" \
  --pairs-path "${SOURCE}/pair_discrimination/pairs.parquet" \
  --output-npz "${WORK}/pairs_input.npz" \
  --output-spectra "${WORK}/pairs_source_spectra.parquet" \
  --output-pairs "${WORK}/pairs_source_pairs.parquet" \
  --seed "${SEED}"

"${GLEAMS_PY}" -u scripts/embed_gleams_benchmark.py \
  --input-npz "${WORK}/pairs_input.npz" \
  --output-npz "${WORK}/pairs_gleams.npz" \
  --seed "${SEED}" --batch-size "${GLEAMS_BATCH_SIZE}" \
  --num-preprocess-workers "${PREPROCESS_WORKERS}" \
  --preprocess-chunk-size "${PREPROCESS_CHUNK_SIZE}"

"${DION_PY}" -u scripts/materialize_external_matched_pairs.py \
  --source-spectra "${WORK}/pairs_source_spectra.parquet" \
  --source-pairs "${WORK}/pairs_source_pairs.parquet" \
  --external-embeddings "${WORK}/pairs_gleams.npz" \
  --output-spectra "${WORK}/pairs_matched_spectra.parquet" \
  --output-pairs "${WORK}/pairs_matched_pairs.parquet" \
  --output-manifest "${WORK}/pairs_matched_manifest.json" \
  --seed "${SEED}"

"${DION_PY}" -u scripts/evaluate_external_pairs.py \
  --embeddings-npz "${WORK}/pairs_gleams.npz" \
  --pairs-path "${WORK}/pairs_matched_pairs.parquet" \
  --manifest-path "${WORK}/pairs_matched_manifest.json" \
  --evaluation-config "${PAIRS_CONFIG}" \
  --output-report "${RESULTS}/gleams_pairs.json" \
  --metrics cosine euclidean

echo "done=$(date --iso-8601=seconds)"
