#!/usr/bin/env bash
# Run released InstaNovo-FM mean-pool embeddings plus the matched retrieval and
# pair evaluation suite. The array cells are bacterial, NineSpecies V2, and
# Kingdoms.
#
# Validation:
#   sbatch --array=0-2 jobs/ClusterA/instanovo_fm_representation_array.sbatch
#
# Held-out test (intentional, explicit opt-in): reports are staged outside
# the repository by default. Set RESULTS_ROOT only to use another staging root.
#   sbatch --array=0-2 --export=ALL,BENCHMARK_SPLIT=heldout_test,ALLOW_LOCKED_TEST=1 \
#     jobs/ClusterA/instanovo_fm_representation_array.sbatch

#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=instanovo_fm_repr
#SBATCH --output=/path/to/results/slurm_logs/instanovo_fm_repr-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/instanovo_fm_repr-%A_%a.err

set -eo pipefail

DION_ROOT="${DION_ROOT:-/path/to/dIon}"
cd "${DION_ROOT}"

DION_PY="${DION_PY:-/path/to/conda-envs/dIon-env/bin/python}"
source jobs/ClusterA/instanovo_fm_common.sh
BENCHMARK_SPLIT="${BENCHMARK_SPLIT:-validation}"
ALLOW_LOCKED_TEST="${ALLOW_LOCKED_TEST:-0}"
SEED="${SEED:-20260817}"
if [[ "${BENCHMARK_SPLIT}" != "validation" && "${BENCHMARK_SPLIT}" != "heldout_test" ]]; then
  echo "BENCHMARK_SPLIT must be validation or heldout_test." >&2
  exit 2
fi
if [[ "${BENCHMARK_SPLIT}" == "heldout_test" && "${ALLOW_LOCKED_TEST}" != "1" ]]; then
  echo "Refusing locked test without ALLOW_LOCKED_TEST=1." >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CORPUS="bacterial"
    if [[ "${BENCHMARK_SPLIT}" == "validation" ]]; then
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
    if [[ "${BENCHMARK_SPLIT}" == "validation" ]]; then
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
    if [[ "${BENCHMARK_SPLIT}" == "validation" ]]; then
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

WORK="/path/to/data/probing_datasets/external_matched/instanovo_fm/${CORPUS}_${BENCHMARK_SPLIT}_charge2to4"
DEFAULT_RESULTS_ROOT="/path/to/results/instanovo_fm_${BENCHMARK_SPLIT}_${SLURM_ARRAY_JOB_ID}"
RESULTS_ROOT="${RESULTS_ROOT:-${DEFAULT_RESULTS_ROOT}}"
RESULTS="${RESULTS_ROOT}/representation/external/instanovo_fm/${CORPUS}_${BENCHMARK_SPLIT}_charge2to4"
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

instanovo_embed "${WORK}/retrieval_input.npz" "${WORK}/retrieval_instanovo_fm.npz"

"${DION_PY}" -u scripts/materialize_external_matched_retrieval.py \
  --source-parquet "${SOURCE}/retrieval.parquet" \
  --external-embeddings "${WORK}/retrieval_instanovo_fm.npz" \
  --output-parquet "${WORK}/retrieval_matched.parquet"

"${DION_PY}" -u scripts/evaluate_external_embeddings.py \
  --embeddings-npz "${WORK}/retrieval_instanovo_fm.npz" \
  --evaluation-config "${RETRIEVAL_CONFIG}" \
  --dataset-parquet "${WORK}/retrieval_matched.parquet" \
  --output-report "${RESULTS}/instanovo_fm_retrieval.json" \
  --metrics cosine euclidean \
  --device cuda

"${DION_PY}" -u scripts/export_external_pair_input.py \
  --spectra-path "${SOURCE}/pair_discrimination/spectra.parquet" \
  --pairs-path "${SOURCE}/pair_discrimination/pairs.parquet" \
  --output-npz "${WORK}/pairs_input.npz" \
  --output-spectra "${WORK}/pairs_source_spectra.parquet" \
  --output-pairs "${WORK}/pairs_source_pairs.parquet" \
  --seed "${SEED}"

instanovo_embed "${WORK}/pairs_input.npz" "${WORK}/pairs_instanovo_fm.npz"

"${DION_PY}" -u scripts/materialize_external_matched_pairs.py \
  --source-spectra "${WORK}/pairs_source_spectra.parquet" \
  --source-pairs "${WORK}/pairs_source_pairs.parquet" \
  --external-embeddings "${WORK}/pairs_instanovo_fm.npz" \
  --output-spectra "${WORK}/pairs_matched_spectra.parquet" \
  --output-pairs "${WORK}/pairs_matched_pairs.parquet" \
  --output-manifest "${WORK}/pairs_matched_manifest.json" \
  --seed "${SEED}"

"${DION_PY}" -u scripts/evaluate_external_pairs.py \
  --embeddings-npz "${WORK}/pairs_instanovo_fm.npz" \
  --pairs-path "${WORK}/pairs_matched_pairs.parquet" \
  --manifest-path "${WORK}/pairs_matched_manifest.json" \
  --evaluation-config "${PAIRS_CONFIG}" \
  --output-report "${RESULTS}/instanovo_fm_pairs.json" \
  --metrics cosine euclidean

echo "done=$(date --iso-8601=seconds)"
