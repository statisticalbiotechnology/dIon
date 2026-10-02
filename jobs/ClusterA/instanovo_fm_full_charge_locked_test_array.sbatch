#!/usr/bin/env bash
# Run released InstaNovo-FM mean-pool embeddings on the unfiltered full-charge locked test.
# Requires explicit authorization; output is staged outside the repository.

#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=instanovo_fm_locked
#SBATCH --output=/path/to/results/slurm_logs/instanovo_fm_locked-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/instanovo_fm_locked-%A_%a.err

set -eo pipefail

DION_ROOT="${DION_ROOT:-/path/to/dIon}"
cd "${DION_ROOT}"

DION_PY="${DION_PY:-/path/to/conda-envs/dIon-env/bin/python}"
source jobs/ClusterA/instanovo_fm_common.sh
BENCHMARK_SPLIT="locked_test"
SEED="${SEED:-20260817}"
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CORPUS="bacterial"
    SOURCE="/path/to/data/probing_datasets/bacterial_paper_benchmarks/test_peptide_disjoint"
    RETRIEVAL_CONFIG="configs/probing_tasks/bacterial_paper_test_retrieval.yaml"
    PAIRS_CONFIG="configs/probing_tasks/bacterial_paper_test_pairs.yaml"
    ;;
  1)
    CORPUS="ninespecies_v2"
    SOURCE="/path/to/data/probing_datasets/ninespecies_v2_paper_test"
    RETRIEVAL_CONFIG="configs/probing_tasks/ninespecies_v2_paper_test_retrieval.yaml"
    PAIRS_CONFIG="configs/probing_tasks/ninespecies_v2_paper_test_pairs.yaml"
    ;;
  2)
    CORPUS="kingdoms"
    SOURCE="/path/to/data/probing_datasets/kingdoms_paper_benchmarks/test"
    RETRIEVAL_CONFIG="configs/probing_tasks/kingdoms_paper_test_retrieval.yaml"
    PAIRS_CONFIG="configs/probing_tasks/kingdoms_paper_test_pairs.yaml"
    ;;
  *)
    echo "Only array indices 0 (bacterial), 1 (ninespecies_v2), and 2 (kingdoms) are valid." >&2
    exit 2
    ;;
esac

WORK="/path/to/data/probing_datasets/external_matched/instanovo_fm/${CORPUS}_locked_test_full_charge"
RESULTS_ROOT="${RESULTS_ROOT:-/path/to/results/locked_test_instanovo_fm_full_charge_${SLURM_ARRAY_JOB_ID}}"
RESULTS="${RESULTS_ROOT}/representation/instanovo_fm/${CORPUS}"
mkdir -p "${WORK}" "${RESULTS}"

echo "start=$(date --iso-8601=seconds)"
echo "corpus=${CORPUS}; split=locked_test_full_charge; source=${SOURCE}"
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
