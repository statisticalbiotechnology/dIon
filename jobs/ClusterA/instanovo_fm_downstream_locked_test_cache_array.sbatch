#!/usr/bin/env bash
# Append only released InstaNovo-FM mean-pool test embeddings after final selection.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --job-name=instanovo_testcache
#SBATCH --output=/path/to/results/slurm_logs/instanovo_testcache-%A_%a.out
#SBATCH --error=/path/to/results/slurm_logs/instanovo_testcache-%A_%a.err
set -eo pipefail
cd /path/to/dIon
if [[ "${ALLOW_LOCKED_TEST:-0}" != 1 ]]; then echo 'Set ALLOW_LOCKED_TEST=1.' >&2; exit 2; fi
DION_PY=/path/to/conda-envs/dIon-env/bin/python
source jobs/ClusterA/instanovo_fm_common.sh
CACHE_ROOT=${CACHE_ROOT:-/path/to/work/instanovo_fm_downstream_validation_cache}
case "$SLURM_ARRAY_TASK_ID" in
  0) TASK=sqa; DATA=/path/to/data/downstream_data_casanovo/spectrum_quality/casanovo_foundation_sqa_v2_balanced_charge1_10; KIND=parquet ;;
  1) TASK=chimericity; DATA=/path/to/work/datasets/auxiliary_benchmarks_lance_v1/chimericity_hye; KIND=lance ;;
  2) TASK=oxidized_met; DATA=/path/to/work/datasets/auxiliary_benchmarks_lance_v1/oxidized_met; KIND=lance ;;
  3) TASK=retention_time; DATA=/path/to/work/datasets/auxiliary_benchmarks_lance_v1/retention_time; KIND=lance ;;
  *) exit 2 ;;
esac
WORK="$CACHE_ROOT/$TASK/artifacts"
CACHE="$CACHE_ROOT/$TASK/cache"
mkdir -p "$WORK" "$CACHE"
if [[ "$KIND" == parquet ]]; then INPUT=(--input-parquet "$DATA/test.parquet"); else INPUT=(--input-lance "$DATA/test.lance"); fi
"$DION_PY" -u scripts/export_external_downstream_input.py "${INPUT[@]}" --output-npz "$WORK/test.input.npz"
instanovo_embed "$WORK/test.input.npz" "$WORK/test.instanovo_fm.npz"
"$DION_PY" -u scripts/materialize_external_downstream_embeddings.py --output-cache "$CACHE" --test-embeddings "$WORK/test.instanovo_fm.npz"
