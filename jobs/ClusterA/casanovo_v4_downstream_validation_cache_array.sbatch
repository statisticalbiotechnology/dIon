#!/usr/bin/env bash
# Build only train/validation Casanovo-v4 frozen caches. Held-out test is intentionally absent.
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH --job-name=cas_v4_cache
#SBATCH --output=slurm_outs/cas_v4_cache-%A_%a.out
#SBATCH --error=slurm_outs/cas_v4_cache-%A_%a.err
set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
DION_PY=/path/to/conda-envs/dIon-env/bin/python
CASANOVO_PY=/path/to/conda-envs/casanovo40/bin/python
CASANOVO_CKPT=/path/to/work/checkpoints/casanovo_cache/casanovo_massivekb_v4.0.0.ckpt
CACHE_ROOT=${CACHE_ROOT:-/path/to/work/casanovo_v4_downstream_validation_cache}
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
for SPLIT in train val; do
  if [[ "$KIND" == parquet ]]; then INPUT=(--input-parquet "$DATA/$SPLIT.parquet"); else INPUT=(--input-lance "$DATA/$SPLIT.lance"); fi
  "$DION_PY" -u scripts/export_external_downstream_input.py "${INPUT[@]}" --output-npz "$WORK/$SPLIT.input.npz"
  "$CASANOVO_PY" -u scripts/embed_casanovo_v4_benchmark.py --input-npz "$WORK/$SPLIT.input.npz" --checkpoint "$CASANOVO_CKPT" --output-npz "$WORK/$SPLIT.casanovo_v4.npz" --pooling peak_mean --batch-size 1024 --num-preprocess-workers 8 --preprocess-chunk-size 256 --device cuda --include-preprocessing-failures
done
"$DION_PY" -u scripts/materialize_external_downstream_embeddings.py --output-cache "$CACHE" --train-embeddings "$WORK/train.casanovo_v4.npz" --val-embeddings "$WORK/val.casanovo_v4.npz"
