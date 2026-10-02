#!/usr/bin/env bash
# Run the null-local view-consistency intervention for Hybrid-300 last.ckpt.
# Deliberately does not use set -u: shared cluster environment variables are optional.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/conda-envs/dIon-env/bin/python}"
CHECKPOINT="/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt"
REPORT="$ROOT/results/representation/precursor_counterfactual/hybrid_300_last_9spec_val_null_local_consistency.json"

mkdir -p "$(dirname "$REPORT")"
cd "$ROOT"
"$PYTHON_BIN" -u scripts/evaluate_precursor_null_local_consistency.py \
  --config configs/master_dion_hybrid_distractor_null_local.yaml \
  --pretrain_config configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml \
  --probing_config configs/probing_tasks/null_local_consistency_evaluation.yaml \
  --encoder_weights "$CHECKPOINT" \
  --accelerator gpu \
  --num_devices 1 \
  --max_peaks 200 \
  --embedding_readout backbone \
  --counterfactual_distance cosine \
  --counterfactual_output_report "$REPORT"

"$PYTHON_BIN" scripts/import_precursor_counterfactual_results.py
"$PYTHON_BIN" scripts/build_paper_results_dashboard.py
