#!/usr/bin/env bash
# Run the fixed 100%-B precursor-swap variants for Hybrid-300 last.ckpt.
# Deliberately does not use set -u: shared cluster environment variables are optional.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/conda-envs/dIon-env/bin/python}"
CHECKPOINT="/path/to/checkpoints/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_300epochs_maxpeaks200_bs128_64gpu_cluster_b/last.ckpt"
REPORT_ROOT="$ROOT/results/representation/precursor_counterfactual"

mkdir -p "$REPORT_ROOT"
cd "$ROOT"

COMMON=(
  --config configs/master_dion_hybrid_distractor_null_local.yaml
  --pretrain_config configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml
  --probing_config configs/probing_tasks/precursor_counterfactual_evaluation_100pctB.yaml
  --encoder_weights "$CHECKPOINT"
  --accelerator gpu
  --num_devices 1
  --max_peaks 200
)

"$PYTHON_BIN" -u scripts/evaluate_precursor_counterfactual.py \
  "${COMMON[@]}" \
  --embedding_readout backbone \
  --counterfactual_distance euclidean \
  --counterfactual_output_report "$REPORT_ROOT/hybrid_300_last_9spec_val_100pctB_euclidean.json"

"$PYTHON_BIN" -u scripts/evaluate_precursor_counterfactual.py \
  "${COMMON[@]}" \
  --embedding_readout dino_logits \
  --counterfactual_distance jensen_shannon \
  --counterfactual_output_report "$REPORT_ROOT/hybrid_300_last_9spec_val_100pctB_dino_head_js.json"
