#!/usr/bin/env bash
# Full official MSKB-final test for one validation-selected de novo checkpoint.
# Set MASTER_CONFIG and DOWNSTREAM_CHECKPOINT explicitly; n_beams remains 1.
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${ALLOW_LOCKED_TEST:-0}" != 1 ]]; then echo 'Set ALLOW_LOCKED_TEST=1.' >&2; exit 2; fi
: "${MASTER_CONFIG:?Set the selected master config.}"
: "${DOWNSTREAM_CHECKPOINT:?Set the selected downstream checkpoint.}"
PYTHON_BIN=${PYTHON_BIN:-python}
OUT=${RESULTS_ROOT:-/path/to/results/denovo_mskb_final_locked_test_$(date -u +%Y%m%dT%H%M%SZ)}
"$PYTHON_BIN" -u -m src.main --config "$MASTER_CONFIG" --downstream_weights "$DOWNSTREAM_CHECKPOINT" --eval_only 1 --validate_on_end 0 --test_on_end 1 --limit_test_batches 1.0 --save_top_k 0 --save_last 0 --output_dir "$OUT" --log_dir "$OUT/logs" --log_wandb 1
