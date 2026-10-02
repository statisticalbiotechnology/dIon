#!/usr/bin/env bash
# Shared released InstaNovo-FM v0.1.0 encoder launcher for ClusterA arrays.
# The overlay is pure Python; it is imported through PYTHONPATH while the
# process continues to use dIon-env's CUDA-enabled PyTorch installation.

INSTANOVO_FM_PY="${INSTANOVO_FM_PY:-/path/to/conda-envs/dIon-env/bin/python}"
INSTANOVO_FM_OVERLAY="${INSTANOVO_FM_OVERLAY:-/path/to/conda-envs/instanovo-fm-gpu-overlay}"
INSTANOVO_FM_CKPT="${INSTANOVO_FM_CKPT:-/path/to/work/checkpoints/instanovo_fm/instanovo-fm-v0.1.0.ckpt}"
INSTANOVO_BATCH_SIZE="${INSTANOVO_BATCH_SIZE:-1024}"
INSTANOVO_THREADS="${INSTANOVO_THREADS:-8}"
INSTANOVO_PREPROCESS_WORKERS="${INSTANOVO_PREPROCESS_WORKERS:-8}"

if [[ ! -x "${INSTANOVO_FM_PY}" ]]; then
  echo "InstaNovo-FM Python is unavailable: ${INSTANOVO_FM_PY}" >&2
  exit 2
fi
if [[ ! -d "${INSTANOVO_FM_OVERLAY}" ]]; then
  echo "InstaNovo-FM package overlay is unavailable: ${INSTANOVO_FM_OVERLAY}" >&2
  exit 2
fi
if [[ ! -f "${INSTANOVO_FM_CKPT}" ]]; then
  echo "InstaNovo-FM released checkpoint is unavailable: ${INSTANOVO_FM_CKPT}" >&2
  exit 2
fi
if ! env PYTHONPATH="${INSTANOVO_FM_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" "${INSTANOVO_FM_PY}" -c 'import torch; assert torch.cuda.is_available(); torch.empty(1, device="cuda")' >/dev/null 2>&1; then
  echo "InstaNovo-FM requires dIon-env CUDA PyTorch or an equivalent CUDA-enabled interpreter." >&2
  exit 2
fi

instanovo_embed() {
  local input_npz="$1"
  local output_npz="$2"
  env PYTHONPATH="${INSTANOVO_FM_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${INSTANOVO_FM_PY}" -u scripts/embed_instanovo_fm_benchmark.py \
      --input-npz "${input_npz}" \
      --output-npz "${output_npz}" \
      --checkpoint "${INSTANOVO_FM_CKPT}" \
      --readout mean_pool \
      --batch-size "${INSTANOVO_BATCH_SIZE}" \
      --threads "${INSTANOVO_THREADS}" \
      --num-preprocess-workers "${INSTANOVO_PREPROCESS_WORKERS}" \
      --device cuda
}
