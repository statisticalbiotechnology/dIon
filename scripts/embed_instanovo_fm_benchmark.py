"""Embed a dIon external-input NPZ with released InstaNovo-FM v0.1.0.

Run under an isolated environment containing ``instanovo-fm==0.1.0``. Inputs
come from ``export_external_embedding_input.py``; outputs preserve the same
row-aligned metadata as the Casanovo and GLEAMS runners, so they can be passed
directly to dIon's external retrieval, pair, and frozen-head cache tools.

The primary ``mean_pool`` readout is the released paper's mean over valid peak
tokens. ``latent`` is retained as an explicit secondary readout only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.instanovo_fm_preprocessing import DEFAULT_PREPROCESSING, process_spectrum


CODE_VERSION = "instanovo_fm_v0_1_0_external_export_v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        default="instanovo-fm-v0.1.0",
        help="Released registry name or an explicit local released checkpoint path.",
    )
    parser.add_argument("--readout", choices=("mean_pool", "latent"), default="mean_pool")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--num-preprocess-workers",
        type=int,
        default=0,
        help="CPU workers for spectrum preprocessing; zero processes inline.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16 CUDA autocast. Disabled by default for conservative parity.",
    )
    return parser.parse_args()


def _validate_input(input_data: np.lib.npyio.NpzFile) -> tuple[int, np.ndarray]:
    required = {
        "export_row_index",
        "precursor_mz",
        "precursor_charge",
        "peak_offsets",
        "mz_values",
        "intensity_values",
    }
    missing = sorted(required - set(input_data.files))
    if missing:
        raise ValueError(f"Input NPZ is missing required arrays: {missing}")
    count = len(input_data["export_row_index"])
    if any(len(input_data[key]) != count for key in ("precursor_mz", "precursor_charge")):
        raise ValueError("Input NPZ row metadata arrays are not aligned.")
    offsets = np.asarray(input_data["peak_offsets"], dtype=np.int64)
    mz_values = input_data["mz_values"]
    intensity_values = input_data["intensity_values"]
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(mz_values):
        raise ValueError("peak_offsets does not agree with flattened peak arrays.")
    if len(mz_values) != len(intensity_values) or (np.diff(offsets) < 0).any():
        raise ValueError("Invalid flattened peak arrays.")
    return count, offsets


def _pad_batch(spectra: list[np.ndarray]) -> torch.Tensor:
    longest = max(spectrum.shape[0] for spectrum in spectra)
    padded = np.zeros((len(spectra), longest, 2), dtype=np.float32)
    for index, spectrum in enumerate(spectra):
        padded[index, : spectrum.shape[0]] = spectrum
    return torch.from_numpy(padded)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    return torch.device(requested)


def _preprocess_one(
    index: int,
    offsets: np.ndarray,
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    precursor_mz: np.ndarray,
) -> tuple[int, np.ndarray]:
    return (
        index,
        process_spectrum(
            mz_values[offsets[index] : offsets[index + 1]],
            intensity_values[offsets[index] : offsets[index + 1]],
            float(precursor_mz[index]),
        ),
    )


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.num_preprocess_workers < 0:
        raise ValueError("--batch-size and --threads must be positive; --num-preprocess-workers cannot be negative.")
    torch.set_num_threads(args.threads)
    input_data = np.load(args.input_npz, allow_pickle=False)
    count, offsets = _validate_input(input_data)
    device = _resolve_device(args.device)

    from instanovo_fm.model.encoder import FoundationModel

    model, _ = FoundationModel.from_pretrained(args.checkpoint)
    model.to(device).eval()
    batches: list[np.ndarray] = []
    retained_indices: list[int] = []
    pending_spectra: list[np.ndarray] = []
    pending_indices: list[int] = []
    dummy_preprocessing = 0

    def flush() -> None:
        if not pending_spectra:
            return
        with torch.inference_mode():
            padded = _pad_batch(pending_spectra).to(device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.amp and device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                values = (
                    model.encode_mean_pooled(padded)
                    if args.readout == "mean_pool"
                    else model.encode(padded)
                )
        batches.append(values.float().cpu().numpy())
        retained_indices.extend(pending_indices)
        pending_spectra.clear()
        pending_indices.clear()

    # NPZ members are compressed independently; load these three once, not per spectrum.
    mz_values = input_data["mz_values"]
    intensity_values = input_data["intensity_values"]
    precursor_mz = input_data["precursor_mz"]
    preprocess = lambda index: _preprocess_one(
        index,
        offsets,
        mz_values,
        intensity_values,
        precursor_mz,
    )
    executor = (
        ThreadPoolExecutor(max_workers=args.num_preprocess_workers)
        if args.num_preprocess_workers > 0
        else None
    )
    def preprocess_batch(start: int) -> list[tuple[int, np.ndarray | None]]:
        indices = range(start, min(start + args.batch_size, count))
        iterator = executor.map(preprocess, indices) if executor is not None else map(preprocess, indices)
        return list(iterator)

    # One future keeps CPU preprocessing overlapped with the preceding CUDA forward
    # without enqueueing an unbounded number of spectra/futures for a large artifact.
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = prefetch_executor.submit(preprocess_batch, 0)
        with tqdm(total=count, desc="InstaNovo-FM encoding", unit="spectrum", dynamic_ncols=True) as progress:
            for start in range(0, count, args.batch_size):
                prepared = future.result()
                next_start = start + args.batch_size
                if next_start < count:
                    future = prefetch_executor.submit(preprocess_batch, next_start)
                for index, spectrum in prepared:
                    progress.update(1)
                    if spectrum.shape == (1, 2) and np.array_equal(
                        spectrum, np.asarray([[0.0, 1.0]], dtype=np.float32)
                    ):
                        dummy_preprocessing += 1
                    pending_spectra.append(spectrum)
                    pending_indices.append(index)
                flush()
    finally:
        prefetch_executor.shutdown(wait=True, cancel_futures=True)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    flush()

    retained = np.asarray(retained_indices, dtype=np.int64)
    dimension = int(getattr(model, "dim_model", 768))
    embeddings = (
        np.vstack(batches).astype(np.float32, copy=False)
        if batches
        else np.empty((0, dimension), dtype=np.float32)
    )
    payload = {
        key: input_data[key][retained]
        for key in input_data.files
        if key not in {"peak_offsets", "mz_values", "intensity_values"}
    }
    payload["embedding"] = embeddings
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)

    checkpoint_path = Path(args.checkpoint)
    manifest = {
        "code_version": CODE_VERSION,
        "input_npz": str(args.input_npz.resolve()),
        "input_sha256": _sha256(args.input_npz),
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path.is_file() else args.checkpoint,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path.is_file() else None,
        "instanovo_fm_version": importlib.metadata.version("instanovo-fm"),
        "readout": {
            "mean_pool": "released paper mean over valid contextualized peak-token embeddings",
            "latent": "released encode() L2-normalized CLS-like latent embedding",
        }[args.readout],
        "readout_mode": args.readout,
        "model_inputs": "MS2 peaks only; precursor metadata is not passed to InstaNovo-FM",
        "preprocessing": DEFAULT_PREPROCESSING.manifest(),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "num_preprocess_workers": args.num_preprocess_workers,
        "device": str(device),
        "amp": args.amp,
        "input_count": count,
        "embedded_count": int(len(retained)),
        "dummy_preprocessing": dummy_preprocessing,
        "embedding_dimension": int(embeddings.shape[1]),
    }
    manifest_path = args.output_npz.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Embedded {len(retained):,}/{count:,} spectra: {args.output_npz}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
