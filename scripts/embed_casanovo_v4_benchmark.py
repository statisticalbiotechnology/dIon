"""Embed an NPZ spectrum artifact with the released Casanovo v4.0.0 encoder.

Run this script under the isolated Casanovo v4 environment. Inputs are
produced by ``scripts/export_external_embedding_input.py`` under dIon-env.
The representation is the mean of the encoder's non-padding *peak* embeddings,
excluding Casanovo's prepended learned spectrum token, as described in the
Casanovo Foundation paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import spectrum_utils.spectrum as sus
import torch
from tqdm.auto import tqdm

from casanovo.denovo.model import Spec2Pep


CODE_VERSION = "casanovo_v4_foundation_export_v1"
PREPROCESSING = {
    "n_peaks": 150,
    "min_mz": 50.0,
    "max_mz": 2500.0,
    "min_intensity_fraction_of_base_peak": 0.01,
    "remove_precursor_tolerance_da": 2.0,
    "intensity_scaling": "root",
    "intensity_l2_normalization": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--pooling",
        choices=("peak_mean", "all_tokens_mean"),
        default="peak_mean",
        help=(
            "peak_mean follows the Foundation paper's mean of individual peak "
            "embeddings. all_tokens_mean additionally averages Casanovo's "
            "contextualized learned spectrum token."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-preprocess-workers",
        type=int,
        default=0,
        help="CPU worker processes for independent spectrum preprocessing; 0 keeps serial behavior.",
    )
    parser.add_argument(
        "--preprocess-chunk-size",
        type=int,
        default=128,
        help="Number of spectra dispatched to a preprocessing worker at once.",
    )
    parser.add_argument(
        "--include-preprocessing-failures",
        action="store_true",
        help=(
            "Preserve every input row by emitting an all-zero embedding for spectra "
            "that Casanovo preprocessing cannot represent. The output includes a "
            "preprocessing_failed mask; use only where a complete split-indexed "
            "downstream cache is required."
        ),
    )
    return parser.parse_args()


def _process_spectrum(
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    precursor_mz: float,
    precursor_charge: int,
) -> np.ndarray | None:
    """Apply Casanovo v4's SpectrumDataset peak processing verbatim."""
    spectrum = sus.MsmsSpectrum(
        "",
        precursor_mz,
        precursor_charge,
        np.asarray(mz_values, dtype=np.float64),
        np.asarray(intensity_values, dtype=np.float32),
    )
    try:
        spectrum.set_mz_range(PREPROCESSING["min_mz"], PREPROCESSING["max_mz"])
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.remove_precursor_peak(PREPROCESSING["remove_precursor_tolerance_da"], "Da")
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.filter_intensity(
            PREPROCESSING["min_intensity_fraction_of_base_peak"],
            PREPROCESSING["n_peaks"],
        )
        if len(spectrum.mz) == 0:
            raise ValueError
        spectrum.scale_intensity("root", 1)
        intensities = spectrum.intensity / np.linalg.norm(spectrum.intensity)
        return np.column_stack((spectrum.mz, intensities)).astype(np.float32)
    except ValueError:
        return None


def _process_indexed_spectrum(
    item: tuple[int, np.ndarray, np.ndarray, float, int],
) -> tuple[int, np.ndarray | None]:
    """Pool-safe preprocessing worker; preserves the original input position."""
    index, mz_values, intensity_values, precursor_mz, precursor_charge = item
    return index, _process_spectrum(
        mz_values, intensity_values, precursor_mz, precursor_charge
    )


def _iter_processed_spectra(
    *,
    count: int,
    offsets: np.ndarray,
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    precursor_mz: np.ndarray,
    precursor_charge: np.ndarray,
    workers: int,
    chunk_size: int,
):
    def items():
        for index in range(count):
            yield (
                index,
                mz_values[offsets[index] : offsets[index + 1]],
                intensity_values[offsets[index] : offsets[index + 1]],
                float(precursor_mz[index]),
                int(precursor_charge[index]),
            )

    if workers == 0:
        for item in items():
            yield _process_indexed_spectrum(item)
        return
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        yield from pool.imap(_process_indexed_spectrum, items(), chunksize=chunk_size)


def _pad_batch(spectra: list[np.ndarray], device: torch.device) -> torch.Tensor:
    max_peaks = max(spectrum.shape[0] for spectrum in spectra)
    batch = np.zeros((len(spectra), max_peaks, 2), dtype=np.float32)
    for index, spectrum in enumerate(spectra):
        batch[index, : spectrum.shape[0]] = spectrum
    return torch.from_numpy(batch).to(device)


def _embed_batch(
    model: Spec2Pep,
    spectra: list[np.ndarray],
    device: torch.device,
    pooling: str,
) -> np.ndarray:
    padded = _pad_batch(spectra, device)
    with torch.inference_mode():
        latent, padding_mask = model.encoder(padded)
        if pooling == "peak_mean":
            latent = latent[:, 1:, :]
            valid_tokens = ~padding_mask[:, 1:]
        elif pooling == "all_tokens_mean":
            valid_tokens = ~padding_mask
        else:  # argparse validates this; keep the encoder contract explicit.
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        embeddings = (
            (latent * valid_tokens.unsqueeze(-1)).sum(dim=1)
            / valid_tokens.sum(dim=1, keepdim=True)
        )
    return embeddings.float().cpu().numpy()


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_preprocess_workers < 0:
        raise ValueError("--num-preprocess-workers must be non-negative.")
    if args.preprocess_chunk_size < 1:
        raise ValueError("--preprocess-chunk-size must be positive.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")

    input_data = np.load(args.input_npz, allow_pickle=False)
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
        raise ValueError(f"{args.input_npz} is missing required arrays: {missing}")
    count = len(input_data["export_row_index"])
    if any(len(input_data[name]) != count for name in ("precursor_mz", "precursor_charge")):
        raise ValueError("Input NPZ row metadata arrays are not aligned.")
    offsets = np.asarray(input_data["peak_offsets"], dtype=np.int64)
    mz_values = input_data["mz_values"]
    intensity_values = input_data["intensity_values"]
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(mz_values):
        raise ValueError("peak_offsets does not agree with flattened peak arrays.")
    if len(mz_values) != len(intensity_values) or (np.diff(offsets) < 0).any():
        raise ValueError("Invalid flattened peak arrays.")

    model = Spec2Pep.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval().to(device)
    kept_input_indices: list[int] = []
    embedding_batches: list[np.ndarray] = []
    pending_spectra: list[np.ndarray] = []
    pending_indices: list[int] = []
    skipped_input_indices: list[int] = []
    skipped_preprocessing = 0

    def flush() -> None:
        if not pending_spectra:
            return
        embedding_batches.append(
            _embed_batch(model, pending_spectra, device, args.pooling)
        )
        kept_input_indices.extend(pending_indices)
        pending_spectra.clear()
        pending_indices.clear()

    if args.num_preprocess_workers:
        print(
            f"Casanovo CPU preprocessing: {args.num_preprocess_workers} spawned workers, "
            f"chunk size {args.preprocess_chunk_size}."
        )
    processed = _iter_processed_spectra(
        count=count,
        offsets=offsets,
        mz_values=mz_values,
        intensity_values=intensity_values,
        precursor_mz=input_data["precursor_mz"],
        precursor_charge=input_data["precursor_charge"],
        workers=args.num_preprocess_workers,
        chunk_size=args.preprocess_chunk_size,
    )
    for index, spectrum in tqdm(
        processed,
        total=count,
        desc="Casanovo v4 encoding",
        unit="spectrum",
        dynamic_ncols=True,
    ):
        if spectrum is None:
            skipped_preprocessing += 1
            skipped_input_indices.append(index)
            continue
        pending_spectra.append(spectrum)
        pending_indices.append(index)
        if len(pending_spectra) == args.batch_size:
            flush()
    flush()

    retained = np.asarray(kept_input_indices, dtype=np.int64)
    embeddings = (
        np.vstack(embedding_batches).astype(np.float32, copy=False)
        if embedding_batches
        else np.empty((0, int(model.hparams.dim_model)), dtype=np.float32)
    )
    preprocessing_failed = np.zeros(count, dtype=bool)
    preprocessing_failed[np.asarray(skipped_input_indices, dtype=np.int64)] = True
    model_embedded_count = int(len(retained))
    if args.include_preprocessing_failures:
        complete_embeddings = np.zeros((count, embeddings.shape[1]), dtype=np.float32)
        complete_embeddings[retained] = embeddings
        embeddings = complete_embeddings
        retained = np.arange(count, dtype=np.int64)
    payload = {
        key: input_data[key][retained]
        for key in input_data.files
        if key not in {"peak_offsets", "mz_values", "intensity_values"}
    }
    payload["embedding"] = embeddings
    payload["preprocessing_failed"] = preprocessing_failed[retained]
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    manifest = {
        "code_version": CODE_VERSION,
        "input_npz": str(args.input_npz.resolve()),
        "input_sha256": _sha256(args.input_npz),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "casanovo_version": "4.0.0",
        "embedding_dimension": int(embeddings.shape[1]),
        "pooling": {
            "peak_mean": "mean of valid encoder peak embeddings; excludes prepended learned spectrum token",
            "all_tokens_mean": "mean of valid encoder peak embeddings plus the contextualized learned spectrum token",
        }[args.pooling],
        "pooling_mode": args.pooling,
        "preprocessing": PREPROCESSING,
        "batch_size": args.batch_size,
        "device": str(device),
        "input_count": count,
        "embedded_count": model_embedded_count,
        "output_count": int(len(retained)),
        "skipped_preprocessing": skipped_preprocessing,
        "preprocessing_failure_handling": (
            "all_zero_embedding_with_mask" if args.include_preprocessing_failures else "dropped"
        ),
    }
    manifest_path = args.output_npz.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Embedded {len(retained):,}/{count:,} spectra: {args.output_npz}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
