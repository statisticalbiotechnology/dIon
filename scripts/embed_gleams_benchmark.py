"""Embed an NPZ spectrum artifact with the official GLEAMS v0.3 model.

Run this under the dedicated ``gleams39`` environment. Inputs are produced by
``scripts/export_external_embedding_input.py`` under dIon-env. The encoder,
reference spectra, charge restriction, and preprocessing match GLEAMS v0.3.

``--num-preprocess-workers`` parallelizes only independent CPU preprocessing
and feature encoding. GPU inference remains ordered in the parent process, so
row alignment and model inputs are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as ss
from spectrum_utils.spectrum import MsmsSpectrum
from tqdm.auto import tqdm

# GLEAMS creates one Keras prediction sequence per flush. Grappler otherwise
# emits a large non-numerical TensorDataset warning for every such sequence.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CODE_VERSION = "gleams_official_export_v2_parallel_preprocessing"
_WORKER_ENCODER: Any | None = None
_WORKER_PREPROCESSING: dict[str, Any] | None = None
_WORKER_SPECTRUM: Any | None = None


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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-preprocess-workers",
        type=int,
        default=0,
        help="Spawned CPU workers for preprocessing/feature encoding; 0 is serial.",
    )
    parser.add_argument(
        "--preprocess-chunk-size",
        type=int,
        default=128,
        help="Spectra sent to one preprocessing worker at a time.",
    )
    return parser.parse_args()


def _preprocessing(config: Any) -> dict[str, Any]:
    return {
        "mz_min": config.fragment_mz_min,
        "mz_max": config.fragment_mz_max,
        "min_peaks": config.min_peaks,
        "min_mz_range": config.min_mz_range,
        "remove_precursor_tolerance": config.remove_precursor_tolerance,
        "min_intensity": config.min_intensity,
        "max_peaks_used": config.max_peaks_used,
        "scaling": config.scaling,
    }


def _build_feature_encoder(preprocessing: dict[str, Any]) -> tuple[Any, Any, Any]:
    from gleams import config
    from gleams.feature import encoder, spectrum

    return (
        encoder.MultipleEncoder([
            encoder.PrecursorEncoder(
                num_bits_mz=config.num_bits_precursor_mz,
                mz_min=config.precursor_mz_min,
                mz_max=config.precursor_mz_max,
                num_bits_mass=config.num_bits_precursor_mass,
                mass_min=config.precursor_mass_min,
                mass_max=config.precursor_mass_max,
                charge_max=config.precursor_charge_max,
            ),
            encoder.FragmentEncoder(
                min_mz=config.fragment_mz_min,
                max_mz=config.fragment_mz_max,
                bin_size=config.bin_size,
            ),
            encoder.ReferenceSpectraEncoder(
                filename=config.ref_spectra_filename,
                preprocessing=preprocessing,
                fragment_mz_tol=config.fragment_mz_tol,
                num_ref_spectra=config.num_ref_spectra,
            ),
        ]),
        spectrum,
        config,
    )


def _encode_record(
    record: tuple[int, float, int, np.ndarray, np.ndarray],
    *,
    feature_encoder: Any,
    spectrum_module: Any,
    minimum_charge: int,
    maximum_charge: int,
    preprocessing: dict[str, Any],
) -> tuple[int, Any | None, str | None]:
    index, precursor_mz, charge, mz, intensity = record
    if not minimum_charge <= charge <= maximum_charge:
        return index, None, "charge"
    spec = MsmsSpectrum(str(index), precursor_mz, charge, mz, intensity)
    # GLEAMS v0.3 caches preprocessing in this attribute. spectrum-utils 0.3.5
    # no longer initializes it, but otherwise preserves the same API.
    spec.is_processed = False
    spec = spectrum_module.preprocess(spec, **preprocessing)
    if not spec.is_valid:
        return index, None, "preprocessing"
    return index, feature_encoder.encode(spec), None


def _initialize_preprocess_worker(preprocessing: dict[str, Any], seed: int) -> None:
    # Spawned workers must never initialize TensorFlow/CUDA. They only create
    # released GLEAMS feature encoders and return sparse feature vectors.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gleams import rndm

    rndm.set_seeds(seed)
    global _WORKER_ENCODER, _WORKER_PREPROCESSING, _WORKER_SPECTRUM
    _WORKER_ENCODER, _WORKER_SPECTRUM, _ = _build_feature_encoder(preprocessing)
    _WORKER_PREPROCESSING = preprocessing


def _encode_chunk(
    records: list[tuple[int, float, int, np.ndarray, np.ndarray]],
) -> tuple[list[tuple[int, Any]], int, int]:
    if _WORKER_ENCODER is None or _WORKER_SPECTRUM is None or _WORKER_PREPROCESSING is None:
        raise RuntimeError("GLEAMS preprocessing worker was not initialized.")
    from gleams import config

    encoded: list[tuple[int, Any]] = []
    skipped_charge = 0
    skipped_preprocessing = 0
    for record in records:
        index, encoding, reason = _encode_record(
            record,
            feature_encoder=_WORKER_ENCODER,
            spectrum_module=_WORKER_SPECTRUM,
            minimum_charge=config.charges[0],
            maximum_charge=config.charges[1],
            preprocessing=_WORKER_PREPROCESSING,
        )
        if reason == "charge":
            skipped_charge += 1
        elif reason == "preprocessing":
            skipped_preprocessing += 1
        else:
            encoded.append((index, encoding))
    return encoded, skipped_charge, skipped_preprocessing


def _configure_tensorflow_memory_growth() -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"GLEAMS TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s).")
    else:
        print("GLEAMS TensorFlow found no GPU; inference will run on CPU.")


def _load_embedder(config: Any) -> Any:
    from gleams.nn import embedder

    model = embedder.Embedder(
        num_precursor_features=config.num_precursor_features,
        num_fragment_features=config.num_fragment_features,
        num_ref_spectra_features=config.num_ref_spectra,
        lr=config.lr,
        filename=config.model_filename,
    )
    model.load()
    return model


def _embed_batch(model: Any, encodings: list[Any], batch_size: int, feature_split: tuple[int, int]) -> np.ndarray:
    from gleams.nn import data_generator

    sequence = data_generator.EncodingsSequence(
        ss.vstack(encodings, format="csr"), batch_size, feature_split
    )
    # GLEAMS' checked-in generate_embeds.py uses predict_on_batch rather than
    # Embedder.embed(sequence). The latter reaches Keras' newer data adapter,
    # which can collapse the three model inputs under TensorFlow 2.7.
    keras_model = model._get_embedder_model()
    batches = [keras_model.predict_on_batch(sequence[index]) for index in range(len(sequence))]
    return np.vstack(batches).astype(np.float32, copy=False)


def _record_chunks(
    *,
    count: int,
    precursor_mz: np.ndarray,
    precursor_charge: np.ndarray,
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    peak_offsets: np.ndarray,
    chunk_size: int,
):
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        yield [
            (
                index,
                float(precursor_mz[index]),
                int(precursor_charge[index]),
                np.asarray(mz_values[peak_offsets[index]:peak_offsets[index + 1]], dtype=np.float64),
                np.asarray(intensity_values[peak_offsets[index]:peak_offsets[index + 1]], dtype=np.float32),
            )
            for index in range(start, stop)
        ]


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.preprocess_chunk_size < 1 or args.num_preprocess_workers < 0:
        raise ValueError("Batch size/chunk size must be positive and worker count non-negative.")

    from gleams import config, rndm

    # GLEAMS itself seeds this before its CLI imports the reference encoder.
    # Reapply it here so the fixed 500 reference spectra match v0.3 behavior.
    rndm.set_seeds(args.seed)
    input_data = np.load(args.input_npz, allow_pickle=True)
    required = {
        "export_row_index", "precursor_mz", "precursor_charge", "peak_offsets", "mz_values", "intensity_values",
    }
    missing = sorted(required - set(input_data.files))
    if missing:
        raise ValueError(f"Missing required input arrays: {missing}")
    count = len(input_data["export_row_index"])
    row_aligned = {"export_row_index", "precursor_mz", "precursor_charge"}
    if any(len(input_data[name]) != count for name in row_aligned):
        raise ValueError("Input NPZ row metadata arrays are not aligned.")

    preprocessing = _preprocessing(config)
    mz_values = input_data["mz_values"]
    intensity_values = input_data["intensity_values"]
    peak_offsets = np.asarray(input_data["peak_offsets"], dtype=np.int64)
    if peak_offsets.shape != (count + 1,) or peak_offsets[0] != 0:
        raise ValueError("peak_offsets must have shape [spectrum_count + 1] and start at zero.")
    if peak_offsets[-1] != len(mz_values) or len(mz_values) != len(intensity_values):
        raise ValueError("Flattened peak values do not agree with peak_offsets.")
    if (np.diff(peak_offsets) < 0).any():
        raise ValueError("peak_offsets must be non-decreasing.")
    precursor_mz = input_data["precursor_mz"]
    precursor_charge = input_data["precursor_charge"]

    pool = None
    feature_encoder = spectrum_module = None
    if args.num_preprocess_workers:
        pool = mp.get_context("spawn").Pool(
            processes=args.num_preprocess_workers,
            initializer=_initialize_preprocess_worker,
            initargs=(preprocessing, args.seed),
        )
        print(
            f"GLEAMS CPU preprocessing: {args.num_preprocess_workers} spawned workers, "
            f"chunk size {args.preprocess_chunk_size}."
        )
    else:
        feature_encoder, spectrum_module, _ = _build_feature_encoder(preprocessing)
        print("GLEAMS CPU preprocessing: serial.")

    # TensorFlow is initialized only in the parent, after spawned workers exist.
    _configure_tensorflow_memory_growth()
    model = _load_embedder(config)
    feature_split = (
        config.num_precursor_features,
        config.num_precursor_features + config.num_fragment_features,
    )
    valid_indices: list[int] = []
    embedding_batches: list[np.ndarray] = []
    batch_encodings: list[Any] = []
    batch_indices: list[int] = []
    skipped_charge = 0
    skipped_preprocessing = 0

    def flush_batch() -> None:
        if not batch_encodings:
            return
        embedding_batches.append(_embed_batch(model, batch_encodings, args.batch_size, feature_split))
        valid_indices.extend(batch_indices)
        batch_encodings.clear()
        batch_indices.clear()

    chunks = _record_chunks(
        count=count,
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
        mz_values=mz_values,
        intensity_values=intensity_values,
        peak_offsets=peak_offsets,
        chunk_size=args.preprocess_chunk_size,
    )
    try:
        with tqdm(total=count, desc="GLEAMS encoding", unit="spectrum", dynamic_ncols=True) as progress:
            if pool is not None:
                results = pool.imap(_encode_chunk, chunks, chunksize=1)
                for encoded, skipped_charge_chunk, skipped_preprocessing_chunk in results:
                    progress.update(min(args.preprocess_chunk_size, count - progress.n))
                    skipped_charge += skipped_charge_chunk
                    skipped_preprocessing += skipped_preprocessing_chunk
                    for index, encoding in encoded:
                        batch_encodings.append(encoding)
                        batch_indices.append(index)
                        if len(batch_encodings) == args.batch_size:
                            flush_batch()
            else:
                assert feature_encoder is not None and spectrum_module is not None
                for records in chunks:
                    for record in records:
                        index, encoding, reason = _encode_record(
                            record,
                            feature_encoder=feature_encoder,
                            spectrum_module=spectrum_module,
                            minimum_charge=config.charges[0],
                            maximum_charge=config.charges[1],
                            preprocessing=preprocessing,
                        )
                        if reason == "charge":
                            skipped_charge += 1
                        elif reason == "preprocessing":
                            skipped_preprocessing += 1
                        else:
                            batch_encodings.append(encoding)
                            batch_indices.append(index)
                            if len(batch_encodings) == args.batch_size:
                                flush_batch()
                    progress.update(len(records))
        flush_batch()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if embedding_batches:
        embeddings = np.vstack(embedding_batches).astype(np.float32, copy=False)
    else:
        embeddings = np.empty((0, config.embedding_size), dtype=np.float32)
    valid_indices_array = np.asarray(valid_indices, dtype=np.int64)
    output_payload = {
        key: input_data[key][valid_indices_array]
        for key in input_data.files
        if key not in {"peak_offsets", "mz_values", "intensity_values"}
    }
    output_payload["embedding"] = embeddings
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **output_payload)
    manifest = {
        "code_version": CODE_VERSION,
        "input_npz": str(args.input_npz.resolve()),
        "input_sha256": _sha256(args.input_npz),
        "checkpoint": str(Path(config.model_filename).resolve()),
        "checkpoint_sha256": _sha256(Path(config.model_filename)),
        "reference_spectra": str(Path(config.ref_spectra_filename).resolve()),
        "reference_spectra_sha256": _sha256(Path(config.ref_spectra_filename)),
        "gleams_embedding_dimension": config.embedding_size,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_preprocess_workers": args.num_preprocess_workers,
        "preprocess_chunk_size": args.preprocess_chunk_size,
        "input_count": count,
        "embedded_count": int(len(valid_indices_array)),
        "skipped_charge_outside_2_to_5": skipped_charge,
        "skipped_preprocessing": skipped_preprocessing,
        "preprocessing": preprocessing,
        "precursor_encoding": {
            "mz": [config.precursor_mz_min, config.precursor_mz_max, config.num_bits_precursor_mz],
            "neutral_mass": [config.precursor_mass_min, config.precursor_mass_max, config.num_bits_precursor_mass],
            "charge_max": config.precursor_charge_max,
        },
    }
    manifest_path = args.output_npz.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Embedded {len(valid_indices_array):,}/{count:,} spectra: {args.output_npz}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
