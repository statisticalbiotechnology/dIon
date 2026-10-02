"""Export one selected static pair benchmark for an external embedder.

Run under dIon-env. Pair selection reuses ``PairBenchmarkDataset`` so the
external model and dIon receive the same balanced protocol subset before
external preprocessing filters any spectra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embed_eval.pair_data import PairBenchmarkDataset

CODE_VERSION = "external_pair_input_v1"


def _flatten_ragged(rows: list[object], *, dtype) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(row, dtype=dtype) for row in rows]
    lengths = np.asarray([array.size for array in arrays], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)
    return values, offsets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-path", type=Path, required=True)
    parser.add_argument("--pairs-path", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-spectra", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs-per-partition-per-label", type=int, default=None)
    args = parser.parse_args()

    dataset = PairBenchmarkDataset(
        args.spectra_path,
        args.pairs_path,
        seed=args.seed,
        max_pairs_per_partition_per_label=args.max_pairs_per_partition_per_label,
    )
    table = dataset.table
    mz_values, peak_offsets = _flatten_ragged(
        table["mz_array"].to_pylist(), dtype=np.float32
    )
    intensity_values, intensity_offsets = _flatten_ragged(
        table["intensity_array"].to_pylist(), dtype=np.float32
    )
    if not np.array_equal(peak_offsets, intensity_offsets):
        raise ValueError("m/z and intensity arrays must have equal length per spectrum.")
    payload = {
        "export_row_index": np.arange(table.num_rows, dtype=np.int64),
        "spectrum_id": np.asarray(table["spectrum_id"].to_pylist()),
        "species": np.asarray(table["species"].to_pylist()),
        "peptide_ion_id": np.asarray(table["peptide_ion_id"].to_pylist()),
        "precursor_mz": np.asarray(table["precursor_mz"].to_pylist(), dtype=np.float32),
        "precursor_charge": np.asarray(table["precursor_charge"].to_pylist(), dtype=np.int16),
        "peak_offsets": peak_offsets,
        "mz_values": mz_values,
        "intensity_values": intensity_values,
    }
    for output_path in (args.output_npz, args.output_spectra, args.output_pairs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    pq.write_table(table, args.output_spectra)
    pq.write_table(dataset.pairs, args.output_pairs)
    manifest_path = args.output_npz.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "code_version": CODE_VERSION,
        "source_spectra": str(args.spectra_path.resolve()),
        "source_spectra_sha256": _sha256(args.spectra_path),
        "source_pairs": str(args.pairs_path.resolve()),
        "source_pairs_sha256": _sha256(args.pairs_path),
        "seed": args.seed,
        "max_pairs_per_partition_per_label": args.max_pairs_per_partition_per_label,
        "selected_spectra": table.num_rows,
        "selected_pairs": dataset.pairs.num_rows,
        "selection": {
            "source_pairs": dataset.selection.source_pairs,
            "selected_pairs_by_set_species_label": dataset.selection.selected_pairs_by_set_species_label,
        },
    }, indent=2, sort_keys=True) + "\n")
    print(f"Exported {table.num_rows:,} pair spectra: {args.output_npz}")
    print(f"Wrote {dataset.pairs.num_rows:,} selected pairs: {args.output_pairs}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
