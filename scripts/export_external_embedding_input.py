"""Export a deterministic Parquet benchmark subset for an external embedder.

Run this under dIon-env. The resulting NPZ intentionally contains only raw
model inputs and alignment metadata, so another environment can embed spectra
without importing dIon or reading Parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embed_eval.data import PeptideRetrievalDataset

CODE_VERSION = "external_embedding_input_v1"
REQUIRED_COLUMNS = {
    "mz_array",
    "intensity_array",
    "precursor_mz",
    "precursor_charge",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_ragged(rows: list[object], *, dtype) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(row, dtype=dtype) for row in rows]
    lengths = np.asarray([array.size for array in arrays], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)
    return values, offsets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--peptide-id-column", default=None)
    parser.add_argument("--partition-column", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-peptides-per-partition", type=int, default=None)
    parser.add_argument("--max-spectra-per-peptide", type=int, default=None)
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=[],
        help="Additional scalar/string columns retained in the output artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    table = pq.read_table(args.input_parquet)
    missing = sorted(REQUIRED_COLUMNS - set(table.column_names))
    if missing:
        raise ValueError(f"{args.input_parquet} is missing required columns: {missing}")
    metadata_columns = list(dict.fromkeys(args.metadata_columns))
    missing_metadata = sorted(set(metadata_columns) - set(table.column_names))
    if missing_metadata:
        raise ValueError(f"Missing requested metadata columns: {missing_metadata}")
    if (args.peptide_id_column is None) != (args.partition_column is None):
        raise ValueError("Set both peptide/partition columns or neither.")

    indices = list(range(table.num_rows))
    selection = None
    if args.peptide_id_column is not None:
        indices, selection = PeptideRetrievalDataset._select_indices(
            peptide_ids=table[args.peptide_id_column].to_pylist(),
            partition_ids=table[args.partition_column].to_pylist(),
            seed=args.seed,
            max_peptides_per_partition=args.max_peptides_per_partition,
            max_spectra_per_peptide=args.max_spectra_per_peptide,
        )
        if not indices:
            raise ValueError("No repeated peptide groups remain after selection.")
    selected = table.take(pa.array(indices, type=pa.int64()))
    mz_values, peak_offsets = _flatten_ragged(
        selected["mz_array"].to_pylist(), dtype=np.float32
    )
    intensity_values, intensity_offsets = _flatten_ragged(
        selected["intensity_array"].to_pylist(), dtype=np.float32
    )
    if not np.array_equal(peak_offsets, intensity_offsets):
        raise ValueError("m/z and intensity arrays must have equal length per spectrum.")
    payload: dict[str, np.ndarray] = {
        "export_row_index": np.asarray(indices, dtype=np.int64),
        "precursor_mz": np.asarray(selected["precursor_mz"].to_pylist(), dtype=np.float32),
        "precursor_charge": np.asarray(selected["precursor_charge"].to_pylist(), dtype=np.int16),
        "peak_offsets": peak_offsets,
        "mz_values": mz_values,
        "intensity_values": intensity_values,
    }
    for column in metadata_columns:
        payload[column] = np.asarray(selected[column].to_pylist())
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    manifest = {
        "code_version": CODE_VERSION,
        "input_parquet": str(args.input_parquet.resolve()),
        "input_sha256": _sha256(args.input_parquet),
        "input_rows": table.num_rows,
        "exported_rows": len(indices),
        "selection": None if selection is None else {
            "source_rows": selection.source_rows,
            "selected_rows": selection.selected_rows,
            "selected_groups": selection.selected_groups,
            "selected_groups_by_partition": selection.selected_groups_by_partition,
            "seed": args.seed,
            "max_peptides_per_partition": args.max_peptides_per_partition,
            "max_spectra_per_peptide": args.max_spectra_per_peptide,
        },
        "metadata_columns": metadata_columns,
    }
    manifest_path = args.output_npz.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(indices):,} spectra: {args.output_npz}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
