"""Export one full downstream split for an external spectrum embedder.

The exported ``export_row_index`` is exactly the index emitted by dIon's
SQA and auxiliary Lance dataloaders.  No task labels are exported or used by
the external model.
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

from scripts.export_external_embedding_input import REQUIRED_COLUMNS, _flatten_ragged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(value for value in path.rglob("*") if value.is_file())
    for value in paths:
        digest.update(str(value.relative_to(path if path.is_dir() else path.parent)).encode())
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-parquet", type=Path)
    source.add_argument("--input-lance", type=Path)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.input_parquet is not None:
        source_path = args.input_parquet
        table = pq.read_table(source_path)
        source_kind = "parquet"
    else:
        source_path = args.input_lance
        import lance

        table = lance.dataset(str(source_path)).to_table()
        source_kind = "lance"
    missing = sorted(REQUIRED_COLUMNS - set(table.column_names))
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")

    mz_values, peak_offsets = _flatten_ragged(table["mz_array"].to_pylist(), dtype=np.float32)
    intensity_values, intensity_offsets = _flatten_ragged(
        table["intensity_array"].to_pylist(), dtype=np.float32
    )
    if not np.array_equal(peak_offsets, intensity_offsets):
        raise ValueError("m/z and intensity arrays must have equal length per spectrum.")
    count = table.num_rows
    payload = {
        "export_row_index": np.arange(count, dtype=np.int64),
        "precursor_mz": np.asarray(table["precursor_mz"].to_pylist(), dtype=np.float32),
        "precursor_charge": np.asarray(table["precursor_charge"].to_pylist(), dtype=np.int16),
        "peak_offsets": peak_offsets,
        "mz_values": mz_values,
        "intensity_values": intensity_values,
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    manifest_path = args.output_npz.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "code_version": "external_downstream_input_v1",
        "source": str(source_path.resolve()),
        "source_kind": source_kind,
        "source_sha256": _sha256(source_path),
        "input_rows": count,
        "index_contract": "export_row_index equals the corresponding dIon split dataset index",
    }, indent=2, sort_keys=True) + "\n")
    print(f"Exported {count:,} downstream spectra: {args.output_npz}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
