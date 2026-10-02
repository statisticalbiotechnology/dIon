"""Materialize the exact retrieval spectra retained by an external embedder.

Run under dIon-env after external inference. The output Parquet is the
matched corpus for a dIon checkpoint evaluation; it is intentionally not
resampled after the external model filters spectra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--external-embeddings", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    args = parser.parse_args()

    table = pq.read_table(args.source_parquet)
    with np.load(args.external_embeddings, allow_pickle=False) as artifact:
        required = {"export_row_index", "precursor_mz", "precursor_charge", "embedding"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"Missing external artifact arrays: {missing}")
        indices = np.asarray(artifact["export_row_index"], dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(artifact["embedding"]):
            raise ValueError("export_row_index must align one-to-one with embeddings.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("External artifact contains duplicate export_row_index values.")
        if indices.size == 0 or indices.min() < 0 or indices.max() >= table.num_rows:
            raise ValueError("External artifact indexes lie outside the source Parquet.")
        matched = table.take(pa.array(indices, type=pa.int64()))
        mz = np.asarray(matched["precursor_mz"].to_pylist(), dtype=np.float32)
        charge = np.asarray(matched["precursor_charge"].to_pylist(), dtype=np.float32)
        if not np.allclose(mz, artifact["precursor_mz"], rtol=0.0, atol=1e-5):
            raise ValueError("External artifact precursor_mz does not match source rows.")
        if not np.array_equal(charge, np.asarray(artifact["precursor_charge"], dtype=np.float32)):
            raise ValueError("External artifact precursor_charge does not match source rows.")

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(matched, args.output_parquet)
    manifest_path = args.output_parquet.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "source_parquet": str(args.source_parquet.resolve()),
        "source_parquet_sha256": _sha256(args.source_parquet),
        "external_embeddings": str(args.external_embeddings.resolve()),
        "external_embeddings_sha256": _sha256(args.external_embeddings),
        "source_rows": table.num_rows,
        "matched_rows": matched.num_rows,
        "selection": "external_preprocessing_retained_rows",
    }, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {matched.num_rows:,} matched retrieval spectra: {args.output_parquet}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
