#!/usr/bin/env python
"""Convert an auxiliary benchmark's immutable Parquet splits to Lance.

The benchmark materializers publish Parquet as their portable source artifact.
dIon's normal downstream path is Lance-based, so this converter streams each
split into ``<input-root>/lance`` without changing rows, columns, or order.

Example:
    python scripts/data_prep/convert_auxiliary_benchmark_parquet_to_lance.py \
        --input-root /path/to/data/oxidized_met_benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_summary(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    return {
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "source_rows": parquet.metadata.num_rows,
        "source_row_groups": parquet.metadata.num_row_groups,
        "schema": str(parquet.schema_arrow),
    }


def convert_split(source: Path, destination: Path, *, batch_rows: int) -> int:
    """Stream one Parquet split into Lance and return its row count."""
    import lance
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source)
    mode = "create"
    written = 0
    for batch in parquet.iter_batches(batch_size=batch_rows):
        table = pa.Table.from_batches([batch], schema=parquet.schema_arrow)
        lance.write_dataset(table, str(destination), mode=mode)
        mode = "append"
        written += table.num_rows
    if written != parquet.metadata.num_rows:
        raise RuntimeError(
            f"Row-count mismatch for {source}: wrote {written}, "
            f"expected {parquet.metadata.num_rows}."
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, required=True,
        help="Directory containing train/val/test.parquet and manifest.json.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Lance output directory. Defaults to <input-root>/lance.",
    )
    parser.add_argument("--batch-rows", type=int, default=65_536)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output root. Required to overwrite anything.",
    )
    args = parser.parse_args()
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be positive.")

    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root / "lance").resolve()
    source_paths = {split: input_root / f"{split}.parquet" for split in SPLITS}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark Parquet splits: {missing}")
    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest_path}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    summaries = {split: _split_summary(path) for split, path in source_paths.items()}
    for split, source in source_paths.items():
        destination = output_root / f"{split}.lance"
        written = convert_split(source, destination, batch_rows=args.batch_rows)
        summaries[split]["lance"] = str(destination)
        summaries[split]["lance_rows"] = written
        print(f"{split}: {written:,} rows -> {destination}")

    conversion_manifest = {
        "builder": "scripts/data_prep/convert_auxiliary_benchmark_parquet_to_lance.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(input_root),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "batch_rows": args.batch_rows,
        "row_order": "preserved from each source Parquet split",
        "splits": summaries,
    }
    output_manifest = output_root / "conversion_manifest.json"
    output_manifest.write_text(json.dumps(conversion_manifest, indent=2) + "\n")
    print(f"Wrote conversion manifest: {output_manifest}")


if __name__ == "__main__":
    main()
