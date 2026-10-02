#!/usr/bin/env python
"""Create a charge-compatible v3 bacterial Lance dataset from regenerated v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lance
import pyarrow as pa
from tqdm.auto import tqdm


WRITE_BATCH_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    root = Path("/path/to/data/bacteria_PXD010000__PXD010613")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root / "annotated_regenerated_v2")
    parser.add_argument("--output-root", type=Path, default=root / "annotated_regenerated_v3")
    parser.add_argument("--max-charge", type=int, default=10)
    return parser.parse_args()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _rewrite_split(source: Path, destination: Path, *, max_charge: int) -> dict[str, int]:
    dataset = lance.dataset(source)
    stats: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    mode = "create"
    for batch in tqdm(dataset.scanner(batch_size=4096).to_batches(), desc=source.name, unit="batch"):
        for row in batch.to_pylist():
            stats["source_rows"] += 1
            charge = row.get("precursor_charge")
            if charge is None or int(charge) < 1:
                stats["dropped_nonpositive_or_missing_charge"] += 1
                continue
            if int(charge) > max_charge:
                stats["dropped_charge_above_max"] += 1
                stats[f"dropped_charge_{int(charge)}"] += 1
                continue
            rows.append(row)
            stats["written_rows"] += 1
            if len(rows) >= WRITE_BATCH_SIZE:
                lance.write_dataset(pa.Table.from_pylist(rows, schema=dataset.schema), destination, mode=mode)
                mode = "append"
                rows.clear()
    if rows:
        lance.write_dataset(pa.Table.from_pylist(rows, schema=dataset.schema), destination, mode=mode)
    return dict(sorted(stats.items()))


def main(args: argparse.Namespace) -> None:
    if args.max_charge < 1:
        raise ValueError("--max-charge must be positive.")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    source_manifest = input_root / "manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    split_paths = {name: input_root / name for name in ("train.lance", "val.lance", "test.lance")}
    for path in split_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    staging_root = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(staging_root)
    staging_root.mkdir(parents=True)
    try:
        split_stats = {
            name: _rewrite_split(path, staging_root / name, max_charge=args.max_charge)
            for name, path in split_paths.items()
        }
        manifest = {
            "dataset_name": "bacteria_annotated_regenerated_v3",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "builder": "scripts/rebuild_bacteria_annotated_v3_charge_filtered.py",
            "source": str(input_root),
            "source_manifest_sha256": _hash_file(source_manifest),
            "charge_policy": {
                "max_charge": args.max_charge,
                "description": "Keep annotated spectra with integer precursor charge in [1, max_charge]; do not clamp higher charges.",
            },
            "splits": split_stats,
        }
        (staging_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
