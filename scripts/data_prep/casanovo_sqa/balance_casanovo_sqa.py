#!/usr/bin/env python3
"""Create deterministic class-balanced Parquet splits for the SQA benchmark.

The input dataset is never modified. Every example from the minority class is
retained; the majority class is randomly undersampled to the same count using
a reproducible seed. The output manifest records the exact selection policy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SPLITS = ("train", "val", "test")
DEFAULT_INPUT = Path(
    "/path/to/data/downstream_data_casanovo/spectrum_quality/"
    "casanovo_foundation_sqa_v2"
)
DEFAULT_OUTPUT = Path(
    "/path/to/data/downstream_data_casanovo/spectrum_quality/"
    "casanovo_foundation_sqa_v2_balanced"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def balanced_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.asarray(labels)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Expected binary labels 0/1, found {np.unique(labels).tolist()}")
    positions = {
        label: np.flatnonzero(labels == label).astype(np.int64).tolist()
        for label in (0, 1)
    }
    counts = {str(label): len(values) for label, values in positions.items()}
    if not positions[0] or not positions[1]:
        raise ValueError(f"Both classes are required; counts={counts}")
    target = min(len(positions[0]), len(positions[1]))
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        values = positions[label]
        selected.extend(values if len(values) == target else rng.sample(values, target))
    return np.sort(np.asarray(selected, dtype=np.int64)), {
        "input_negative": len(positions[0]),
        "input_positive": len(positions[1]),
        "output_negative": target,
        "output_positive": target,
        "removed_negative": len(positions[0]) - target,
        "removed_positive": len(positions[1]) - target,
    }


def read_labels_and_charges(
    path: Path, batch_size: int, split: str
) -> tuple[np.ndarray, np.ndarray]:
    parquet = pq.ParquetFile(path)
    label_chunks = []
    charge_chunks = []
    for batch in tqdm(
        parquet.iter_batches(
            columns=["label", "precursor_charge"], batch_size=batch_size
        ),
        desc=f"Scanning {split} labels",
        unit="batch",
    ):
        label_chunks.append(batch.column(0).to_numpy(zero_copy_only=False))
        charge_chunks.append(batch.column(1).to_numpy(zero_copy_only=False))
    if not label_chunks:
        raise ValueError(f"Empty Parquet split: {path}")
    return np.concatenate(label_chunks), np.concatenate(charge_chunks)


def supported_charge_indices(
    charges: np.ndarray, min_charge: int, max_charge: int
) -> np.ndarray:
    charges = np.asarray(charges, dtype=np.float64)
    valid = np.isfinite(charges) & (charges == np.floor(charges))
    valid &= (charges >= min_charge) & (charges <= max_charge)
    return np.flatnonzero(valid).astype(np.int64)


def write_balanced_split(
    input_path: Path,
    output_path: Path,
    selected: np.ndarray,
    batch_size: int,
    split: str,
) -> None:
    parquet = pq.ParquetFile(input_path)
    schema = parquet.schema_arrow
    selected_set = set(selected.tolist())
    offset = 0
    writer = None
    try:
        for batch in tqdm(
            parquet.iter_batches(batch_size=batch_size),
            desc=f"Writing balanced {split}",
            unit="batch",
        ):
            end = offset + batch.num_rows
            local = [index - offset for index in selected_set if offset <= index < end]
            if local:
                local.sort()
                local_set = set(local)
                mask = pa.array([index in local_set for index in range(batch.num_rows)])
                table = pa.Table.from_batches([batch]).filter(mask)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                writer.write_table(table)
            offset = end
    finally:
        if writer is not None:
            writer.close()
    if offset != parquet.metadata.num_rows:
        raise RuntimeError(f"Read {offset} rows but expected {parquet.metadata.num_rows}: {input_path}")
    if not output_path.exists():
        raise ValueError(f"No rows selected for {split}: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--min-charge", type=int, default=1)
    parser.add_argument("--max-charge", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_charge < 1 or args.max_charge < args.min_charge:
        raise ValueError("Require 1 <= --min-charge <= --max-charge")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "name": "casanovo_foundation_sqa_balanced",
        "created_at_utc": utc_now(),
        "input_root": str(args.input_root.resolve()),
        "seed": args.seed,
        "charge_filter": {
            "min_charge": args.min_charge,
            "max_charge": args.max_charge,
            "policy": "retain only finite integer precursor charges in the inclusive range",
        },
        "selection_policy": (
            "retain every example from the minority class and randomly undersample "
            "the majority class to the same count per split"
        ),
        "splits": {},
    }
    for split_index, split in enumerate(SPLITS):
        input_path = args.input_root / f"{split}.parquet"
        output_path = args.output_root / f"{split}.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split: {input_path}")
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        labels, charges = read_labels_and_charges(input_path, args.batch_size, split)
        eligible = supported_charge_indices(charges, args.min_charge, args.max_charge)
        if not len(eligible):
            raise ValueError(f"No supported-charge rows in {input_path}")
        selected_local, summary = balanced_indices(
            labels[eligible], args.seed + split_index
        )
        selected = eligible[selected_local]
        summary.update(
            {
                "input_rows": int(len(labels)),
                "eligible_rows": int(len(eligible)),
                "discarded_unsupported_charge": int(len(labels) - len(eligible)),
            }
        )
        temporary = args.output_root / f".{split}.parquet.incomplete"
        temporary.unlink(missing_ok=True)
        write_balanced_split(input_path, temporary, selected, args.batch_size, split)
        table = pq.read_table(temporary, columns=["label"])
        actual = Counter(int(value) for value in table["label"].to_pylist())
        if actual[0] != actual[1] or actual[0] != summary["output_negative"]:
            raise ValueError(f"Balanced output validation failed for {split}: {actual}")
        os.replace(temporary, output_path)
        manifest["splits"][split] = {
            **summary,
            "output_rows": int(len(selected)),
            "seed": args.seed + split_index,
            "input_path": str(input_path.resolve()),
            "output_path": str(output_path.resolve()),
        }
        atomic_json_write(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote balanced SQA dataset: {args.output_root}")


if __name__ == "__main__":
    main()
