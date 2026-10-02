#!/usr/bin/env python
"""Materialize an immutable, deterministic per-species cap of a Kingdoms Lance set."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path

import lance
import pyarrow as pa

IDENTITY_COLUMNS = ("species", "peak_file", "scan_id", "source_row_index")


def stable_rank(row: dict[str, object], seed: int) -> int:
    identity = "\x1f".join(str(row[column]) for column in IDENTITY_COLUMNS)
    return int.from_bytes(
        hashlib.blake2b(f"{seed}\x1e{identity}".encode("utf-8"), digest_size=16).digest(), "big"
    )


def rank_thresholds(dataset, cap: int, seed: int) -> tuple[dict[str, int], Counter]:
    heaps: dict[str, list[int]] = {}
    counts: Counter = Counter()
    for batch in dataset.scanner(columns=list(IDENTITY_COLUMNS), batch_size=16_384).to_batches():
        for row in batch.to_pylist():
            species = str(row["species"])
            counts[species] += 1
            rank = stable_rank(row, seed)
            heap = heaps.setdefault(species, [])
            if len(heap) < cap:
                heapq.heappush(heap, -rank)
            elif rank < -heap[0]:
                heapq.heapreplace(heap, -rank)
    return {species: -heap[0] for species, heap in heaps.items()}, counts


def selected_batches(dataset, thresholds: dict[str, int], seed: int):
    for batch in dataset.scanner(batch_size=4_096).to_batches():
        identities = {
            column: batch.column(batch.schema.get_field_index(column)).to_pylist()
            for column in IDENTITY_COLUMNS
        }
        indices = [
            index
            for index in range(batch.num_rows)
            if stable_rank(
                {column: identities[column][index] for column in IDENTITY_COLUMNS}, seed
            ) <= thresholds[str(identities["species"][index])]
        ]
        if indices:
            yield batch.take(pa.array(indices, type=pa.int64()))


def write_species_datasets(dataset, destination: Path, expected_counts: Counter) -> dict[str, str]:
    """Write all one-species datasets in one pass over the capped table."""
    destination.mkdir()
    created: set[str] = set()
    written_counts: Counter = Counter()
    for batch in dataset.scanner(batch_size=4_096).to_batches():
        species_values = batch.column(batch.schema.get_field_index("species")).to_pylist()
        indices_by_species: dict[str, list[int]] = {}
        for index, species in enumerate(species_values):
            indices_by_species.setdefault(str(species), []).append(index)
        for species, indices in indices_by_species.items():
            lance.write_dataset(
                batch.take(pa.array(indices, type=pa.int64())),
                str(destination / f"{species}.lance"),
                schema=dataset.schema,
                mode="append" if species in created else "create",
                max_rows_per_file=100_000,
                max_rows_per_group=4_096,
            )
            created.add(species)
            written_counts[species] += len(indices)
    if dict(written_counts) != dict(expected_counts):
        raise RuntimeError("Per-species output row counts differ from the capped test table.")
    return {
        species: str(Path("test_by_species") / f"{species}.lance")
        for species in sorted(written_counts)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-species-cap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--copy-validation", action="store_true")
    parser.add_argument(
        "--write-species-datasets", action="store_true",
        help="Also write test_by_species/<species>.lance datasets from the capped test set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_species_cap < 1:
        raise ValueError("--per-species-cap must be positive.")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_test = source_root / "test.lance"
    source_validation = source_root / "validation.lance"
    source_manifest = source_root / "manifest.json"
    if not source_test.is_dir() or not source_validation.is_dir() or not source_manifest.is_file():
        raise FileNotFoundError("Expected test.lance, validation.lance, and manifest.json under --source-root.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")

    test_dataset = lance.dataset(str(source_test))
    thresholds, source_counts = rank_thresholds(test_dataset, args.per_species_cap, args.seed)
    expected_counts = {species: min(count, args.per_species_cap) for species, count in source_counts.items()}
    staging_root = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True)
    try:
        lance.write_dataset(
            selected_batches(test_dataset, thresholds, args.seed),
            str(staging_root / "test.lance"), schema=test_dataset.schema, mode="create",
        )
        if args.copy_validation:
            shutil.copytree(source_validation, staging_root / "validation.lance")
        retained_dataset = lance.dataset(str(staging_root / "test.lance"))
        retained_counts = Counter(
            row["species"]
            for batch in retained_dataset.scanner(columns=["species"], batch_size=16_384).to_batches()
            for row in batch.to_pylist()
        )
        if dict(retained_counts) != expected_counts:
            raise RuntimeError("Retained per-species counts differ from the requested cap; output was not published.")
        species_test_datasets = {}
        if args.write_species_datasets:
            species_test_datasets = write_species_datasets(
                retained_dataset, staging_root / "test_by_species", retained_counts
            )
        source_metadata = json.loads(source_manifest.read_text())
        output_manifest = {
            "dataset": "denovo_kingdoms_species_capped",
            "source_root": str(source_root),
            "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "sampling": {
                "method": "lowest stable BLAKE2b-128 rank per species",
                "identity_columns": list(IDENTITY_COLUMNS),
                "seed": args.seed,
                "per_species_cap": args.per_species_cap,
            },
            "source_test_rows": test_dataset.count_rows(),
            "retained_test_rows": retained_dataset.count_rows(),
            "source_test_rows_by_species": dict(sorted(source_counts.items())),
            "retained_test_rows_by_species": dict(sorted(retained_counts.items())),
            "validation_copied": args.copy_validation,
            "species_test_datasets": species_test_datasets,
            "source_split_protocol": source_metadata.get("source_split_protocol"),
        }
        (staging_root / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
