#!/usr/bin/env python3
"""Materialize a charge-restricted de novo test artifact without changing its source.

This is intentionally a test-only derivative. It preserves every source column,
row order among retained spectra, and original peptide notation. The canonical
MSKB-final corpus and its official test Lance directory remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import lance
import pyarrow as pa
from tqdm import tqdm


SCRIPT_VERSION = "denovo_charge_restricted_test_v1"
DEFAULT_SOURCE = Path(
    "/path/to/data/denovo_mskb_final/"
    "lance_peptidoform_val10k_seed42/test.lance"
)
DEFAULT_OUTPUT = Path(
    "/path/to/data/denovo_mskb_final/"
    "lance_peptidoform_val10k_seed42/"
    "test_charge_lt5_for_casanovo_v_gt_5_0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lance", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-charge", type=int, default=1)
    parser.add_argument("--max-charge-exclusive", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batches(dataset: lance.LanceDataset, batch_size: int) -> Iterator[pa.RecordBatch]:
    yield from tqdm(
        dataset.to_batches(batch_size=batch_size),
        desc="Filter MSKB-final test by precursor charge",
        unit="batch",
        mininterval=1.0,
    )


def main() -> None:
    args = parse_args()
    if args.min_charge < 1:
        raise ValueError("--min-charge must be at least 1")
    if args.max_charge_exclusive <= args.min_charge:
        raise ValueError("--max-charge-exclusive must exceed --min-charge")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    source = args.source_lance.resolve()
    output = args.output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing derivative: {output}. "
            "Choose a new --output-root after inspecting its manifest."
        )

    dataset = lance.dataset(str(source))
    charge_field = "precursor_charge"
    if charge_field not in dataset.schema.names:
        raise ValueError(f"{source} has no {charge_field!r} field")

    destination = output / "test.lance"
    output.mkdir(parents=True)
    source_histogram: Counter[int | None] = Counter()
    retained_histogram: Counter[int] = Counter()
    source_rows = 0
    retained_rows = 0
    created = False

    for batch in batches(dataset, args.batch_size):
        charges = batch.column(batch.schema.get_field_index(charge_field)).to_pylist()
        source_rows += len(charges)
        source_histogram.update(charges)
        indices = [
            index
            for index, value in enumerate(charges)
            if value is not None
            and args.min_charge <= int(value) < args.max_charge_exclusive
        ]
        if not indices:
            continue
        retained_charges = [int(charges[index]) for index in indices]
        retained_histogram.update(retained_charges)
        table = pa.Table.from_batches(
            [batch.take(pa.array(indices, type=pa.int64()))]
        )
        lance.write_dataset(
            table,
            str(destination),
            mode="append" if created else "create",
            max_rows_per_file=100_000,
            max_rows_per_group=4096,
        )
        created = True
        retained_rows += table.num_rows

    if not created:
        raise RuntimeError("No source rows passed the requested charge filter")

    retained_dataset = lance.dataset(str(destination))
    if retained_dataset.count_rows() != retained_rows:
        raise RuntimeError("Written row count does not match retained source rows")
    for batch in retained_dataset.to_batches(columns=[charge_field], batch_size=args.batch_size):
        if any(
            value is None
            or int(value) < args.min_charge
            or int(value) >= args.max_charge_exclusive
            for value in batch.column(0).to_pylist()
        ):
            raise RuntimeError("Charge audit found an out-of-range retained row")

    corpus_root = source.parent
    source_manifest = corpus_root / "manifest.json"
    manifest = {
        "artifact": "denovo_charge_restricted_test",
        "script_version": SCRIPT_VERSION,
        "label": "charge <5 for Casanovo v>5.0",
        "purpose": (
            "Shared de novo evaluation subset restricted to the precursor-charge "
            "domain supported by Casanovo v5-series releases."
        ),
        "source_lance": str(source),
        "source_corpus_manifest": str(source_manifest),
        "source_corpus_manifest_sha256": (
            sha256_file(source_manifest) if source_manifest.is_file() else None
        ),
        "charge_field": charge_field,
        "charge_filter": {
            "minimum_inclusive": args.min_charge,
            "maximum_exclusive": args.max_charge_exclusive,
        },
        "counts": {
            "source_rows": source_rows,
            "retained_rows": retained_rows,
            "excluded_rows": source_rows - retained_rows,
            "retained_fraction": retained_rows / source_rows,
            "source_charge_histogram": {
                str(key): value for key, value in sorted(
                    source_histogram.items(), key=lambda item: (-1 if item[0] is None else item[0])
                )
            },
            "retained_charge_histogram": {
                str(key): value for key, value in sorted(retained_histogram.items())
            },
        },
        "integrity": {
            "source_unchanged": (
                "This artifact reads the canonical test.lance and never modifies it."
            ),
            "retained_row_order": "Preserved relative to the source Lance scan order.",
            "all_source_columns_retained": True,
            "source_sequence_notation_retained": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote charge-restricted de novo test artifact: {output}")


if __name__ == "__main__":
    main()
