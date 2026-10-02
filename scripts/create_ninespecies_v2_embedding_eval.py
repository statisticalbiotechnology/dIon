"""Create a balanced peptide-retrieval corpus from nine-species V2 Lance data.

The generated Parquet has canonical ``precursor_mz`` semantics. This matches
``PEPTIDE_DATASET_SPECS['ninespecies_v2']``: the Lance field named
``precursor_mass`` contains the observed precursor m/z for this dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_COLUMNS = [
    "modified_sequence",
    "precursor_mass",
    "precursor_charge",
    "mz_array",
    "intensity_array",
]


def _stable_rank(seed: int, *parts: object) -> bytes:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _select_indices(
    dataset: lance.LanceDataset,
    species: str,
    seed: int,
    max_peptides: int,
    max_spectra_per_peptide: int,
) -> tuple[dict[str, list[int]], int]:
    """Return stable selected row indices for repeated peptide groups."""
    group_rows: dict[str, list[int]] = defaultdict(list)
    labels = dataset.to_table(columns=["modified_sequence"])[
        "modified_sequence"
    ].to_pylist()
    for row_index, peptide_id in enumerate(labels):
        if peptide_id:
            group_rows[peptide_id].append(row_index)

    repeated_groups = {
        peptide_id: indices
        for peptide_id, indices in group_rows.items()
        if len(indices) >= 2
    }
    selected_peptides = sorted(
        repeated_groups,
        key=lambda peptide_id: _stable_rank(seed, species, peptide_id),
    )[:max_peptides]
    selected_rows = {
        peptide_id: sorted(
            repeated_groups[peptide_id],
            key=lambda row_index: _stable_rank(seed, species, peptide_id, row_index),
        )[:max_spectra_per_peptide]
        for peptide_id in selected_peptides
    }
    return selected_rows, len(repeated_groups)


def _selected_table(
    dataset: lance.LanceDataset,
    species: str,
    selected_rows: dict[str, list[int]],
) -> pa.Table:
    """Materialize selected rows with evaluation-specific metadata."""
    row_indices = [
        row_index
        for peptide_id in sorted(selected_rows)
        for row_index in selected_rows[peptide_id]
    ]
    source = dataset.take(row_indices, columns=SOURCE_COLUMNS)
    peptide_ids = source["modified_sequence"].to_pylist()
    source_group_sizes = pa.array(
        [len(selected_rows[peptide_id]) for peptide_id in peptide_ids],
        type=pa.int16(),
    )
    return pa.table(
        {
            "species": pa.array([species] * len(row_indices)),
            "peptide_id": pa.array(peptide_ids),
            "source_row_index": pa.array(row_indices, type=pa.int64()),
            "selected_group_size": source_group_sizes,
            # The ninespecies_v2 Lance export misnames observed precursor m/z.
            "precursor_mz": source["precursor_mass"],
            "precursor_charge": source["precursor_charge"],
            "mz_array": source["mz_array"],
            "intensity_array": source["intensity_array"],
        }
    )


def create_corpus(args: argparse.Namespace) -> dict[str, object]:
    """Create one reproducible, species-balanced evaluation Parquet."""
    source_root = Path(args.source_root)
    source_paths = sorted(source_root.glob("*.lance"))
    if not source_paths:
        raise FileNotFoundError(f"No Lance datasets found in {source_root}")
    if args.max_peptides < 1 or args.max_spectra_per_peptide < 2:
        raise ValueError("Use at least one peptide and two spectra per peptide.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Pass --overwrite to replace it."
        )

    writer = None
    species_summary = {}
    try:
        for source_path in source_paths:
            species = source_path.stem
            dataset = lance.dataset(str(source_path))
            selected_rows, repeated_group_count = _select_indices(
                dataset=dataset,
                species=species,
                seed=args.seed,
                max_peptides=args.max_peptides,
                max_spectra_per_peptide=args.max_spectra_per_peptide,
            )
            table = _selected_table(dataset, species, selected_rows)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            species_summary[species] = {
                "source_spectra": dataset.count_rows(),
                "source_repeated_peptides": repeated_group_count,
                "selected_peptides": len(selected_rows),
                "selected_spectra": table.num_rows,
            }
    finally:
        if writer is not None:
            writer.close()

    report = {
        "source_root": str(source_root),
        "output_path": str(output_path),
        "seed": args.seed,
        "max_peptides_per_species": args.max_peptides,
        "max_spectra_per_peptide": args.max_spectra_per_peptide,
        "species": species_summary,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="/path/to/data/9_species_V2/lance_species",
    )
    parser.add_argument(
        "--output-path",
        default=(
            "/path/to/data/probing_datasets/"
            "ninespecies_v2_peptide_retrieval.parquet"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-peptides", type=int, default=1000)
    parser.add_argument("--max-spectra-per-peptide", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = create_corpus(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
