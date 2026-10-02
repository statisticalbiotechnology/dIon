"""Create a deterministic, organism-diverse peptide retrieval corpus.

The processed kingdoms Parquets use the field name ``precursor_mass`` for
observed precursor m/z, as confirmed by its peptide/charge-scale values. This
script writes it as canonical ``precursor_mz`` for dIon evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


REQUIRED_COLUMNS = {
    "modified_sequence",
    "precursor_mass",
    "precursor_charge",
    "mz_array",
    "intensity_array",
}


def stable_key(seed: int, *parts: object) -> bytes:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).digest()


def select_rows(
    table: pa.Table, *, species: str, seed: int, groups_per_species: int, max_spectra: int
) -> list[int]:
    sequences = table["modified_sequence"].to_pylist()
    charges = table["precursor_charge"].to_pylist()
    precursor_mz = table["precursor_mass"].to_pylist()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (sequence, charge, mz) in enumerate(zip(sequences, charges, precursor_mz, strict=True)):
        if not sequence or charge is None or mz is None:
            continue
        try:
            if int(charge) < 1 or float(mz) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        groups[str(sequence)].append(index)
    repeated = [sequence for sequence, indices in groups.items() if len(indices) >= 2]
    selected_groups = sorted(
        repeated, key=lambda sequence: stable_key(seed, "group", species, sequence)
    )[:groups_per_species]
    selected = []
    for sequence in selected_groups:
        selected.extend(
            sorted(
                groups[sequence],
                key=lambda index: stable_key(seed, "spectrum", species, sequence, index),
            )[:max_spectra]
        )
    return selected


def build(args: argparse.Namespace) -> dict[str, object]:
    source_root = Path(args.source_root)
    output_path = Path(args.output_path)
    paths = sorted(source_root.glob("*.parquet"))
    if args.species:
        requested = set(args.species)
        paths = [path for path in paths if path.stem in requested]
        missing = sorted(requested - {path.stem for path in paths})
        if missing:
            raise FileNotFoundError(f"Requested kingdoms species not found: {missing}")
    if not paths:
        raise FileNotFoundError(f"No Parquet files found in {source_root}")
    selected_by_species = {}
    eligible_species = []
    for path in tqdm(paths, desc="Selecting kingdoms species", unit="species", dynamic_ncols=True):
        schema = pq.ParquetFile(path).schema_arrow
        missing = REQUIRED_COLUMNS - set(schema.names)
        if missing:
            continue
        table = pq.read_table(path, columns=["modified_sequence", "precursor_mass", "precursor_charge"])
        selected = select_rows(
            table,
            species=path.stem,
            seed=args.seed,
            groups_per_species=args.groups_per_species,
            max_spectra=args.max_spectra_per_peptide,
        )
        if len(selected) >= 2 * args.min_groups_per_species:
            eligible_species.append(path)
            selected_by_species[path] = selected
    ordered_species = sorted(
        eligible_species, key=lambda path: stable_key(args.seed, "species", path.stem)
    )[:args.max_species]
    if len(ordered_species) < args.max_species:
        raise ValueError(
            f"Only {len(ordered_species)} species meet the selection requirement; "
            f"requested {args.max_species}."
        )
    rows = []
    selection = {}
    for path in tqdm(ordered_species, desc="Writing kingdoms spectra", unit="species", dynamic_ncols=True):
        species = path.stem
        table = pq.read_table(path, columns=sorted(REQUIRED_COLUMNS))
        indices = selected_by_species[path]
        selected = table.take(pa.array(indices, type=pa.int64()))
        for source_index, row in zip(indices, selected.to_pylist(), strict=True):
            sequence = str(row["modified_sequence"])
            charge = int(row["precursor_charge"])
            rows.append(
                {
                    "species": species,
                    "peptide_id": sequence,
                    "peptide_ion_id": f"{sequence}|z={charge}",
                    "source_row_index": source_index,
                    "precursor_mz": float(row["precursor_mass"]),
                    "precursor_charge": charge,
                    "mz_array": row["mz_array"],
                    "intensity_array": row["intensity_array"],
                }
            )
        selection[species] = {"selected_spectra": len(indices)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="zstd")
    manifest = {
        "source_root": str(source_root),
        "output_path": str(output_path),
        "seed": args.seed,
        "max_species": args.max_species,
        "groups_per_species": args.groups_per_species,
        "max_spectra_per_peptide": args.max_spectra_per_peptide,
        "peptide_id": "modified_sequence",
        "peptide_ion_id": "modified_sequence plus charge",
        "precursor_mz_source": "processed precursor_mass column",
        "selection": selection,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="/path/to/data/kingdoms/processed",
    )
    parser.add_argument(
        "--output-path",
        default="/path/to/data/probing_datasets/kingdoms_peptide_retrieval.parquet",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-species", type=int, default=48)
    parser.add_argument("--groups-per-species", type=int, default=188)
    parser.add_argument("--min-groups-per-species", type=int, default=188)
    parser.add_argument("--max-spectra-per-peptide", type=int, default=10)
    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="Optional explicit processed-Parquet stems for a targeted benchmark.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
