#!/usr/bin/env python
"""Materialize run-disjoint Kingdoms retrieval and pair-evaluation benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_ninespecies_v2_embedding_eval import _stable_rank
from scripts.create_ninespecies_v2_pair_benchmark import (
    SpectrumRecord,
    _pair_records_for_protocol,
    _positive_pairs,
    _selected_groups,
)


SAMPLING_CODE_VERSION = "kingdoms_run_disjoint_paper_benchmarks_v1"
ACQUISITION_PATTERN = re.compile(r"^(\d{8}_QX\d+)")
VALIDATION_SPECIES = {
    "Arabidopsis_thaliana_Callus",
    "Arabidopsis_thaliana_Root",
    "Canis_lupus",
    "Chlamydomonas_reinhardtii",
    "Dictyostelium_discoideum",
    "Glycine_max",
    "Mus_musculus",
    "Sacharomyces_cerevisiae",
    "Thalassiosira_pseudonana",
    "Vitis_vinifera",
}
SOURCE_COLUMNS = [
    "modified_sequence",
    "precursor_charge",
    "precursor_mass",
    "raw_file",
    "Qvalue",
    "chimeric",
    "mz_array",
    "intensity_array",
]


def _metadata_fingerprint(path: Path) -> str:
    """Hash Parquet footer metadata as a stable source-version fingerprint."""
    metadata = pq.ParquetFile(path).metadata.to_dict()
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()


def _acquisition_id(raw_file: str) -> str:
    match = ACQUISITION_PATTERN.match(raw_file)
    return match.group(1) if match else raw_file


def _validation_acquisitions(species: str, raw_files: set[str], seed: int) -> set[str]:
    """Choose one whole filename-derived acquisition batch for validation."""
    if species not in VALIDATION_SPECIES:
        return set()
    groups: dict[str, set[str]] = defaultdict(set)
    for raw_file in raw_files:
        groups[_acquisition_id(raw_file)].add(raw_file)
    if len(groups) < 2:
        raise ValueError(f"{species} no longer has two acquisition batches.")
    selected = min(groups, key=lambda group: _stable_rank(seed, "validation-acquisition", species, group))
    return groups[selected]


def _valid_indices(table: pa.Table, *, validation_raw_files: set[str], split: str) -> list[int]:
    indices = []
    for index, (sequence, charge, mz, raw_file, qvalue, chimeric) in enumerate(
        zip(
            table["modified_sequence"].to_pylist(),
            table["precursor_charge"].to_pylist(),
            table["precursor_mass"].to_pylist(),
            table["raw_file"].to_pylist(),
            table["Qvalue"].to_pylist(),
            table["chimeric"].to_pylist(),
        )
    ):
        if not sequence or charge is None or mz is None or raw_file is None or qvalue is None:
            continue
        if bool(chimeric) or float(qvalue) > 0.01 or int(charge) < 1 or float(mz) <= 0.0:
            continue
        is_validation = str(raw_file) in validation_raw_files
        if (split == "validation" and is_validation) or (split == "test" and not is_validation):
            indices.append(index)
    return indices


def _selected_peptide_rows(
    table: pa.Table, indices: list[int], *, species: str, seed: int, max_groups: int
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    labels = table["modified_sequence"].to_pylist()
    for index in indices:
        groups[str(labels[index])].append(index)
    selected = {}
    for peptide in sorted(groups, key=lambda value: _stable_rank(seed, species, "retrieval", value)):
        rows = groups[peptide]
        if len(rows) < 2:
            continue
        selected[peptide] = sorted(rows, key=lambda value: _stable_rank(seed, species, peptide, value))[:10]
        if len(selected) >= max_groups:
            break
    return selected


def _retrieval_table(table: pa.Table, *, species: str, selected: dict[str, list[int]]) -> pa.Table:
    indices = [index for peptide in sorted(selected) for index in selected[peptide]]
    source = table.take(pa.array(indices, type=pa.int64()))
    peptide_ids = source["modified_sequence"].to_pylist()
    return pa.table(
        {
            "species": pa.array([species] * len(indices)),
            "peptide_id": pa.array(peptide_ids),
            "source_row_index": pa.array(indices, type=pa.int64()),
            "selected_group_size": pa.array([len(selected[peptide]) for peptide in peptide_ids], type=pa.int16()),
            # Kingdoms' historical field name is misleading: values are observed m/z.
            "precursor_mz": source["precursor_mass"],
            "precursor_charge": source["precursor_charge"],
            "mz_array": source["mz_array"],
            "intensity_array": source["intensity_array"],
        }
    )


def _pair_records(table: pa.Table, indices: list[int], species: str) -> list[SpectrumRecord]:
    sequences = table["modified_sequence"].to_pylist()
    charges = table["precursor_charge"].to_pylist()
    mz_values = table["precursor_mass"].to_pylist()
    return [
        SpectrumRecord(
            row_index=index,
            species=species,
            peptide_ion_id=f"{sequences[index]}|z={int(charges[index])}",
            precursor_mz=float(mz_values[index]),
            precursor_charge=int(charges[index]),
        )
        for index in indices
    ]


def _spectra_table(table: pa.Table, records: dict[str, SpectrumRecord], requested: set[str]) -> pa.Table:
    selected = sorted((records[key] for key in requested), key=lambda record: record.row_index)
    source = table.take(pa.array([record.row_index for record in selected], type=pa.int64()))
    return pa.table(
        {
            "spectrum_id": pa.array([record.spectrum_id for record in selected]),
            "species": pa.array([record.species for record in selected]),
            "peptide_ion_id": pa.array([record.peptide_ion_id for record in selected]),
            "modified_sequence": source["modified_sequence"],
            "source_row_index": pa.array([record.row_index for record in selected], type=pa.int64()),
            "precursor_mz": source["precursor_mass"],
            "precursor_charge": source["precursor_charge"],
            "mz_array": source["mz_array"],
            "intensity_array": source["intensity_array"],
        }
    )


def _materialize_split(args: argparse.Namespace, *, split: str, source_paths: list[Path]) -> dict[str, object]:
    output_root = Path(args.output_root) / split
    retrieval_path = output_root / "retrieval.parquet"
    pair_dir = output_root / "pair_discrimination"
    output_paths = [retrieval_path, output_root / "retrieval.manifest.json", pair_dir / "spectra.parquet", pair_dir / "pairs.parquet", pair_dir / "manifest.json"]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists ({existing[0]}). Pass --overwrite to replace it.")
    if args.overwrite:
        for path in existing:
            path.unlink()
    pair_dir.mkdir(parents=True, exist_ok=True)

    retrieval_writer = spectra_writer = pairs_writer = None
    species_summary = {}
    pair_sets: dict[str, dict[str, object]] = defaultdict(dict)
    try:
        for path in tqdm(source_paths, desc=f"Kingdoms {split}", unit="species"):
            species = path.stem
            table = pq.read_table(path, columns=SOURCE_COLUMNS)
            raw_files = set(str(raw) for raw in table["raw_file"].to_pylist() if raw)
            validation_raw_files = _validation_acquisitions(species, raw_files, args.seed)
            indices = _valid_indices(table, validation_raw_files=validation_raw_files, split=split)
            max_retrieval = args.validation_max_peptide_groups if split == "validation" else args.test_max_peptide_groups
            max_pair_groups = args.validation_max_peptide_ion_groups if split == "validation" else args.test_max_peptide_ion_groups
            max_pairs = args.validation_max_pairs_per_species if split == "validation" else args.test_max_pairs_per_species
            selected = _selected_peptide_rows(table, indices, species=species, seed=args.seed, max_groups=max_retrieval)
            if split == "validation" and not selected:
                continue
            if split == "test" and not selected:
                raise ValueError(f"{species} has no repeated quality-filtered peptides in test.")
            retrieval = _retrieval_table(table, species=species, selected=selected)
            if retrieval_writer is None:
                retrieval_writer = pq.ParquetWriter(retrieval_path, retrieval.schema)
            retrieval_writer.write_table(retrieval)

            records = _pair_records(table, indices, species)
            selected_groups = _selected_groups(records, seed=args.seed, max_peptide_ion_groups=max_pair_groups, max_spectra_per_peptide_ion=args.max_spectra_per_group)
            positives = _positive_pairs(selected_groups, seed=args.seed, max_positive_pairs_per_group=args.max_positive_pairs_per_group)
            by_charge: dict[int, list[SpectrumRecord]] = defaultdict(list)
            for record in records:
                by_charge[record.precursor_charge].append(record)
            mz_by_charge = {}
            for charge, candidates in by_charge.items():
                candidates.sort(key=lambda record: (record.precursor_mz, record.row_index))
                mz_by_charge[charge] = [record.precursor_mz for record in candidates]
            pair_rows = []
            for pair_set in ("all_random", "same_charge_random", "same_charge_10ppm"):
                rows, summary = _pair_records_for_protocol(
                    positives, pair_set=pair_set, all_candidates=records,
                    candidates_by_charge=by_charge, mz_by_charge=mz_by_charge,
                    seed=args.seed, tolerance_ppm=args.precursor_tolerance_ppm,
                    max_pairs_per_species=max_pairs,
                )
                pair_sets[pair_set][species] = summary
                pair_rows.extend(rows)
            if pair_rows:
                pairs = pa.Table.from_pylist(pair_rows)
                if pairs_writer is None:
                    pairs_writer = pq.ParquetWriter(pair_dir / "pairs.parquet", pairs.schema)
                pairs_writer.write_table(pairs)
                requested = {str(row[column]) for row in pair_rows for column in ("left_spectrum_id", "right_spectrum_id")}
                spectra = _spectra_table(table, {record.spectrum_id: record for record in records}, requested)
                if spectra_writer is None:
                    spectra_writer = pq.ParquetWriter(pair_dir / "spectra.parquet", spectra.schema)
                spectra_writer.write_table(spectra)
            species_summary[species] = {
                "source_rows": table.num_rows,
                "quality_split_rows": len(indices),
                "source_metadata_sha256": _metadata_fingerprint(path),
                "validation_raw_files": sorted(validation_raw_files),
                "selected_retrieval_peptide_groups": len(selected),
                "selected_pair_peptide_ion_groups": len(selected_groups),
            }
    finally:
        for writer in (retrieval_writer, spectra_writer, pairs_writer):
            if writer is not None:
                writer.close()

    retrieval_manifest = {
        "sampling_code_version": SAMPLING_CODE_VERSION, "seed": args.seed, "split": split,
        "source_root": args.source_root, "quality_gate": "Qvalue <= 0.01 and chimeric == false",
        "canonical_precursor_mz_source_column": "precursor_mass",
        "selection": {"max_peptide_groups_per_species": args.validation_max_peptide_groups if split == "validation" else args.test_max_peptide_groups, "max_spectra_per_peptide": args.max_spectra_per_group},
        "species": species_summary,
    }
    (output_root / "retrieval.manifest.json").write_text(json.dumps(retrieval_manifest, indent=2, sort_keys=True) + "\n")
    pair_manifest = {
        "sampling_code_version": SAMPLING_CODE_VERSION, "seed": args.seed, "split": split,
        "quality_gate": "Qvalue <= 0.01 and chimeric == false", "pair_sets": pair_sets,
        "construction": {"group_id": "modified_sequence + precursor_charge", "negative_to_positive_ratio": 1, "same_charge_required": True, "hard_negative_precursor_tolerance_ppm": args.precursor_tolerance_ppm},
        "species": species_summary,
    }
    (pair_dir / "manifest.json").write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")
    return {"retrieval": retrieval_manifest, "pair_discrimination": pair_manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/path/to/data/kingdoms/processed")
    parser.add_argument("--output-root", default="/path/to/data/probing_datasets/kingdoms_paper_benchmarks")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--validation-max-peptide-groups", type=int, default=500)
    parser.add_argument("--test-max-peptide-groups", type=int, default=5000)
    parser.add_argument("--validation-max-peptide-ion-groups", type=int, default=500)
    parser.add_argument("--test-max-peptide-ion-groups", type=int, default=5000)
    parser.add_argument("--max-spectra-per-group", type=int, default=10)
    parser.add_argument("--max-positive-pairs-per-group", type=int, default=20)
    parser.add_argument("--validation-max-pairs-per-species", type=int, default=5000)
    parser.add_argument("--test-max-pairs-per-species", type=int, default=50000)
    parser.add_argument("--precursor-tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = sorted(Path(args.source_root).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet inputs found under {args.source_root}")
    print(json.dumps({split: _materialize_split(args, split=split, source_paths=paths) for split in ("validation", "test")}, indent=2))
