#!/usr/bin/env python
"""Materialize large, probe-disjoint NineSpecies V2 paper-test benchmarks.

The existing ``ninespecies_v2_*`` artifacts remain development benchmarks.
This script creates separate static retrieval and pair-discrimination corpora,
excluding peptide backbones used by the historical end-amino-acid probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lance
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from scripts.create_ninespecies_v2_embedding_eval import (
    SOURCE_COLUMNS as RETRIEVAL_COLUMNS,
    _selected_table,
    _stable_rank,
)
from scripts.create_ninespecies_v2_pair_benchmark import (
    METADATA_COLUMNS,
    SPECTRUM_COLUMNS,
    SpectrumRecord,
    _hash_file,
    _materialize_spectra,
    _pair_records_for_protocol,
    _positive_pairs,
    _selected_groups,
)
from src.embed_eval.sequence_normalization import (
    BACKBONE_NORMALIZATION_VERSION,
    load_backbone_exclusion,
    normalize_peptide_backbone,
)


SAMPLING_CODE_VERSION = "ninespecies_v2_probe_disjoint_paper_test_v1"


def _eligible_retrieval_groups(
    dataset: lance.LanceDataset,
    *,
    species: str,
    seed: int,
    max_peptides: int,
    max_spectra_per_peptide: int,
    excluded_backbones: set[str],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    labels = dataset.to_table(columns=["modified_sequence"])["modified_sequence"].to_pylist()
    excluded_rows = 0
    for row_index, peptide_id in enumerate(labels):
        if not peptide_id:
            continue
        if normalize_peptide_backbone(peptide_id) in excluded_backbones:
            excluded_rows += 1
            continue
        groups[peptide_id].append(row_index)
    repeated = {peptide: rows for peptide, rows in groups.items() if len(rows) >= 2}
    peptide_ids = sorted(repeated, key=lambda peptide: _stable_rank(seed, species, peptide))
    selected = {}
    for peptide_id in peptide_ids[:max_peptides]:
        rows = sorted(
            repeated[peptide_id],
            key=lambda row_index: _stable_rank(seed, species, peptide_id, row_index),
        )[:max_spectra_per_peptide]
        if len(rows) >= 2:
            selected[peptide_id] = rows
    return selected, {
        "source_spectra": len(labels),
        "excluded_spectra": excluded_rows,
        "eligible_spectra": len(labels) - excluded_rows,
        "eligible_repeated_peptide_groups": len(repeated),
    }


def _eligible_pair_records(
    dataset: lance.LanceDataset,
    *,
    species: str,
    excluded_backbones: set[str],
) -> tuple[list[SpectrumRecord], str, int]:
    table = dataset.to_table(columns=METADATA_COLUMNS)
    metadata_hash = hashlib.sha256()
    records: list[SpectrumRecord] = []
    excluded_rows = 0
    for row_index, row in enumerate(table.to_pylist()):
        sequence = row["modified_sequence"]
        precursor_mz = row["precursor_mass"]
        charge = row["precursor_charge"]
        metadata_hash.update(
            f"{row_index}\x1f{sequence!s}\x1f{precursor_mz!r}\x1f{charge!r}\n".encode("utf-8")
        )
        if not sequence or precursor_mz is None or charge is None:
            continue
        if normalize_peptide_backbone(sequence) in excluded_backbones:
            excluded_rows += 1
            continue
        try:
            mz_value = float(precursor_mz)
            charge_value = int(charge)
        except (TypeError, ValueError):
            continue
        if mz_value <= 0.0 or charge_value < 1:
            continue
        records.append(
            SpectrumRecord(
                row_index=row_index,
                species=species,
                peptide_ion_id=f"{sequence}|z={charge_value}",
                precursor_mz=mz_value,
                precursor_charge=charge_value,
            )
        )
    return records, metadata_hash.hexdigest(), excluded_rows


def _create_retrieval(
    *,
    source_paths: list[Path],
    output_path: Path,
    seed: int,
    max_peptides: int,
    max_spectra_per_peptide: int,
    excluded_backbones: set[str],
) -> dict[str, object]:
    writer = None
    species_summary = {}
    try:
        for source_path in tqdm(source_paths, desc="Paper retrieval test", unit="species"):
            species = source_path.stem
            dataset = lance.dataset(str(source_path))
            selected, summary = _eligible_retrieval_groups(
                dataset,
                species=species,
                seed=seed,
                max_peptides=max_peptides,
                max_spectra_per_peptide=max_spectra_per_peptide,
                excluded_backbones=excluded_backbones,
            )
            if len(selected) < max_peptides:
                raise ValueError(
                    f"{species} has only {len(selected)} eligible repeated groups; "
                    f"requested {max_peptides}."
                )
            table = _selected_table(dataset, species, selected)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            species_summary[species] = {
                **summary,
                "selected_peptide_groups": len(selected),
                "selected_spectra": table.num_rows,
            }
    finally:
        if writer is not None:
            writer.close()
    return species_summary


def _create_pairs(
    *,
    source_paths: list[Path],
    output_dir: Path,
    seed: int,
    max_groups: int,
    max_spectra_per_group: int,
    max_positive_pairs_per_group: int,
    max_pairs_per_species: int,
    precursor_tolerance_ppm: float,
    excluded_backbones: set[str],
) -> tuple[dict[str, object], dict[str, dict[str, dict[str, int | float]]]]:
    spectra_path = output_dir / "spectra.parquet"
    pairs_path = output_dir / "pairs.parquet"
    spectra_writer = None
    pair_rows = []
    source_summary = {}
    pair_set_summary: dict[str, dict[str, dict[str, int | float]]] = defaultdict(dict)
    try:
        for source_path in tqdm(source_paths, desc="Paper pair test", unit="species"):
            species = source_path.stem
            dataset = lance.dataset(str(source_path))
            records, metadata_hash, excluded_rows = _eligible_pair_records(
                dataset, species=species, excluded_backbones=excluded_backbones
            )
            selected_groups = _selected_groups(
                records,
                seed=seed,
                max_peptide_ion_groups=max_groups,
                max_spectra_per_peptide_ion=max_spectra_per_group,
            )
            if len(selected_groups) < max_groups:
                raise ValueError(
                    f"{species} has only {len(selected_groups)} eligible peptide-ion groups; "
                    f"requested {max_groups}."
                )
            positives = _positive_pairs(
                selected_groups,
                seed=seed,
                max_positive_pairs_per_group=max_positive_pairs_per_group,
            )
            candidates_by_charge: dict[int, list[SpectrumRecord]] = defaultdict(list)
            for record in records:
                candidates_by_charge[record.precursor_charge].append(record)
            mz_by_charge = {}
            for charge, candidates in candidates_by_charge.items():
                candidates.sort(key=lambda record: (record.precursor_mz, record.row_index))
                mz_by_charge[charge] = [record.precursor_mz for record in candidates]

            species_rows = []
            for pair_set in ("all_random", "same_charge_random", "same_charge_10ppm"):
                rows, summary = _pair_records_for_protocol(
                    positives,
                    pair_set=pair_set,
                    all_candidates=records,
                    candidates_by_charge=candidates_by_charge,
                    mz_by_charge=mz_by_charge,
                    seed=seed,
                    tolerance_ppm=precursor_tolerance_ppm,
                    max_pairs_per_species=max_pairs_per_species,
                )
                if not rows:
                    raise ValueError(f"{species} produced no {pair_set} pairs.")
                pair_rows.extend(rows)
                species_rows.extend(rows)
                pair_set_summary[pair_set][species] = summary

            requested_ids = {
                str(row[column])
                for row in species_rows
                for column in ("left_spectrum_id", "right_spectrum_id")
            }
            records_by_id = {record.spectrum_id: record for record in records}
            spectra = _materialize_spectra(dataset, records_by_id, requested_ids)
            if spectra_writer is None:
                spectra_writer = pq.ParquetWriter(spectra_path, spectra.schema)
            spectra_writer.write_table(spectra)
            source_summary[species] = {
                "source_spectra": dataset.count_rows(),
                "excluded_spectra": excluded_rows,
                "eligible_spectra": len(records),
                "source_metadata_sha256": metadata_hash,
                "selected_peptide_ion_groups": len(selected_groups),
                "selected_spectra_before_pairing": sum(
                    len(group) for group in selected_groups.values()
                ),
                "materialized_spectra": spectra.num_rows,
            }
    finally:
        if spectra_writer is not None:
            spectra_writer.close()
    pq.write_table(pa.Table.from_pylist(pair_rows), pairs_path)
    return source_summary, pair_set_summary


def create_benchmarks(args: argparse.Namespace) -> dict[str, object]:
    if args.max_peptide_groups < 1 or args.max_peptide_ion_groups < 1:
        raise ValueError("Group caps must be positive.")
    if args.max_spectra_per_group < 2:
        raise ValueError("--max-spectra-per-group must be at least two.")
    source_root = Path(args.source_root)
    source_paths = sorted(source_root.glob("*.lance"))
    if not source_paths:
        raise FileNotFoundError(f"No Lance datasets found in {source_root}")
    output_root = Path(args.output_root)
    retrieval_path = output_root / "retrieval.parquet"
    retrieval_manifest_path = output_root / "retrieval.manifest.json"
    pair_dir = output_root / "pair_discrimination"
    pair_paths = [pair_dir / "spectra.parquet", pair_dir / "pairs.parquet", pair_dir / "manifest.json"]
    output_paths = [retrieval_path, retrieval_manifest_path, *pair_paths]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists ({existing[0]}). Pass --overwrite to replace it.")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    exclusion_path = Path(args.excluded_backbones_path)
    excluded_backbones = load_backbone_exclusion(exclusion_path)

    retrieval_summary = _create_retrieval(
        source_paths=source_paths,
        output_path=retrieval_path,
        seed=args.seed,
        max_peptides=args.max_peptide_groups,
        max_spectra_per_peptide=args.max_spectra_per_group,
        excluded_backbones=excluded_backbones,
    )
    retrieval_manifest = {
        "sampling_code_version": SAMPLING_CODE_VERSION,
        "seed": args.seed,
        "source_root": str(source_root),
        "source_dataset_version": args.source_dataset_version,
        "output_path": str(retrieval_path),
        "selection": {
            "max_peptide_groups_per_species": args.max_peptide_groups,
            "max_spectra_per_peptide": args.max_spectra_per_group,
            "retrieval_protocol": "one_deterministic_query_per_peptide_group_vs_same-species gallery",
        },
        "backbone_exclusion": {
            "path": str(exclusion_path),
            "normalization_version": BACKBONE_NORMALIZATION_VERSION,
            "excluded_backbone_count": len(excluded_backbones),
        },
        "species": retrieval_summary,
        "sha256": _hash_file(retrieval_path),
    }
    retrieval_manifest_path.write_text(json.dumps(retrieval_manifest, indent=2, sort_keys=True) + "\n")

    pair_summary, pair_set_summary = _create_pairs(
        source_paths=source_paths,
        output_dir=pair_dir,
        seed=args.seed,
        max_groups=args.max_peptide_ion_groups,
        max_spectra_per_group=args.max_spectra_per_group,
        max_positive_pairs_per_group=args.max_positive_pairs_per_group,
        max_pairs_per_species=args.max_pairs_per_species,
        precursor_tolerance_ppm=args.precursor_tolerance_ppm,
        excluded_backbones=excluded_backbones,
    )
    pair_manifest = {
        "sampling_code_version": SAMPLING_CODE_VERSION,
        "seed": args.seed,
        "source": {
            "root": str(source_root),
            "dataset_version": args.source_dataset_version,
            "canonical_precursor_mz_source_column": "precursor_mass",
            "datasets": pair_summary,
        },
        "construction": {
            "group_id": "modified_sequence + precursor_charge",
            "max_peptide_ion_groups_per_species": args.max_peptide_ion_groups,
            "max_spectra_per_peptide_ion": args.max_spectra_per_group,
            "max_positive_pairs_per_group": args.max_positive_pairs_per_group,
            "max_pairs_per_species_per_protocol": args.max_pairs_per_species,
            "negative_to_positive_ratio": 1,
            "same_charge_required": True,
            "hard_negative_precursor_tolerance_ppm": args.precursor_tolerance_ppm,
            "theoretical_fragment_overlap_filter": False,
        },
        "backbone_exclusion": {
            "path": str(exclusion_path),
            "normalization_version": BACKBONE_NORMALIZATION_VERSION,
            "excluded_backbone_count": len(excluded_backbones),
        },
        "pair_sets": pair_set_summary,
        "files": {
            "spectra": str(pair_dir / "spectra.parquet"),
            "spectra_sha256": _hash_file(pair_dir / "spectra.parquet"),
            "pairs": str(pair_dir / "pairs.parquet"),
            "pairs_sha256": _hash_file(pair_dir / "pairs.parquet"),
        },
    }
    (pair_dir / "manifest.json").write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")
    return {"retrieval": retrieval_manifest, "pair_discrimination": pair_manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/path/to/data/9_species_V2/lance_species")
    parser.add_argument("--source-dataset-version", default="9_species_V2")
    parser.add_argument(
        "--excluded-backbones-path",
        default=(
            "/path/to/data/probing_datasets/ninespecies_v2_paper_test/"
            "probe_excluded_backbones.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="/path/to/data/probing_datasets/ninespecies_v2_paper_test",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-peptide-groups", type=int, default=4000)
    parser.add_argument("--max-peptide-ion-groups", type=int, default=4000)
    parser.add_argument("--max-spectra-per-group", type=int, default=10)
    parser.add_argument("--max-positive-pairs-per-group", type=int, default=20)
    parser.add_argument("--max-pairs-per-species", type=int, default=100000)
    parser.add_argument("--precursor-tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(create_benchmarks(parse_args()), indent=2, sort_keys=True))
