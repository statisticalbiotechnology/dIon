#!/usr/bin/env python
"""Materialize locked bacterial DINO validation and PXD010613 test benchmarks.

This follows the static retrieval and GLEAMS-style pair construction used by
the NineSpecies and Kingdoms benchmarks.  It intentionally separates the
PXD010000 validation runs from the PXD010613 test runs.  A second PXD010613
view excludes modified peptide sequences observed in either PXD010000 split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import lance
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


SAMPLING_CODE_VERSION = "bacterial_pxd010000_pxd010613_paper_benchmarks_v1"
BENCHMARK_SPECIES = "bacteria_pxd010613"
SOURCE_COLUMNS = [
    "peak_file",
    "scan_id",
    "seq",
    "precursor_mz",
    "precursor_charge",
    "mz_array",
    "intensity_array",
]
METADATA_COLUMNS = ["seq", "precursor_mz", "precursor_charge"]


def _hash_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_records(
    dataset: lance.LanceDataset,
    *,
    excluded_sequences: set[str],
) -> tuple[list[SpectrumRecord], dict[str, list[int]], dict[str, int]]:
    """Return valid records and exact modified-sequence retrieval groups."""
    metadata = dataset.to_table(columns=METADATA_COLUMNS)
    records: list[SpectrumRecord] = []
    retrieval_groups: dict[str, list[int]] = defaultdict(list)
    summary = {"source_rows": metadata.num_rows, "excluded_sequence_rows": 0, "invalid_rows": 0}
    for row_index, (sequence, mz, charge) in enumerate(
        zip(
            metadata["seq"].to_pylist(),
            metadata["precursor_mz"].to_pylist(),
            metadata["precursor_charge"].to_pylist(),
        )
    ):
        if not sequence or sequence in excluded_sequences:
            summary["excluded_sequence_rows"] += int(bool(sequence))
            continue
        try:
            mz_value = float(mz)
            charge_value = int(charge)
        except (TypeError, ValueError):
            summary["invalid_rows"] += 1
            continue
        if mz_value <= 0.0 or charge_value < 1:
            summary["invalid_rows"] += 1
            continue
        records.append(
            SpectrumRecord(
                row_index=row_index,
                species=BENCHMARK_SPECIES,
                peptide_ion_id=f"{sequence}|z={charge_value}",
                precursor_mz=mz_value,
                precursor_charge=charge_value,
            )
        )
        retrieval_groups[str(sequence)].append(row_index)
    summary["eligible_rows"] = len(records)
    summary["eligible_repeated_sequence_groups"] = sum(
        len(rows) >= 2 for rows in retrieval_groups.values()
    )
    return records, retrieval_groups, summary


def _selected_retrieval_groups(
    groups: dict[str, list[int]], *, seed: int, max_groups: int | None, max_spectra: int
) -> dict[str, list[int]]:
    repeated = {sequence: rows for sequence, rows in groups.items() if len(rows) >= 2}
    sequence_ids = sorted(repeated, key=lambda sequence: _stable_rank(seed, "bacterial-retrieval", sequence))
    if max_groups is not None:
        sequence_ids = sequence_ids[:max_groups]
    return {
        sequence: sorted(
            repeated[sequence], key=lambda row: _stable_rank(seed, "bacterial-retrieval-row", sequence, row)
        )[:max_spectra]
        for sequence in sequence_ids
    }


def _retrieval_table(dataset: lance.LanceDataset, selected: dict[str, list[int]]) -> pa.Table:
    rows = [row for sequence in sorted(selected) for row in selected[sequence]]
    source = dataset.take(pa.array(rows, type=pa.int64()), columns=SOURCE_COLUMNS)
    peptide_ids = source["seq"].to_pylist()
    return pa.table(
        {
            "species": pa.array([BENCHMARK_SPECIES] * len(rows)),
            "peptide_id": source["seq"],
            "source_row_index": pa.array(rows, type=pa.int64()),
            "selected_group_size": pa.array([len(selected[sequence]) for sequence in peptide_ids], type=pa.int16()),
            "precursor_mz": source["precursor_mz"],
            "precursor_charge": source["precursor_charge"],
            "mz_array": source["mz_array"],
            "intensity_array": source["intensity_array"],
        }
    )


def _pair_spectra_table(
    dataset: lance.LanceDataset, records_by_id: dict[str, SpectrumRecord], requested_ids: set[str]
) -> pa.Table:
    records = sorted((records_by_id[spectrum_id] for spectrum_id in requested_ids), key=lambda record: record.row_index)
    source = dataset.take(pa.array([record.row_index for record in records], type=pa.int64()), columns=SOURCE_COLUMNS)
    return pa.table(
        {
            "spectrum_id": pa.array([record.spectrum_id for record in records]),
            "species": pa.array([BENCHMARK_SPECIES] * len(records)),
            "peptide_ion_id": pa.array([record.peptide_ion_id for record in records]),
            "modified_sequence": source["seq"],
            "source_row_index": pa.array([record.row_index for record in records], type=pa.int64()),
            "precursor_mz": source["precursor_mz"],
            "precursor_charge": source["precursor_charge"],
            "mz_array": source["mz_array"],
            "intensity_array": source["intensity_array"],
        }
    )


def _write_split(
    *,
    dataset_path: Path,
    output_root: Path,
    split_name: str,
    seed: int,
    excluded_sequences: set[str],
    exclusion_description: str,
    max_retrieval_groups: int | None,
    max_pair_groups: int | None,
    max_pairs_per_protocol: int | None,
    max_spectra_per_group: int,
    max_positive_pairs_per_group: int,
    tolerance_ppm: float,
    overwrite: bool,
) -> dict[str, object]:
    """Build one deterministic retrieval and pair benchmark from one Lance split."""
    retrieval_path = output_root / "retrieval.parquet"
    retrieval_manifest_path = output_root / "retrieval.manifest.json"
    pair_dir = output_root / "pair_discrimination"
    spectra_path = pair_dir / "spectra.parquet"
    pairs_path = pair_dir / "pairs.parquet"
    pair_manifest_path = pair_dir / "manifest.json"
    outputs = [retrieval_path, retrieval_manifest_path, spectra_path, pairs_path, pair_manifest_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{existing[0]} exists; pass --overwrite to replace it.")
    if overwrite:
        for path in existing:
            path.unlink()
    pair_dir.mkdir(parents=True, exist_ok=True)

    dataset = lance.dataset(dataset_path)
    records, all_groups, summary = _valid_records(dataset, excluded_sequences=excluded_sequences)
    selected_retrieval = _selected_retrieval_groups(
        all_groups, seed=seed, max_groups=max_retrieval_groups, max_spectra=max_spectra_per_group
    )
    if not selected_retrieval:
        raise ValueError(f"{split_name} has no repeated eligible peptide groups.")
    tqdm.write(f"{split_name}: materializing {len(selected_retrieval):,} retrieval peptide groups")
    retrieval = _retrieval_table(dataset, selected_retrieval)
    pq.write_table(retrieval, retrieval_path, compression="zstd")

    selected_pair_groups = _selected_groups(
        records,
        seed=seed,
        max_peptide_ion_groups=max_pair_groups,
        max_spectra_per_peptide_ion=max_spectra_per_group,
    )
    positives = _positive_pairs(
        selected_pair_groups, seed=seed, max_positive_pairs_per_group=max_positive_pairs_per_group
    )
    by_charge: dict[int, list[SpectrumRecord]] = defaultdict(list)
    for record in records:
        by_charge[record.precursor_charge].append(record)
    mz_by_charge = {}
    for charge, candidates in by_charge.items():
        candidates.sort(key=lambda record: (record.precursor_mz, record.row_index))
        mz_by_charge[charge] = [record.precursor_mz for record in candidates]

    pair_rows = []
    pair_summary = {}
    for pair_set in ("all_random", "same_charge_random", "same_charge_10ppm"):
        rows, protocol_summary = _pair_records_for_protocol(
            positives,
            pair_set=pair_set,
            all_candidates=records,
            candidates_by_charge=by_charge,
            mz_by_charge=mz_by_charge,
            seed=seed,
            tolerance_ppm=tolerance_ppm,
            max_pairs_per_species=max_pairs_per_protocol if max_pairs_per_protocol is not None else len(positives),
        )
        if not rows:
            raise ValueError(f"{split_name} produced no {pair_set} pairs.")
        pair_rows.extend(rows)
        pair_summary[pair_set] = protocol_summary
    tqdm.write(f"{split_name}: writing {len(pair_rows):,} fixed pair rows")
    pq.write_table(pa.Table.from_pylist(pair_rows), pairs_path, compression="zstd")
    requested_ids = {str(row[column]) for row in pair_rows for column in ("left_spectrum_id", "right_spectrum_id")}
    spectra = _pair_spectra_table(dataset, {record.spectrum_id: record for record in records}, requested_ids)
    pq.write_table(spectra, spectra_path, compression="zstd")

    common = {
        "sampling_code_version": SAMPLING_CODE_VERSION,
        "seed": seed,
        "split": split_name,
        "source_lance": str(dataset_path),
        "source_row_count": dataset.count_rows(),
        "source_regeneration_manifest_sha256": _hash_json(dataset_path.parent / "manifest.json"),
        "exclusion": {"description": exclusion_description, "excluded_modified_sequence_count": len(excluded_sequences)},
    }
    retrieval_manifest = {
        **common,
        "selection": {
            "max_peptide_groups": max_retrieval_groups,
            "max_spectra_per_peptide": max_spectra_per_group,
            "retrieval_protocol": "one deterministic query per repeated modified-peptide group",
        },
        "summary": {**summary, "selected_peptide_groups": len(selected_retrieval), "selected_spectra": retrieval.num_rows},
    }
    retrieval_manifest_path.write_text(json.dumps(retrieval_manifest, indent=2, sort_keys=True) + "\n")
    pair_manifest = {
        **common,
        "construction": {
            "group_id": "exact modified sequence + precursor charge",
            "max_peptide_ion_groups": max_pair_groups,
            "max_spectra_per_peptide_ion": max_spectra_per_group,
            "max_positive_pairs_per_group": max_positive_pairs_per_group,
            "max_pairs_per_protocol": max_pairs_per_protocol,
            "negative_to_positive_ratio": 1,
            "same_charge_required": True,
            "hard_negative_precursor_tolerance_ppm": tolerance_ppm,
        },
        "summary": {**summary, "selected_peptide_ion_groups": len(selected_pair_groups), "materialized_spectra": spectra.num_rows},
        "pair_sets": pair_summary,
    }
    pair_manifest_path.write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")
    return {"retrieval": retrieval_manifest, "pair_discrimination": pair_manifest}


def _training_sequences(train_path: Path, val_path: Path) -> set[str]:
    """Exact modified peptide sequences observed in PXD010000 pretraining splits."""
    sequences: set[str] = set()
    for path in (train_path, val_path):
        dataset = lance.dataset(path)
        sequences.update(sequence for sequence in dataset.to_table(columns=["seq"])["seq"].to_pylist() if sequence)
    return sequences


def parse_args() -> argparse.Namespace:
    root = "/path/to/data/bacteria_PXD010000__PXD010613/annotated_regenerated_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-lance", default=f"{root}/train.lance")
    parser.add_argument("--validation-lance", default=f"{root}/val.lance")
    parser.add_argument("--test-lance", default=f"{root}/test.lance")
    parser.add_argument("--output-root", default="/path/to/data/probing_datasets/bacterial_paper_benchmarks")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--validation-max-retrieval-groups", type=int, default=20000)
    parser.add_argument("--validation-max-pair-groups", type=int, default=20000)
    parser.add_argument("--validation-max-pairs-per-protocol", type=int, default=100000)
    parser.add_argument("--test-max-retrieval-groups", type=int, default=None)
    parser.add_argument("--test-max-pair-groups", type=int, default=None)
    parser.add_argument("--test-max-pairs-per-protocol", type=int, default=None)
    parser.add_argument("--max-spectra-per-group", type=int, default=10)
    parser.add_argument("--max-positive-pairs-per-group", type=int, default=20)
    parser.add_argument("--precursor-tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    train_path = Path(args.train_lance)
    validation_path = Path(args.validation_lance)
    test_path = Path(args.test_lance)
    root = Path(args.output_root)
    for path in (train_path, validation_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)
    train_sequences = _training_sequences(train_path, validation_path)
    results = {
        "validation": _write_split(
            dataset_path=validation_path, output_root=root / "validation", split_name="pxd010000_validation",
            seed=args.seed, excluded_sequences=set(), exclusion_description="none", max_retrieval_groups=args.validation_max_retrieval_groups,
            max_pair_groups=args.validation_max_pair_groups, max_pairs_per_protocol=args.validation_max_pairs_per_protocol,
            max_spectra_per_group=args.max_spectra_per_group, max_positive_pairs_per_group=args.max_positive_pairs_per_group,
            tolerance_ppm=args.precursor_tolerance_ppm, overwrite=args.overwrite,
        ),
        "test_run_held_out": _write_split(
            dataset_path=test_path, output_root=root / "test_run_held_out", split_name="pxd010613_run_held_out",
            seed=args.seed, excluded_sequences=set(), exclusion_description="none; all eligible PXD010613 spectra retained", max_retrieval_groups=args.test_max_retrieval_groups,
            max_pair_groups=args.test_max_pair_groups, max_pairs_per_protocol=args.test_max_pairs_per_protocol,
            max_spectra_per_group=args.max_spectra_per_group, max_positive_pairs_per_group=args.max_positive_pairs_per_group,
            tolerance_ppm=args.precursor_tolerance_ppm, overwrite=args.overwrite,
        ),
        "test_peptide_disjoint": _write_split(
            dataset_path=test_path, output_root=root / "test_peptide_disjoint", split_name="pxd010613_run_and_peptide_disjoint",
            seed=args.seed, excluded_sequences=train_sequences,
            exclusion_description="exact modified sequences present in PXD010000 train.lance or val.lance", max_retrieval_groups=args.test_max_retrieval_groups,
            max_pair_groups=args.test_max_pair_groups, max_pairs_per_protocol=args.test_max_pairs_per_protocol,
            max_spectra_per_group=args.max_spectra_per_group, max_positive_pairs_per_group=args.max_positive_pairs_per_group,
            tolerance_ppm=args.precursor_tolerance_ppm, overwrite=args.overwrite,
        ),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
