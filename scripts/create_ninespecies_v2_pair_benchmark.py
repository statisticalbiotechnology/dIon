"""Create a deterministic peptide-ion pair benchmark from nine-species V2.

The source Lance field named ``precursor_mass`` contains observed precursor m/z
for NineSpecies V2. This script writes that value as canonical ``precursor_mz``
so benchmark extraction matches the DINO training input convention.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

# Permit direct ``python scripts/create_ninespecies_v2_pair_benchmark.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lance
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from src.embed_eval.pair_data import stable_rank


SAMPLING_CODE_VERSION = "ninespecies_v2_peptide_ion_pairs_v2"
METADATA_COLUMNS = ["modified_sequence", "precursor_mass", "precursor_charge"]
SPECTRUM_COLUMNS = METADATA_COLUMNS + ["mz_array", "intensity_array"]


@dataclass(frozen=True)
class SpectrumRecord:
    """One valid source spectrum with canonical precursor-m/z semantics."""

    row_index: int
    species: str
    peptide_ion_id: str
    precursor_mz: float
    precursor_charge: int

    @property
    def spectrum_id(self) -> str:
        return f"{self.species}:{self.row_index}"


def _stable_index(seed: int, size: int, *parts: object) -> int:
    if size < 1:
        raise ValueError("Cannot sample from an empty candidate list.")
    digest = stable_rank(seed, *parts)
    return int.from_bytes(digest[:8], "big") % size


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _valid_metadata_rows(
    dataset: lance.LanceDataset,
    species: str,
) -> tuple[list[SpectrumRecord], str]:
    """Read pair-construction metadata and return its reproducibility hash."""
    table = dataset.to_table(columns=METADATA_COLUMNS)
    hasher = hashlib.sha256()
    records = []
    for row_index, row in enumerate(table.to_pylist()):
        sequence = row["modified_sequence"]
        precursor_mz = row["precursor_mass"]
        charge = row["precursor_charge"]
        hasher.update(
            f"{row_index}\x1f{sequence!s}\x1f{precursor_mz!r}\x1f{charge!r}\n".encode(
                "utf-8"
            )
        )
        if not sequence or precursor_mz is None or charge is None:
            continue
        try:
            mz_value = float(precursor_mz)
            charge_value = int(charge)
        except (TypeError, ValueError):
            continue
        if mz_value <= 0.0 or charge_value < 1:
            continue
        peptide_ion_id = f"{sequence}|z={charge_value}"
        records.append(
            SpectrumRecord(
                row_index=row_index,
                species=species,
                peptide_ion_id=peptide_ion_id,
                precursor_mz=mz_value,
                precursor_charge=charge_value,
            )
        )
    return records, hasher.hexdigest()


def _selected_groups(
    records: Iterable[SpectrumRecord],
    *,
    seed: int,
    max_peptide_ion_groups: int | None,
    max_spectra_per_peptide_ion: int,
) -> dict[str, list[SpectrumRecord]]:
    groups: dict[str, list[SpectrumRecord]] = defaultdict(list)
    for record in records:
        groups[record.peptide_ion_id].append(record)
    repeated_groups = {
        group_id: group_records
        for group_id, group_records in groups.items()
        if len(group_records) >= 2
    }
    selected_group_ids = sorted(
        repeated_groups,
        key=lambda group_id: stable_rank(seed, "group", group_id),
    )
    if max_peptide_ion_groups is not None:
        selected_group_ids = selected_group_ids[:max_peptide_ion_groups]
    selected = {}
    for group_id in selected_group_ids:
        group_records = sorted(
            repeated_groups[group_id],
            key=lambda record: stable_rank(seed, "spectrum", group_id, record.row_index),
        )[:max_spectra_per_peptide_ion]
        if len(group_records) >= 2:
            selected[group_id] = group_records
    return selected


def _positive_pairs(
    groups: dict[str, list[SpectrumRecord]],
    *,
    seed: int,
    max_positive_pairs_per_group: int,
) -> list[tuple[SpectrumRecord, SpectrumRecord]]:
    pairs = []
    for group_id, records in sorted(groups.items()):
        candidate_pairs = list(itertools.combinations(records, 2))
        candidate_pairs.sort(
            key=lambda pair: stable_rank(
                seed, "positive", group_id, pair[0].row_index, pair[1].row_index
            )
        )
        pairs.extend(candidate_pairs[:max_positive_pairs_per_group])
    return sorted(
        pairs,
        key=lambda pair: stable_rank(
            seed, "positive-order", pair[0].species, pair[0].row_index, pair[1].row_index
        ),
    )


def _random_negative(
    anchor: SpectrumRecord,
    candidates: list[SpectrumRecord],
    *,
    seed: int,
    pair_set: str,
) -> SpectrumRecord | None:
    if not candidates:
        return None
    start = _stable_index(seed, len(candidates), pair_set, anchor.spectrum_id)
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        if candidate.peptide_ion_id != anchor.peptide_ion_id:
            return candidate
    return None


def _hard_negative(
    anchor: SpectrumRecord,
    candidates_by_charge: dict[int, list[SpectrumRecord]],
    mz_by_charge: dict[int, list[float]],
    *,
    seed: int,
    tolerance_ppm: float,
) -> SpectrumRecord | None:
    """Sample a different peptide ion with same charge within the m/z ppm window.

    With charge fixed, precursor m/z ppm and neutral-mass ppm are equivalent.
    """
    candidates = candidates_by_charge.get(anchor.precursor_charge, [])
    mz_values = mz_by_charge.get(anchor.precursor_charge, [])
    tolerance = anchor.precursor_mz * tolerance_ppm * 1e-6
    left = bisect.bisect_left(mz_values, anchor.precursor_mz - tolerance)
    right = bisect.bisect_right(mz_values, anchor.precursor_mz + tolerance)
    eligible = [
        candidate
        for candidate in candidates[left:right]
        if candidate.peptide_ion_id != anchor.peptide_ion_id
    ]
    if not eligible:
        return None
    return eligible[
        _stable_index(seed, len(eligible), "same-charge-10ppm", anchor.spectrum_id)
    ]


def _pair_records_for_protocol(
    positive_pairs: list[tuple[SpectrumRecord, SpectrumRecord]],
    *,
    pair_set: str,
    all_candidates: list[SpectrumRecord],
    candidates_by_charge: dict[int, list[SpectrumRecord]],
    mz_by_charge: dict[int, list[float]],
    seed: int,
    tolerance_ppm: float,
    max_pairs_per_species: int,
) -> tuple[list[dict[str, object]], dict[str, int | float]]:
    rows = []
    all_positive_anchor_ids = {left.spectrum_id for left, _ in positive_pairs}
    hard_anchor_ids = set()
    for left, right in positive_pairs:
        if pair_set == "all_random":
            negative = _random_negative(
                left, all_candidates, seed=seed, pair_set=pair_set
            )
        elif pair_set == "same_charge_random":
            negative = _random_negative(
                left,
                candidates_by_charge.get(left.precursor_charge, []),
                seed=seed,
                pair_set=pair_set,
            )
        elif pair_set == "same_charge_10ppm":
            negative = _hard_negative(
                left,
                candidates_by_charge,
                mz_by_charge,
                seed=seed,
                tolerance_ppm=tolerance_ppm,
            )
            if negative is not None:
                hard_anchor_ids.add(left.spectrum_id)
        else:
            raise ValueError(f"Unsupported pair set: {pair_set!r}")
        if negative is None:
            continue
        pair_number = len(rows) // 2
        base = {
            "pair_set": pair_set,
            "species": left.species,
            "anchor_spectrum_id": left.spectrum_id,
            "anchor_peptide_ion_id": left.peptide_ion_id,
            "anchor_precursor_charge": left.precursor_charge,
            "pair_number": pair_number,
        }
        rows.append(
            {
                **base,
                "pair_id": f"{pair_set}:{left.species}:{pair_number}:positive",
                "label": 1,
                "left_spectrum_id": left.spectrum_id,
                "right_spectrum_id": right.spectrum_id,
                "right_peptide_ion_id": right.peptide_ion_id,
                "precursor_ppm_difference": 0.0,
            }
        )
        ppm_difference = (
            abs(left.precursor_mz - negative.precursor_mz)
            / left.precursor_mz
            * 1e6
        )
        rows.append(
            {
                **base,
                "pair_id": f"{pair_set}:{left.species}:{pair_number}:negative",
                "label": 0,
                "left_spectrum_id": left.spectrum_id,
                "right_spectrum_id": negative.spectrum_id,
                "right_peptide_ion_id": negative.peptide_ion_id,
                "precursor_ppm_difference": ppm_difference,
            }
        )
        if pair_number + 1 >= max_pairs_per_species:
            break
    pair_count = len(rows) // 2
    summary: dict[str, int | float] = {
        "positive_pair_count": pair_count,
        "negative_pair_count": pair_count,
        "pair_count": len(rows),
    }
    if pair_set == "same_charge_10ppm":
        positive_anchor_count = len(all_positive_anchor_ids)
        summary.update(
            {
                # Denominator: unique anchors among selected positive pairs, before
                # hard-negative availability filters remove difficult anchors.
                "hard_negative_anchor_count": len(hard_anchor_ids),
                "positive_anchor_count": positive_anchor_count,
                "hard_negative_anchor_coverage": (
                    len(hard_anchor_ids) / positive_anchor_count
                    if positive_anchor_count
                    else 0.0
                ),
                "hard_negative_pair_count": pair_count,
            }
        )
    return rows, summary


def _materialize_spectra(
    dataset: lance.LanceDataset,
    records_by_id: dict[str, SpectrumRecord],
    requested_ids: set[str],
) -> pa.Table:
    selected_records = sorted(
        (records_by_id[spectrum_id] for spectrum_id in requested_ids),
        key=lambda record: record.row_index,
    )
    source = dataset.take(
        [record.row_index for record in selected_records], columns=SPECTRUM_COLUMNS
    )
    return pa.table(
        {
            "spectrum_id": pa.array([record.spectrum_id for record in selected_records]),
            "species": pa.array([record.species for record in selected_records]),
            "peptide_ion_id": pa.array(
                [record.peptide_ion_id for record in selected_records]
            ),
            "modified_sequence": source["modified_sequence"],
            "source_row_index": pa.array(
                [record.row_index for record in selected_records], type=pa.int64()
            ),
            "precursor_mz": source["precursor_mass"],
            "precursor_charge": source["precursor_charge"],
            "mz_array": source["mz_array"],
            "intensity_array": source["intensity_array"],
        }
    )


def create_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Write static spectra, balanced pair tables, and reproducibility manifest."""
    if args.max_peptide_ion_groups is not None and args.max_peptide_ion_groups < 1:
        raise ValueError("max_peptide_ion_groups must be positive or null.")
    if args.max_spectra_per_peptide_ion < 2:
        raise ValueError("max_spectra_per_peptide_ion must be at least two.")
    if args.max_positive_pairs_per_group < 1 or args.max_pairs_per_species < 1:
        raise ValueError("Pair caps must be positive.")
    if args.precursor_tolerance_ppm <= 0.0:
        raise ValueError("precursor_tolerance_ppm must be positive.")

    source_root = Path(args.source_root)
    source_paths = sorted(source_root.glob("*.lance"))
    if not source_paths:
        raise FileNotFoundError(f"No Lance datasets found in {source_root}")
    output_dir = Path(args.output_dir)
    output_paths = {
        "spectra": output_dir / "spectra.parquet",
        "pairs": output_dir / "pairs.parquet",
        "manifest": output_dir / "manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Benchmark output already exists ({existing[0]}). Pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()

    spectra_writer = None
    all_pair_rows = []
    source_summary = {}
    pair_set_summary: dict[str, dict[str, dict[str, int | float]]] = defaultdict(dict)
    try:
        for source_path in tqdm(source_paths, desc="Building pair benchmark", unit="species"):
            species = source_path.stem
            dataset = lance.dataset(str(source_path))
            records, metadata_hash = _valid_metadata_rows(dataset, species)
            selected_groups = _selected_groups(
                records,
                seed=args.seed,
                max_peptide_ion_groups=args.max_peptide_ion_groups,
                max_spectra_per_peptide_ion=args.max_spectra_per_peptide_ion,
            )
            selected_records = [record for group in selected_groups.values() for record in group]
            records_by_id = {record.spectrum_id: record for record in records}
            candidates_by_charge: dict[int, list[SpectrumRecord]] = defaultdict(list)
            for record in records:
                candidates_by_charge[record.precursor_charge].append(record)
            mz_by_charge = {}
            for charge, candidates in candidates_by_charge.items():
                candidates.sort(key=lambda record: (record.precursor_mz, record.row_index))
                mz_by_charge[charge] = [record.precursor_mz for record in candidates]

            positives = _positive_pairs(
                selected_groups,
                seed=args.seed,
                max_positive_pairs_per_group=args.max_positive_pairs_per_group,
            )
            source_summary[species] = {
                "source_spectra": dataset.count_rows(),
                "valid_spectra": len(records),
                "source_metadata_sha256": metadata_hash,
                "selected_peptide_ion_groups": len(selected_groups),
                "selected_spectra_before_pairing": len(selected_records),
            }
            species_pair_rows = []
            for pair_set in (
                "all_random",
                "same_charge_random",
                "same_charge_10ppm",
            ):
                rows, summary = _pair_records_for_protocol(
                    positives,
                    pair_set=pair_set,
                    all_candidates=records,
                    candidates_by_charge=candidates_by_charge,
                    mz_by_charge=mz_by_charge,
                    seed=args.seed,
                    tolerance_ppm=args.precursor_tolerance_ppm,
                    max_pairs_per_species=args.max_pairs_per_species,
                )
                if not rows:
                    raise ValueError(
                        f"{species} produced no {pair_set} records; adjust benchmark caps."
                    )
                all_pair_rows.extend(rows)
                species_pair_rows.extend(rows)
                pair_set_summary[pair_set][species] = summary
            requested_ids = {
                str(row[column])
                for row in species_pair_rows
                for column in ("left_spectrum_id", "right_spectrum_id")
            }
            spectra = _materialize_spectra(dataset, records_by_id, requested_ids)
            if spectra_writer is None:
                spectra_writer = pq.ParquetWriter(output_paths["spectra"], spectra.schema)
            spectra_writer.write_table(spectra)
            source_summary[species]["materialized_spectra"] = spectra.num_rows
    finally:
        if spectra_writer is not None:
            spectra_writer.close()

    pairs = pa.Table.from_pylist(all_pair_rows)
    pq.write_table(pairs, output_paths["pairs"])
    manifest = {
        "sampling_code_version": SAMPLING_CODE_VERSION,
        "seed": args.seed,
        "source": {
            "root": str(source_root),
            "dataset_version": args.source_dataset_version,
            "datasets": source_summary,
            "canonical_precursor_mz_source_column": "precursor_mass",
        },
        "construction": {
            "group_id": "modified_sequence + precursor_charge",
            "max_peptide_ion_groups_per_species": args.max_peptide_ion_groups,
            "max_spectra_per_peptide_ion": args.max_spectra_per_peptide_ion,
            "max_positive_pairs_per_group": args.max_positive_pairs_per_group,
            "max_pairs_per_species_per_protocol": args.max_pairs_per_species,
            "negative_to_positive_ratio": 1,
            "same_charge_required": True,
            "hard_negative_precursor_tolerance_ppm": args.precursor_tolerance_ppm,
            "theoretical_fragment_overlap_filter": False,
        },
        "pair_sets": pair_set_summary,
        "files": {
            "spectra": str(output_paths["spectra"]),
            "spectra_sha256": _hash_file(output_paths["spectra"]),
            "pairs": str(output_paths["pairs"]),
            "pairs_sha256": _hash_file(output_paths["pairs"]),
        },
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="/path/to/data/9_species_V2/lance_species",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/path/to/data/probing_datasets/"
            "ninespecies_v2_peptide_ion_pairs"
        ),
    )
    parser.add_argument("--source-dataset-version", default="9_species_V2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-peptide-ion-groups", type=int, default=1000)
    parser.add_argument("--max-spectra-per-peptide-ion", type=int, default=10)
    parser.add_argument("--max-positive-pairs-per-group", type=int, default=20)
    parser.add_argument("--max-pairs-per-species", type=int, default=10000)
    parser.add_argument("--precursor-tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(create_benchmark(parse_args()), indent=2, sort_keys=True))
