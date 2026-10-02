#!/usr/bin/env python
"""Materialize a charge-filtered retrieval and static-pair benchmark.

The source root must contain ``retrieval.parquet`` and
``pair_discrimination/{spectra.parquet,pairs.parquet}``, as produced by the
canonical embedding-benchmark builders. Retrieval rows are filtered directly.
For static pairs, both endpoints must pass the charge filter; surviving pairs
are then deterministically rebalanced within each (pair_set, species) stratum.

The implementation streams source Parquet row groups and bounds pair selection
memory to one stratum. It is suitable for the multi-million-row locked tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embed_eval.pair_data import stable_rank


BATCH_SIZE = 65_536


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle, tqdm(
        total=path.stat().st_size,
        desc=f"Hashing {path.name}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            progress.update(len(chunk))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _charge_indices(table: pa.Table, minimum: int, maximum: int) -> list[int]:
    if "precursor_charge" not in table.column_names:
        raise ValueError("Benchmark table is missing precursor_charge.")
    return [
        index
        for index, charge in enumerate(table["precursor_charge"].to_pylist())
        if charge is not None and minimum <= int(charge) <= maximum
    ]


def _take(table: pa.Table, indices: list[int]) -> pa.Table:
    return table.take(pa.array(indices, type=pa.int64()))


def _batch_count(parquet: pq.ParquetFile) -> int:
    return (parquet.metadata.num_rows + BATCH_SIZE - 1) // BATCH_SIZE


def _stream_charge_filtered(
    source: Path,
    output: Path,
    *,
    minimum: int,
    maximum: int,
) -> int:
    parquet = pq.ParquetFile(source)
    writer = None
    count = 0
    try:
        for batch in tqdm(
            parquet.iter_batches(batch_size=BATCH_SIZE),
            total=_batch_count(parquet),
            desc=f"Filter retrieval {source.parent.name}",
            unit="batch",
            dynamic_ncols=True,
        ):
            table = pa.Table.from_batches([batch])
            filtered = _take(table, _charge_indices(table, minimum, maximum))
            if not filtered.num_rows:
                continue
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output, filtered.schema)
            writer.write_table(filtered)
            count += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()
    if count == 0:
        raise ValueError(f"No rows pass charge filter for {source}.")
    return count


def _qualified_spectrum_ids(
    spectra_path: Path,
    *,
    minimum: int,
    maximum: int,
) -> tuple[set[str], int]:
    qualified: set[str] = set()
    count = 0
    parquet = pq.ParquetFile(spectra_path)
    for batch in tqdm(
        parquet.iter_batches(batch_size=BATCH_SIZE, columns=["spectrum_id", "precursor_charge"]),
        total=_batch_count(parquet),
        desc=f"Index charge-qualified spectra {spectra_path.parent.parent.name}",
        unit="batch",
        dynamic_ncols=True,
    ):
        table = pa.Table.from_batches([batch])
        indices = _charge_indices(table, minimum, maximum)
        count += len(indices)
        qualified.update(str(table["spectrum_id"][index].as_py()) for index in indices)
    return qualified, count


def _stratum_key(pair_set: str, species: str, label: int) -> str:
    payload = "\x1f".join((pair_set, species, str(label))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_candidate_strata(
    pairs_path: Path,
    candidate_root: Path,
    *,
    retained_ids: set[str],
) -> tuple[dict[tuple[str, str, int], Path], dict[tuple[str, str, int], int]]:
    candidate_root.mkdir(parents=True)
    paths: dict[tuple[str, str, int], Path] = {}
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    writers: dict[tuple[str, str, int], pq.ParquetWriter] = {}
    parquet = pq.ParquetFile(pairs_path)
    try:
        for batch in tqdm(
            parquet.iter_batches(batch_size=BATCH_SIZE),
            total=_batch_count(parquet),
            desc=f"Filter pair endpoints {pairs_path.parent.parent.name}",
            unit="batch",
            dynamic_ncols=True,
        ):
            table = pa.Table.from_batches([batch])
            eligible_indices = [
                index
                for index, (left, right) in enumerate(
                    zip(
                        table["left_spectrum_id"].to_pylist(),
                        table["right_spectrum_id"].to_pylist(),
                        strict=True,
                    )
                )
                if str(left) in retained_ids and str(right) in retained_ids
            ]
            if not eligible_indices:
                continue
            eligible = _take(table, eligible_indices)
            groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
            for index, (pair_set, species, label) in enumerate(
                zip(
                    eligible["pair_set"].to_pylist(),
                    eligible["species"].to_pylist(),
                    eligible["label"].to_pylist(),
                    strict=True,
                )
            ):
                groups[(str(pair_set), str(species), int(label))].append(index)
            for key, indices in groups.items():
                subset = _take(eligible, indices)
                if key not in writers:
                    path = candidate_root / f"{_stratum_key(*key)}.parquet"
                    paths[key] = path
                    writers[key] = pq.ParquetWriter(path, subset.schema)
                writers[key].write_table(subset)
                counts[key] += subset.num_rows
    finally:
        for writer in writers.values():
            writer.close()
    return paths, dict(counts)


def _ranked_subset(
    path: Path,
    *,
    count: int,
    seed: int,
    pair_set: str,
    species: str,
    label: int,
) -> pa.Table:
    table = pq.read_table(path)
    if table.num_rows < count:
        raise ValueError(f"Candidate stratum has {table.num_rows}, expected at least {count}.")
    pair_ids = table["pair_id"].to_pylist()
    indices = sorted(
        range(table.num_rows),
        key=lambda index: stable_rank(seed, pair_set, species, label, pair_ids[index]),
    )[:count]
    return _take(table, indices)


def _rebalance_pairs(
    candidate_paths: dict[tuple[str, str, int], Path],
    candidate_counts: dict[tuple[str, str, int], int],
    output_path: Path,
    *,
    seed: int,
) -> tuple[int, dict[str, dict[str, dict[str, int]]], list[str], set[str]]:
    strata = {(pair_set, species) for pair_set, species, _ in candidate_counts}
    writer = None
    retained_count = 0
    selected_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    dropped: list[str] = []
    referenced_ids: set[str] = set()
    try:
        for pair_set, species in tqdm(
            sorted(strata),
            desc="Rebalance pair strata",
            unit="stratum",
            dynamic_ncols=True,
        ):
            positive_key = (pair_set, species, 1)
            negative_key = (pair_set, species, 0)
            if positive_key not in candidate_counts or negative_key not in candidate_counts:
                dropped.append(f"{pair_set}/{species}")
                continue
            count = min(candidate_counts[positive_key], candidate_counts[negative_key])
            if count < 1:
                dropped.append(f"{pair_set}/{species}")
                continue
            for label, key in ((0, negative_key), (1, positive_key)):
                selected = _ranked_subset(
                    candidate_paths[key],
                    count=count,
                    seed=seed,
                    pair_set=pair_set,
                    species=species,
                    label=label,
                )
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, selected.schema)
                writer.write_table(selected)
                retained_count += selected.num_rows
                referenced_ids.update(str(value) for value in selected["left_spectrum_id"].to_pylist())
                referenced_ids.update(str(value) for value in selected["right_spectrum_id"].to_pylist())
            selected_counts[pair_set][species] = {"0": count, "1": count}
    finally:
        if writer is not None:
            writer.close()
    if retained_count == 0:
        raise ValueError("No balanced static pairs remain after charge filtering.")
    return retained_count, selected_counts, dropped, referenced_ids


def _write_referenced_spectra(
    source: Path,
    output: Path,
    *,
    minimum: int,
    maximum: int,
    referenced_ids: set[str],
) -> int:
    writer = None
    count = 0
    parquet = pq.ParquetFile(source)
    try:
        for batch in tqdm(
            parquet.iter_batches(batch_size=BATCH_SIZE),
            total=_batch_count(parquet),
            desc=f"Write referenced pair spectra {source.parent.parent.name}",
            unit="batch",
            dynamic_ncols=True,
        ):
            table = pa.Table.from_batches([batch])
            indices = [
                index
                for index in _charge_indices(table, minimum, maximum)
                if str(table["spectrum_id"][index].as_py()) in referenced_ids
            ]
            if not indices:
                continue
            selected = _take(table, indices)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output, selected.schema)
            writer.write_table(selected)
            count += selected.num_rows
    finally:
        if writer is not None:
            writer.close()
    if count != len(referenced_ids):
        raise ValueError(
            f"Wrote {count} pair spectra but pairs reference {len(referenced_ids)} unique IDs."
        )
    return count


def main() -> None:
    args = parse_args()
    if args.min_charge < 1 or args.max_charge < args.min_charge:
        raise ValueError("Require 1 <= min-charge <= max-charge.")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    retrieval_path = source_root / "retrieval.parquet"
    spectra_path = source_root / "pair_discrimination" / "spectra.parquet"
    pairs_path = source_root / "pair_discrimination" / "pairs.parquet"
    source_manifest_path = source_root / "pair_discrimination" / "manifest.json"
    required_paths = (retrieval_path, spectra_path, pairs_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing canonical benchmark artifacts: {missing}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)
    pair_root = output_root / "pair_discrimination"
    candidate_root = output_root / ".pair_candidates"
    source_retrieval_rows = pq.ParquetFile(retrieval_path).metadata.num_rows
    source_spectra_rows = pq.ParquetFile(spectra_path).metadata.num_rows
    source_pair_rows = pq.ParquetFile(pairs_path).metadata.num_rows
    retained_retrieval_rows = _stream_charge_filtered(
        retrieval_path,
        output_root / "retrieval.parquet",
        minimum=args.min_charge,
        maximum=args.max_charge,
    )
    qualified_ids, qualified_spectra_rows = _qualified_spectrum_ids(
        spectra_path, minimum=args.min_charge, maximum=args.max_charge
    )
    candidate_paths, candidate_counts = _write_candidate_strata(
        pairs_path, candidate_root, retained_ids=qualified_ids
    )
    retained_pair_rows, selected_counts, dropped, referenced_ids = _rebalance_pairs(
        candidate_paths,
        candidate_counts,
        pair_root / "pairs.parquet",
        seed=args.seed,
    )
    retained_pair_spectra_rows = _write_referenced_spectra(
        spectra_path,
        pair_root / "spectra.parquet",
        minimum=args.min_charge,
        maximum=args.max_charge,
        referenced_ids=referenced_ids,
    )
    shutil.rmtree(candidate_root)

    manifest = {
        "artifact": "charge_filtered_embedding_benchmark",
        "source_root": str(source_root),
        "charge_filter": {"min": args.min_charge, "max": args.max_charge},
        "seed": args.seed,
        "source": {
            "retrieval_path": str(retrieval_path),
            "retrieval_sha256": sha256(retrieval_path),
            "pair_spectra_path": str(spectra_path),
            "pair_spectra_sha256": sha256(spectra_path),
            "pairs_path": str(pairs_path),
            "pairs_sha256": sha256(pairs_path),
            "pair_manifest_path": str(source_manifest_path) if source_manifest_path.exists() else None,
            "pair_manifest_sha256": sha256(source_manifest_path) if source_manifest_path.exists() else None,
        },
        "counts": {
            "source_retrieval_rows": source_retrieval_rows,
            "retained_retrieval_rows": retained_retrieval_rows,
            "source_pair_spectra": source_spectra_rows,
            "charge_qualified_pair_spectra": qualified_spectra_rows,
            "retained_pair_spectra": retained_pair_spectra_rows,
            "source_pairs": source_pair_rows,
            "retained_balanced_pairs": retained_pair_rows,
        },
        "selected_pairs_by_set_species_label": selected_counts,
        "dropped_pair_set_species": dropped,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {retained_retrieval_rows:,} retrieval rows: {output_root / 'retrieval.parquet'}")
    print(f"Wrote {retained_pair_spectra_rows:,} pair spectra and {retained_pair_rows:,} balanced pairs: {pair_root}")
    print(f"Wrote manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
