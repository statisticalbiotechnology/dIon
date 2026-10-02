#!/usr/bin/env python
"""Materialize nested peptide-label budgets for bacterial metric learning.

The label identity is the exact ``seq`` value in the active bacterial Lance
corpus.  This is the project's canonical *modified-sequence* identity used by
the bacterial retrieval benchmarks.  It is deliberately neither normalized to
an amino-acid backbone nor combined with precursor charge.  Charge is instead
available to the later metric-learning sampler for same-charge hard negatives.

The locked validation and strict PXD010613 test benchmark manifests are only
read for leakage auditing; this script never changes them.  The strict test
manifest is exact-modified-sequence disjoint from train/validation by design,
whereas the PXD010000 validation manifest is run-held-out and may overlap in
peptide identity with training.

By default, an identity must have at least two base-eligible spectra before it
enters the pool.  dIon does not yet contain the supervised metric-learning
sampler, but this explicit default guarantees that every selected identity can
form a same-peptide positive pair.  It is applied before the 1/10/100 percent
budgets are sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import lance
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


CODE_VERSION = "bacterial_metric_learning_peptide_budget_v1"
IDENTITY_DEFINITION = "exact_modified_sequence_from_seq_v1"
DEFAULT_SOURCE_ROOT = Path(
    "/path/to/data/bacteria_PXD010000__PXD010613/annotated_regenerated_v3"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/path/to/data/bacteria_PXD010000__PXD010613/annotated_regenerated_v3/metric_learning"
)
DEFAULT_VALIDATION_MANIFEST = Path(
    "/path/to/data/probing_datasets/bacterial_paper_benchmarks/validation/retrieval.parquet"
)
DEFAULT_TEST_MANIFEST = Path(
    "/path/to/data/probing_datasets/bacterial_paper_benchmarks/test_peptide_disjoint/retrieval.parquet"
)
SOURCE_COLUMNS = ["seq", "peak_file", "scan_id", "precursor_mz", "precursor_charge"]
FRACTIONS = (("1pct", 0.01), ("10pct", 0.10), ("100pct", 1.00))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lance", type=Path, default=DEFAULT_SOURCE_ROOT / "train.lance")
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--min-spectra-per-peptide",
        type=int,
        default=2,
        help="Apply before sampling; 2 guarantees an identity can form a positive pair.",
    )
    parser.add_argument("--max-charge", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output root. Default behavior preserves immutable manifests.",
    )
    return parser.parse_args()


def _stable_rank(seed: int, peptide_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|metric-learning-peptide-budget|{peptide_id}".encode()).digest()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_fingerprint(source_lance: Path) -> dict[str, Any]:
    root_manifest = source_lance.parent / "manifest.json"
    manifest_sha256 = None
    if root_manifest.exists():
        manifest_sha256 = hashlib.sha256(root_manifest.read_bytes()).hexdigest()
    dataset = lance.dataset(source_lance)
    return {
        "path": str(source_lance.resolve()),
        "row_count": dataset.count_rows(),
        "schema": str(dataset.schema),
        "dataset_version": dataset.version,
        "root_manifest_sha256": manifest_sha256,
    }


def _base_eligible(
    sequence: object, precursor_mz: object, charge: object, *, max_charge: int
) -> tuple[bool, str | None]:
    if not isinstance(sequence, str) or not sequence:
        return False, "missing_sequence"
    try:
        mz_value = float(precursor_mz)
    except (TypeError, ValueError):
        return False, "invalid_precursor_mz"
    if not math.isfinite(mz_value) or mz_value <= 0.0:
        return False, "invalid_precursor_mz"
    try:
        charge_value = int(charge)
    except (TypeError, ValueError):
        return False, "invalid_charge"
    if charge_value < 1 or charge_value > max_charge:
        return False, "unsupported_charge"
    return True, None


def _iter_source_batches(
    dataset: lance.LanceDataset, batch_size: int, *, columns: list[str] | None = SOURCE_COLUMNS
) -> Iterable[pa.RecordBatch]:
    yield from dataset.scanner(columns=columns, batch_size=batch_size).to_batches()


def _count_eligible_identities(
    dataset: lance.LanceDataset, *, max_charge: int, batch_size: int
) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    progress = tqdm(total=dataset.count_rows(), desc="Counting eligible peptide identities", unit="spectrum")
    for batch in _iter_source_batches(dataset, batch_size):
        sequences = batch.column("seq").to_pylist()
        mzs = batch.column("precursor_mz").to_pylist()
        charges = batch.column("precursor_charge").to_pylist()
        for sequence, mz, charge in zip(sequences, mzs, charges):
            valid, reason = _base_eligible(sequence, mz, charge, max_charge=max_charge)
            if valid:
                counts[str(sequence)] += 1
            else:
                excluded[str(reason)] += 1
        progress.update(batch.num_rows)
    progress.close()
    return counts, excluded


def _manifest_identities(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["peptide_id"])
    return {str(value) for value in table["peptide_id"].to_pylist() if value}


def _quantile(values: list[int], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "count": len(data),
        "min": min(data),
        "max": max(data),
        "mean": sum(data) / len(data),
        "median": float(median(data)),
        "p25": _quantile(data, 0.25),
        "p75": _quantile(data, 0.75),
        "p90": _quantile(data, 0.90),
    }


def _manifest_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("peptide_id", pa.string()),
            pa.field("source_row_index", pa.int64()),
            pa.field("sample_id", pa.string()),
            pa.field("peak_file", pa.string()),
            pa.field("scan_id", pa.int64()),
            pa.field("precursor_mz", pa.float64()),
            pa.field("precursor_charge", pa.int16()),
        ]
    )


def _dataset_name(name: str, seed: int) -> str:
    return f"bacterial_metric_train_peptides_{name}_seed{seed}"


def _write_datasets_and_manifests(
    dataset: lance.LanceDataset,
    selected: dict[str, set[str]],
    staging_root: Path,
    *,
    seed: int,
    max_charge: int,
    batch_size: int,
) -> tuple[dict[str, int], dict[str, Counter[str]]]:
    schema = _manifest_schema()
    manifest_writers = {
        name: pq.ParquetWriter(
            staging_root / f"{_dataset_name(name, seed)}.manifest.parquet", schema, compression="zstd"
        )
        for name, _ in FRACTIONS
    }
    lance_paths = {name: staging_root / f"{_dataset_name(name, seed)}.lance" for name, _ in FRACTIONS}
    lance_modes = {name: "create" for name, _ in FRACTIONS}
    row_counts = Counter()
    peptide_counts = {name: Counter() for name, _ in FRACTIONS}
    row_index = 0
    progress = tqdm(total=dataset.count_rows(), desc="Writing nested label manifests", unit="spectrum")
    try:
        for batch in _iter_source_batches(dataset, batch_size, columns=None):
            sequences = batch.column("seq").to_pylist()
            files = batch.column("peak_file").to_pylist()
            scan_ids = batch.column("scan_id").to_pylist()
            mzs = batch.column("precursor_mz").to_pylist()
            charges = batch.column("precursor_charge").to_pylist()
            records = {name: {field.name: [] for field in schema} for name, _ in FRACTIONS}
            selected_offsets = {name: [] for name, _ in FRACTIONS}
            for offset, (sequence, peak_file, scan_id, mz, charge) in enumerate(
                zip(sequences, files, scan_ids, mzs, charges)
            ):
                valid, _ = _base_eligible(sequence, mz, charge, max_charge=max_charge)
                current_row = row_index + offset
                if not valid:
                    continue
                peptide_id = str(sequence)
                for name, _ in FRACTIONS:
                    if peptide_id not in selected[name]:
                        continue
                    record = records[name]
                    record["peptide_id"].append(peptide_id)
                    record["source_row_index"].append(current_row)
                    record["sample_id"].append(f"{peak_file}::scan={scan_id}")
                    record["peak_file"].append(peak_file)
                    record["scan_id"].append(int(scan_id))
                    record["precursor_mz"].append(float(mz))
                    record["precursor_charge"].append(int(charge))
                    selected_offsets[name].append(offset)
                    row_counts[name] += 1
                    peptide_counts[name][peptide_id] += 1
            for name, _ in FRACTIONS:
                if records[name]["peptide_id"]:
                    manifest_writers[name].write_table(pa.table(records[name], schema=schema))
                    selected_batch = pa.Table.from_batches([batch]).take(
                        pa.array(selected_offsets[name], type=pa.int64())
                    )
                    lance.write_dataset(selected_batch, lance_paths[name], mode=lance_modes[name])
                    lance_modes[name] = "append"
            row_index += batch.num_rows
            progress.update(batch.num_rows)
    finally:
        progress.close()
        for writer in manifest_writers.values():
            writer.close()
    return dict(row_counts), peptide_counts


def _readme() -> str:
    return """# Bacterial supervised metric-learning label budgets

These Lance datasets expose 1%, 10%, and 100% of **exact modified-peptide
identities** from the active bacterial training Lance split.  Identity is the
source `seq` string exactly as stored; no modification notation is removed and
charge is not part of the label.  A single stable seeded permutation makes the
budgets nested: `1pct` is contained in `10pct`, which is contained in `100pct`.

Every selected identity retains every source spectrum that passes the recorded
base eligibility filter.  There is no spectrum-level subsampling after identity
selection.  Each `.lance` output retains the full source schema and is directly
consumable by the standard Lance datamodule.  The matching `.manifest.parquet`
stores `source_row_index`, `peak_file`, and `scan_id` for audit/reload.  The
script only reads the locked validation/test benchmark manifests to audit
leakage.  The test split is expected to have zero exact-sequence overlap;
validation is run-held-out and can overlap by peptide identity under the
established benchmark policy.
"""


def main(args: argparse.Namespace) -> None:
    if args.min_spectra_per_peptide < 1:
        raise ValueError("--min-spectra-per-peptide must be positive.")
    if args.max_charge < 1:
        raise ValueError("--max-charge must be positive.")
    for path in (args.source_lance, args.validation_manifest, args.test_manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; manifests are immutable by default.")
        shutil.rmtree(output_root)
    staging_root = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(staging_root)

    dataset = lance.dataset(args.source_lance)
    source_fingerprint = _source_fingerprint(args.source_lance)
    counts, base_excluded = _count_eligible_identities(
        dataset, max_charge=args.max_charge, batch_size=args.batch_size
    )
    eligible = {
        peptide_id: count
        for peptide_id, count in counts.items()
        if count >= args.min_spectra_per_peptide
    }
    if not eligible:
        raise ValueError("No identities satisfy the requested eligibility filter.")

    ranked_peptides = sorted(eligible, key=lambda peptide_id: _stable_rank(args.seed, peptide_id))
    selected: dict[str, set[str]] = {}
    selection_counts: dict[str, int] = {}
    for name, fraction in FRACTIONS:
        count = len(ranked_peptides) if fraction == 1.0 else max(1, math.floor(len(ranked_peptides) * fraction))
        selection_counts[name] = count
        selected[name] = set(ranked_peptides[:count])
    if not (selected["1pct"] <= selected["10pct"] <= selected["100pct"]):
        raise AssertionError("Nested peptide budget invariant failed.")

    validation_ids = _manifest_identities(args.validation_manifest)
    test_ids = _manifest_identities(args.test_manifest)
    staging_root.mkdir(parents=True)
    try:
        row_counts, selected_counts = _write_datasets_and_manifests(
            dataset,
            selected,
            staging_root,
            seed=args.seed,
            max_charge=args.max_charge,
            batch_size=args.batch_size,
        )
        summaries: dict[str, dict[str, Any]] = {}
        for name, fraction in FRACTIONS:
            validation_overlap = sorted(selected[name] & validation_ids)
            test_overlap = sorted(selected[name] & test_ids)
            if test_overlap:
                preview = ", ".join(test_overlap[:5])
                raise ValueError(f"Strict test leakage for {name}: {len(test_overlap)} identities; e.g. {preview}")
            summaries[name] = {
                "fraction": fraction,
                "selected_unique_peptide_identities": selection_counts[name],
                "retained_spectra": row_counts[name],
                "spectra_per_peptide": _distribution(selected_counts[name].values()),
                "validation_manifest_exact_identity_overlap": len(validation_overlap),
                "test_manifest_exact_identity_overlap": len(test_overlap),
                "dataset": f"{_dataset_name(name, args.seed)}.lance",
                "manifest": f"{_dataset_name(name, args.seed)}.manifest.parquet",
            }
            if len(selected_counts[name]) != selection_counts[name]:
                raise AssertionError(f"Selected identities without retained spectra in {name}.")

        summary = {
            "code_version": CODE_VERSION,
            "identity_definition": {
                "name": IDENTITY_DEFINITION,
                "source_field": "seq",
                "description": "Exact modified sequence string; no backbone normalization and no charge suffix.",
            },
            "seed": args.seed,
            "source": source_fingerprint,
            "base_eligibility": {
                "sequence": "non-empty string",
                "precursor_mz": "finite and > 0",
                "precursor_charge": f"integer in [1, {args.max_charge}]",
                "min_spectra_per_peptide_before_sampling": args.min_spectra_per_peptide,
            },
            "pool": {
                "base_eligible_unique_peptide_identities": len(counts),
                "base_eligible_spectra": sum(counts.values()),
                "eligible_unique_peptide_identities": len(eligible),
                "eligible_spectra": sum(eligible.values()),
                "excluded_identities_below_minimum": len(counts) - len(eligible),
                "excluded_spectra_below_minimum": sum(counts.values()) - sum(eligible.values()),
                "base_filter_excluded_spectra": dict(sorted(base_excluded.items())),
            },
            "audit_manifests": {
                "validation": str(args.validation_manifest.resolve()),
                "test": str(args.test_manifest.resolve()),
                "validation_policy": "Run-held-out PXD010000 view; exact peptide overlap is reported, not prohibited.",
                "test_policy": "Strict PXD010613 exact-modified-sequence-disjoint view; overlap is prohibited.",
            },
            "budgets": summaries,
        }
        _json_dump(staging_root / "summary.json", summary)
        (staging_root / "README.md").write_text(_readme())
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print("\nMetric-learning label-budget audit")
    print("budget  identities  spectra  val_seq_overlap  test_seq_overlap")
    for name, _ in FRACTIONS:
        item = summary["budgets"][name]
        print(
            f"{name:>6}  {item['selected_unique_peptide_identities']:>10,}"
            f"  {item['retained_spectra']:>7,}"
            f"  {item['validation_manifest_exact_identity_overlap']:>15,}"
            f"  {item['test_manifest_exact_identity_overlap']:>16,}"
        )
    print(f"Wrote immutable manifests: {output_root}")


if __name__ == "__main__":
    main(parse_args())
