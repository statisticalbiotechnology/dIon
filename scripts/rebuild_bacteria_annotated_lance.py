#!/usr/bin/env python
"""Regenerate run-disjoint labelled bacterial Lance datasets from source MGFs.

The legacy ``annotated`` and ``annotated_v2`` directories are not used here:
they contain duplicated splits and PXD010613 labels processed through
``swap_C.py``.  This builder preserves the original ``SEQ`` text in the
authoritative annotated MGF files.  In particular, PXD010613 cysteines remain
bare ``C`` rather than being converted to ``C+57.021``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path(
    "/path/to/data/bacteria_PXD010000__PXD010613"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "annotated_regenerated_v1"
MIN_PEAKS = 10
WRITE_BATCH_SIZE = 10_000

SCHEMA = pa.schema(
    [
        pa.field("peak_file", pa.string()),
        pa.field("scan_id", pa.int64()),
        pa.field("ms_level", pa.uint8()),
        pa.field("precursor_mz", pa.float64()),
        pa.field("precursor_charge", pa.int16()),
        pa.field("mz_array", pa.list_(pa.float64())),
        pa.field("intensity_array", pa.list_(pa.float64())),
        pa.field("seq", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing no_labels and the authoritative annotated MGF sources.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="New output directory. It must not already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source/run membership without writing Lance datasets.",
    )
    return parser.parse_args()


def _lance_file_stems(path: Path) -> set[str]:
    """Return source-run stems without materializing an entire split table."""
    stems: set[str] = set()
    dataset = lance.dataset(str(path))
    scanner = dataset.scanner(columns=["peak_file"], batch_size=65_536)
    for batch in scanner.to_batches():
        stems.update(Path(value).stem for value in batch.column(0).to_pylist())
    return stems


def _source_files(source_dir: Path) -> dict[str, Path]:
    files = {path.stem: path for path in source_dir.glob("*.mgf")}
    if not files:
        raise FileNotFoundError(f"No MGF files found in {source_dir}")
    if len(files) != len(list(source_dir.glob("*.mgf"))):
        raise ValueError(f"Duplicate MGF stems found in {source_dir}")
    return files


def _parse_charge(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"-?\d+", value)
    return int(match.group()) if match else None


def _parse_scan_id(headers: dict[str, str]) -> int | None:
    title = headers.get("TITLE", "")
    match = re.search(r"(?:^|\s)scan=(\d+)(?:\s|$)", title)
    if match:
        return int(match.group(1))

    scans = headers.get("SCANS", "")
    match = re.search(r"\d+", scans)
    return int(match.group()) if match else None


def _record_from_spectrum(
    source_path: Path,
    headers: dict[str, str],
    mz_values: list[float],
    intensity_values: list[float],
    counters: Counter[str],
) -> dict[str, Any] | None:
    sequence = headers.get("SEQ")
    charge = _parse_charge(headers.get("CHARGE"))
    scan_id = _parse_scan_id(headers)
    pepmass = headers.get("PEPMASS", "").split()

    if not sequence or charge is None or scan_id is None or not pepmass:
        counters["skipped_missing_required_header"] += 1
        return None
    if charge < 1:
        counters["skipped_invalid_charge"] += 1
        return None
    if len(mz_values) < MIN_PEAKS:
        counters["skipped_min_peaks"] += 1
        return None

    try:
        precursor_mz = float(pepmass[0])
    except ValueError:
        counters["skipped_invalid_pepmass"] += 1
        return None

    if len(mz_values) != len(intensity_values):
        raise ValueError(f"Mismatched peak arrays in {source_path}")

    counters["written"] += 1
    counters["sequences_with_bare_c"] += int("C" in sequence and "C+" not in sequence)
    counters["sequences_with_carbamidomethyl_c"] += int("C+57.021" in sequence)
    return {
        "peak_file": source_path.name,
        "scan_id": scan_id,
        "ms_level": 2,
        "precursor_mz": precursor_mz,
        "precursor_charge": charge,
        "mz_array": mz_values,
        "intensity_array": intensity_values,
        "seq": sequence,
    }


def iter_mgf_records(
    source_path: Path, counters: Counter[str]
) -> Iterator[dict[str, Any]]:
    """Yield eligible spectra from one simple text MGF file without relabelling."""
    headers: dict[str, str] = {}
    mz_values: list[float] = []
    intensity_values: list[float] = []
    in_spectrum = False

    with source_path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                if in_spectrum:
                    raise ValueError(f"Nested BEGIN IONS in {source_path}")
                headers = {}
                mz_values = []
                intensity_values = []
                in_spectrum = True
                counters["total_mgf_spectra"] += 1
                continue
            if line == "END IONS":
                if not in_spectrum:
                    raise ValueError(f"END IONS without BEGIN IONS in {source_path}")
                record = _record_from_spectrum(
                    source_path, headers, mz_values, intensity_values, counters
                )
                if record is not None:
                    yield record
                in_spectrum = False
                continue
            if not in_spectrum:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                headers[key.upper()] = value
                continue

            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid peak line in {source_path}: {line!r}")
            try:
                mz_values.append(float(fields[0]))
                intensity_values.append(float(fields[1]))
            except ValueError as error:
                raise ValueError(
                    f"Invalid peak values in {source_path}: {line!r}"
                ) from error

    if in_spectrum:
        raise ValueError(f"Unterminated spectrum in {source_path}")


def _write_record_batches(
    records: Iterable[dict[str, Any]],
    destination: Path,
) -> int:
    buffer: list[dict[str, Any]] = []
    mode = "create"
    written = 0
    for record in records:
        buffer.append(record)
        if len(buffer) == WRITE_BATCH_SIZE:
            lance.write_dataset(
                pa.RecordBatch.from_pylist(buffer, schema=SCHEMA),
                str(destination),
                schema=SCHEMA,
                mode=mode,
            )
            mode = "append"
            written += len(buffer)
            buffer.clear()
    if buffer:
        lance.write_dataset(
            pa.RecordBatch.from_pylist(buffer, schema=SCHEMA),
            str(destination),
            schema=SCHEMA,
            mode=mode,
        )
        written += len(buffer)
    if written == 0:
        raise ValueError(f"No eligible spectra were written to {destination}")
    return written


def _validate_split_file_assignment(
    split: str,
    raw_stems: set[str],
    source_files: dict[str, Path],
) -> list[Path]:
    source_stems = set(source_files)
    missing = raw_stems - source_stems
    extra = source_stems & raw_stems
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{split} has raw runs absent from annotated MGF source: {preview}")
    if not extra:
        raise ValueError(f"{split} selected no annotated MGF source files")
    return [source_files[stem] for stem in sorted(raw_stems)]


def _split_source_plan(data_root: Path) -> dict[str, dict[str, Any]]:
    """Map source MGFs to the canonical raw-run split without scanning 9M rows.

    PXD010000 contains all 235 labelled source runs.  The canonical no-label
    validation split names 12 of them; its complement is the canonical
    PXD010000 training set.  PXD010613 is reserved entirely for test and is
    checked against the no-label test split exactly.
    """
    no_labels = data_root / "no_labels"
    source_000 = _source_files(data_root / "bacteria_PXD010000_annotated")
    source_613 = _source_files(data_root / "bacteria_PXD010613_annotated")

    val_path = no_labels / "val.lance"
    test_path = no_labels / "test.lance"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected canonical no_labels val.lance and test.lance")

    val_stems = _lance_file_stems(val_path)
    test_stems = _lance_file_stems(test_path)
    val_files = _validate_split_file_assignment("val", val_stems, source_000)
    test_files = _validate_split_file_assignment("test", test_stems, source_613)
    if test_stems != set(source_613):
        missing = sorted(set(source_613) - test_stems)
        unexpected = sorted(test_stems - set(source_613))
        raise ValueError(
            "PXD010613 source/test run mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    train_stems = set(source_000) - val_stems
    if not train_stems:
        raise ValueError("No PXD010000 source runs remain after validation assignment")
    if val_stems & set(source_613):
        raise ValueError("Validation split unexpectedly contains PXD010613 source runs")

    return {
        "train": {
            "project": "PXD010000",
            "raw_split": no_labels / "train.lance",
            "raw_stems": train_stems,
            "files": [source_000[stem] for stem in sorted(train_stems)],
        },
        "val": {
            "project": "PXD010000",
            "raw_split": val_path,
            "raw_stems": val_stems,
            "files": val_files,
        },
        "test": {
            "project": "PXD010613",
            "raw_split": test_path,
            "raw_stems": test_stems,
            "files": test_files,
        },
    }


def _build_split(split: str, split_plan: dict[str, Any], destination: Path) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    sequence_ids: set[str] = set()

    def records() -> Iterator[dict[str, Any]]:
        for source_path in tqdm(
            split_plan["files"], desc=f"Building {split}", unit="file"
        ):
            for record in iter_mgf_records(source_path, counters):
                sequence_ids.add(record["seq"])
                yield record

    written = _write_record_batches(records(), destination)
    dataset = lance.dataset(str(destination))
    if dataset.count_rows() != written:
        raise RuntimeError(
            f"{split} row-count mismatch: wrote {written}, Lance has {dataset.count_rows()}"
        )
    if dataset.schema != SCHEMA:
        raise RuntimeError(f"{split} schema mismatch: {dataset.schema}")
    return {
        "source_project": split_plan["project"],
        "source_files": [path.name for path in split_plan["files"]],
        "source_file_count": len(split_plan["files"]),
        "raw_run_count": len(split_plan["raw_stems"]),
        "rows": written,
        "unique_sequences": len(sequence_ids),
        "sequence_ids": sequence_ids,
        "filter_counts": dict(sorted(counters.items())),
    }


def _manifest(
    data_root: Path,
    output_root: Path,
    split_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    split_names = tuple(split_results)
    raw_run_overlap = {
        f"{left}_{right}": 0
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    sequence_overlap = {
        f"{left}_{right}": len(
            split_results[left]["sequence_ids"] & split_results[right]["sequence_ids"]
        )
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    }
    splits = {}
    for name, result in split_results.items():
        result = dict(result)
        result.pop("sequence_ids")
        splits[name] = result

    return {
        "dataset_name": "bacteria_annotated_regenerated_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/rebuild_bacteria_annotated_lance.py",
        "output_root": str(output_root),
        "canonical_no_label_root": str(data_root / "no_labels"),
        "source_directories": {
            "PXD010000": str(data_root / "bacteria_PXD010000_annotated"),
            "PXD010613": str(data_root / "bacteria_PXD010613_annotated"),
        },
        "sequence_policy": {
            "description": "Preserve source MGF SEQ text exactly; do not apply swap_C.py.",
            "pxd010613_cysteine": "bare C from bacteria_PXD010613_annotated",
            "excluded_legacy_sources": [
                "bacteria_PXD010613_annotated_preprocessed",
                "bacteria_PXD010613_annotated_MSKB_notation",
            ],
        },
        "eligibility_filter": {
            "minimum_peaks": MIN_PEAKS,
            "precursor_charge": "all positive annotated charge states",
        },
        "split_policy": {
            "train": "PXD010000 raw runs assigned to no_labels/train.lance",
            "val": "PXD010000 raw runs assigned to no_labels/val.lance",
            "test": "PXD010613 raw runs assigned to no_labels/test.lance",
            "raw_run_overlap": raw_run_overlap,
            "raw_run_split_disjoint": True,
            "sequence_overlap_counts": sequence_overlap,
        },
        "splits": splits,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    plan = _split_source_plan(data_root)
    print(
        "Validated source/run assignment: "
        + ", ".join(
            f"{split}={len(details['files'])} MGF files ({details['project']})"
            for split, details in plan.items()
        )
    )
    if args.dry_run:
        return
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_root}"
        )

    staging_root = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_root}")
    staging_root.mkdir(parents=True)
    try:
        split_results = {
            split: _build_split(split, details, staging_root / f"{split}.lance")
            for split, details in plan.items()
        }
        manifest = _manifest(data_root, output_root, split_results)
        (staging_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(f"Wrote regenerated labelled Lance splits: {output_root}")


if __name__ == "__main__":
    main()
