#!/usr/bin/env python3
"""Build dIon-de-novo-labeled-v1 (DNLv1) by replacing V4's legacy MSKB component.

This is deliberately a replacement build, not a raw-source rebuild. It copies
all final KitchenSink V4 rows except the legacy MSKB component, then adds the
official ``mskb_final`` train/test source. Official test is retained as test;
a deterministic peptidoform-level validation carve-out is selected from the
official training split. The preserved V3 cleanup then enforces its usual
``test > val > train`` peptidoform precedence and exact-m/z duplicate policy.

The script never modifies V4 or the official MSKB source. Existing output is
refused unless ``--force`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import lance
import pyarrow as pa
from tqdm import tqdm


SCRIPT_VERSION = "dnlv1_materialize_v1"
SPLITS = ("train", "val", "test")
LEGACY_V4_MSKB_SOURCE_ID = "mskb_v5_seed"
DEFAULT_V4_ROOT = Path("/path/to/data/denovo_kitchensink_v4/lance")
DEFAULT_MSKB_ROOT = Path(
    "/path/to/data/casanovo_official/v5_mskb_final/lance_no_peak_cap"
)
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/denovo_dnlv1")
REFERENCE_ROOT = Path(__file__).resolve().parent / "legacy/kitchensink_v3_reference"

KNOWN_EXT_RE = re.compile(r"\.(?:raw|mzml|mzxml|mgf)$", re.IGNORECASE)
PXD_PREFIX_RE = re.compile(r"^pxd\d+_", re.IGNORECASE)
TITLE_SCAN_RE = re.compile(r"^(?P<run>.+?):scan:(?P<scan>\d+)$", re.IGNORECASE)
RESIDUE_TOKEN_RE = re.compile(r"(?P<aa>[A-Z])(?P<mods>(?:[+-]\d+(?:\.\d+)?)*)")
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("materialize", "cleanup", "all"), default="all")
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--mskb-root", type=Path, default=DEFAULT_MSKB_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-target-spectra", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--rows-per-write", type=int, default=50_000)
    parser.add_argument("--force", action="store_true", help="Remove a prior incomplete output root.")
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "nan", "none", "null"} else value


def _canonicalize_label(value: Any) -> str:
    """Match the V3 canonical peptidoform representation exactly."""
    text = _as_text(value).replace(" ", "").replace("_", "")
    text = re.sub(r"M\(ox\)", "M+15.995", text, flags=re.IGNORECASE)
    text = re.sub(r"M\[Oxidation\]", "M+15.995", text, flags=re.IGNORECASE)
    text = re.sub(r"C\[Carbamidomethyl\]", "C+57.021", text, flags=re.IGNORECASE)
    text = re.sub(r"([NQ])\[Deamidated\]", r"\1+0.984", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Acetyl\]-", "+42.011", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Carbamyl\]-", "+43.006", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Ammonia-loss\]-", "-17.027", text, flags=re.IGNORECASE)
    text = text.upper()
    if not text or any(mark in text for mark in "[](){};"):
        raise ValueError(f"Unsupported official mskb_final sequence notation: {value!r}")
    prefix = re.match(r"^(?:[+-]\d+(?:\.\d+)?)*", text)
    assert prefix is not None
    position = prefix.end()
    residues = 0
    while position < len(text):
        match = RESIDUE_TOKEN_RE.match(text, position)
        if match is None or match.group("aa") not in VALID_RESIDUES:
            raise ValueError(f"Unsupported official mskb_final sequence notation: {value!r}")
        residues += 1
        position = match.end()
    if not residues:
        raise ValueError(f"Empty official mskb_final sequence: {value!r}")
    return text


def _stable_u63(*parts: Any) -> int:
    value = "\x1f".join(_as_text(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _normalize_run(value: Any) -> str:
    text = Path(_as_text(value)).name
    if text.lower().endswith(".gz"):
        text = text[:-3]
    text = KNOWN_EXT_RE.sub("", text).lower()
    return PXD_PREFIX_RE.sub("", text)


def _parse_title(value: Any) -> tuple[str, int]:
    match = TITLE_SCAN_RE.match(_as_text(value))
    if match is None:
        raise ValueError(f"Unparseable official mskb_final title: {value!r}")
    return match.group("run"), int(match.group("scan"))


def _physical_key(run_file: str, scan: int, charge: int, mz: float) -> str:
    run = _normalize_run(run_file)
    if not run:
        raise ValueError(f"Cannot construct physical key without run: {run_file!r}")
    return f"{run}|scan={scan}|z={charge}|mz={mz:.2f}"


def _selection(counts: dict[str, int], target: int, seed: int) -> tuple[set[str], int]:
    selected: set[str] = set()
    rows = 0
    for seq in sorted(
        counts,
        key=lambda value: (_stable_u63("dnlv1_mskb_final_validation", seed, value), value),
    ):
        if rows >= target:
            break
        selected.add(seq)
        rows += counts[seq]
    return selected, rows


def _iter_official(
    root: Path, split: str, batch_size: int, columns: list[str]
) -> Iterator[pa.RecordBatch]:
    dataset = lance.dataset(str(root / f"{split}.lance"))
    yield from tqdm(
        dataset.to_batches(columns=columns, batch_size=batch_size),
        desc=f"official mskb_final {split}", unit="batch", mininterval=2.0,
    )


def _derive_membership(mskb_root: Path, seed: int, target: int, batch_size: int) -> tuple[set[str], set[str], set[str], dict[str, int]]:
    test_sequences: set[str] = set()
    for batch in _iter_official(mskb_root, "test", batch_size, ["seq"]):
        test_sequences.update(_canonicalize_label(value) for value in batch.column(0).to_pylist())

    train_counts: dict[str, int] = defaultdict(int)
    overlap_rows = 0
    for batch in _iter_official(mskb_root, "train", batch_size, ["seq"]):
        for value in batch.column(0).to_pylist():
            seq = _canonicalize_label(value)
            if seq in test_sequences:
                overlap_rows += 1
            else:
                train_counts[seq] += 1
    if overlap_rows:
        raise RuntimeError(f"Official mskb_final train/test peptidoform overlap: {overlap_rows} rows")

    validation, validation_rows = _selection(train_counts, target, seed)
    train = set(train_counts) - validation
    return train, validation, test_sequences, {
        "official_test_unique_peptidoforms": len(test_sequences),
        "official_eligible_train_unique_peptidoforms": len(train_counts),
        "validation_unique_peptidoforms": len(validation),
        "validation_rows": validation_rows,
        "train_unique_peptidoforms": len(train),
        "train_rows": sum(train_counts[seq] for seq in train),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append(path: Path, table: pa.Table, created: set[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(
        table, str(path), mode="append" if path in created else "create",
        max_rows_per_file=100_000, max_rows_per_group=4096,
    )
    created.add(path)


def _mskb_table(
    batch: pa.RecordBatch, row_offset: int, split: str, schema: pa.Schema, mskb_root: Path
) -> pa.Table:
    values = {name: batch.column(index).to_pylist() for index, name in enumerate(batch.schema.names)}
    rows: list[dict[str, Any]] = []
    source_path = str(mskb_root / f"{split}.lance")
    for index in range(batch.num_rows):
        run_file, scan = _parse_title(values["title"][index])
        charge = int(values["precursor_charge"][index])
        precursor_mz = float(values["precursor_mz"][index])
        label = _as_text(values["seq"][index])
        rows.append({
            "peak_file": _as_text(values["peak_file"][index]) or run_file,
            "scan_id": scan,
            "ms_level": int(values["ms_level"][index]),
            "precursor_mz": precursor_mz,
            "precursor_charge": charge,
            "mz_array": values["mz_array"][index],
            "intensity_array": values["intensity_array"][index],
            "seq": _canonicalize_label(label),
            "source_dataset": "mskb_final",
            "source_id": "mskb_final",
            "project_accession": "",
            "msv_accession": "",
            "run_file": run_file,
            "physical_scan_id": scan,
            "source_path": source_path,
            "source_row": row_offset + index,
            "source_label": label,
            "source_quality": None,
            "quality_metric": "official_mskb_final",
            "experiment": "mskb_final",
            "species": "",
            "enzyme_class": "mixed_tryptic_nontryptic",
            "is_chimeric": False,
            "physical_key": _physical_key(run_file, scan, charge, precursor_mz),
            "split_origin": f"official_mskb_final_{split}",
        })
    return pa.Table.from_pylist(rows, schema=schema)


def _validate_inputs(v4_root: Path, mskb_root: Path) -> pa.Schema:
    for split in SPLITS:
        if not (v4_root / f"{split}.lance").is_dir():
            raise FileNotFoundError(v4_root / f"{split}.lance")
    for split in ("train", "test"):
        if not (mskb_root / f"{split}.lance").is_dir():
            raise FileNotFoundError(mskb_root / f"{split}.lance")
    schema = lance.dataset(str(v4_root / "train.lance")).schema
    required = {"seq", "source_id", "source_dataset", "physical_key"}
    if not required <= set(schema.names):
        raise RuntimeError(f"Unexpected V4 schema; missing {sorted(required - set(schema.names))}")
    return schema


def materialize(args: argparse.Namespace) -> None:
    v4_root = args.v4_root.resolve()
    mskb_root = args.mskb_root.resolve()
    output_root = args.output_root.resolve()
    schema = _validate_inputs(v4_root, mskb_root)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output root exists: {output_root}; refusing to alter it without --force")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    train_sequences, validation_sequences, test_sequences, membership_summary = _derive_membership(
        mskb_root, args.seed, args.validation_target_spectra, args.batch_size
    )
    created: set[Path] = set()
    written: Counter[str] = Counter()
    excluded_legacy: Counter[str] = Counter()

    # Copy final V4 rows verbatim except its old MSKB component. The subsequent
    # V3 cleanup handles cross-source heldout peptidoform and duplicate removal.
    for split in SPLITS:
        dataset = lance.dataset(str(v4_root / f"{split}.lance"))
        output = output_root / "lance" / f"{split}.lance"
        for batch in tqdm(dataset.to_batches(batch_size=args.batch_size), desc=f"copy V4 {split}", unit="batch", mininterval=2.0):
            source_ids = batch.column(batch.schema.get_field_index("source_id")).to_pylist()
            source_datasets = batch.column(batch.schema.get_field_index("source_dataset")).to_pylist()
            keep = [
                source_id != LEGACY_V4_MSKB_SOURCE_ID and source_dataset != LEGACY_V4_MSKB_SOURCE_ID
                for source_id, source_dataset in zip(source_ids, source_datasets)
            ]
            excluded_legacy[split] += len(keep) - sum(keep)
            if any(keep):
                table = pa.Table.from_batches([batch]).filter(pa.array(keep, type=pa.bool_()))
                _append(output, table, created)
                written[split] += table.num_rows

    # Add official source with train-derived validation membership.
    columns = ["peak_file", "ms_level", "precursor_mz", "precursor_charge", "mz_array", "intensity_array", "title", "seq"]
    official_written: Counter[str] = Counter()
    for official_split in ("train", "test"):
        row_offset = 0
        for batch in _iter_official(mskb_root, official_split, args.batch_size, columns):
            table = _mskb_table(batch, row_offset, official_split, schema, mskb_root)
            row_offset += batch.num_rows
            seqs = table.column("seq").to_pylist()
            if official_split == "test":
                destination = "test"
                if any(seq not in test_sequences for seq in seqs):
                    raise RuntimeError("Official test membership mismatch")
                _append(output_root / "lance" / "test.lance", table, created)
                written["test"] += table.num_rows
                official_written["test"] += table.num_rows
                continue
            for destination, allowed in (("val", validation_sequences), ("train", train_sequences)):
                keep = [seq in allowed for seq in seqs]
                if any(keep):
                    filtered = table.filter(pa.array(keep, type=pa.bool_()))
                    _append(output_root / "lance" / f"{destination}.lance", filtered, created)
                    written[destination] += filtered.num_rows
                    official_written[destination] += filtered.num_rows
            if any(seq not in validation_sequences and seq not in train_sequences for seq in seqs):
                raise RuntimeError("Official train membership mismatch")

    cleanup_config = json.loads((REFERENCE_ROOT / "config.public.json").read_text())
    cleanup_config["max_output_gib"] = 110
    cleanup_config["sources"] = [
        {"id": "mskb_final", "source_priority": 0},
        *[
            {"id": source["id"], "source_priority": source["source_priority"]}
            for source in cleanup_config["sources"]
            if source.get("source_priority") is not None
        ],
    ]
    _write_json(output_root / "cleanup_config.json", cleanup_config)
    _write_json(output_root / "build_summary_pre_cleanup.json", {
        "script_version": SCRIPT_VERSION,
        "mode": "replacement_materialization_before_v3_cleanup",
        "v4_root": str(v4_root),
        "official_mskb_final_root": str(mskb_root),
        "legacy_v4_component_removed": LEGACY_V4_MSKB_SOURCE_ID,
        "validation_selection": {
            "seed": args.seed,
            "unit": "canonical_peptidoform",
            "target_spectra": args.validation_target_spectra,
            "selection": "stable_hash_ranked_peptidoforms_accumulated_to_target",
        },
        "membership": membership_summary,
        "rows_written_pre_cleanup": dict(written),
        "official_mskb_final_rows_written": dict(official_written),
        "legacy_component_rows_excluded_from_v4": dict(excluded_legacy),
        "cleanup": "required; preserved v3 test>val>train peptidoform and exact-mz policy",
    })


def cleanup(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    input_root = output_root / "lance"
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing pre-cleanup Lance root: {input_root}")
    command = [
        sys.executable, str(REFERENCE_ROOT / "cleanup_kitchensink_v3_lance.py"),
        "--input-root", str(input_root), "--output-root", str(output_root),
        "--config", str(output_root / "cleanup_config.json"),
        "--batch-size", str(args.batch_size), "--rows-per-write", str(args.rows_per_write), "--replace",
    ]
    if args.force:
        command.append("--force")
    print("Running preserved V3 cleanup:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.validation_target_spectra < 1 or args.batch_size < 1 or args.rows_per_write < 1:
        raise ValueError("Batch sizes and validation target must be positive")
    if args.stage in {"materialize", "all"}:
        materialize(args)
    if args.stage in {"cleanup", "all"}:
        cleanup(args)


if __name__ == "__main__":
    main()
