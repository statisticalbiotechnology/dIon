#!/usr/bin/env python3
"""Materialize tokenizer-compatible de novo Kingdoms validation and test sets.

This is deliberately separate from the capped retrieval/pair artifacts.  It
reuses their run-disjoint assignment, retains every quality-qualified spectrum,
and writes standard dIon de novo Lance fields plus source provenance.

The output target is canonical numeric mass-delta notation accepted by the
PA1.1/MSKB tokenizer.  Unsupported modification notation is rejected and
audited; labels are never silently stripped or approximated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import lance
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_kingdoms_paper_benchmarks import (
    SOURCE_COLUMNS,
    _metadata_fingerprint,
    _valid_indices,
    _validation_acquisitions,
)
from src.data.unified_tokenizer import PeptideTokenizer


SCRIPT_VERSION = "kingdoms_denovo_benchmarks_v1"
DEFAULT_SOURCE_ROOT = Path("/path/to/data/kingdoms/processed")
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/denovo_kingdoms/run_disjoint_v1")
DEFAULT_TOKENIZER_MANIFEST = PROJECT_ROOT / "configs/tokenizers/pa11.json"
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYO")
DELTA_RE = r"[+-]\d+(?:\.\d+)?"
RESIDUE_RE = re.compile(rf"(?P<aa>[A-Z])(?P<mods>(?:{DELTA_RE})*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-manifest", type=Path, default=DEFAULT_TOKENIZER_MANIFEST)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _canonical_numeric_sequence(value: Any, tokenizer: PeptideTokenizer) -> str:
    """Normalize known spelling variants, then require exact PA1.1 tokens."""
    text = _as_text(value).replace(" ", "").replace("_", "")
    replacements = (
        (r"M\(ox\)", "M+15.995"),
        (r"M\[Oxidation\]", "M+15.995"),
        (r"C\[Carbamidomethyl\]", "C+57.021"),
        (r"([NQ])\[Deamidated\]", r"\1+0.984"),
        (r"S\[Phospho\]", "S+79.966"),
        (r"T\[Phospho\]", "T+79.966"),
        (r"Y\[Phospho\]", "Y+79.966"),
        (r"\[Acetyl\]-", "+42.011"),
        (r"\[Carbamyl\]-", "+43.006"),
        (r"\[Ammonia-loss\]-", "-17.027"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Bracket and parenthesis numeric notation are harmless spelling variants.
    text = re.sub(r"([A-Za-z])\[([+-]\d+(?:\.\d+)?)\]", r"\1\2", text)
    text = re.sub(r"([A-Za-z])\(([+-]\d+(?:\.\d+)?)\)", r"\1\2", text)
    text = text.upper()
    if not text or any(marker in text for marker in "[](){};"):
        raise ValueError(f"unsupported notation: {value!r}")

    prefix = re.match(rf"^(?:{DELTA_RE})*", text)
    assert prefix is not None
    position = prefix.end()
    nterm = text[:position]
    tokens: list[str] = [nterm] if nterm else []
    while position < len(text):
        match = RESIDUE_RE.match(text, position)
        if match is None or match.group("aa") not in VALID_RESIDUES:
            raise ValueError(f"unsupported notation: {value!r}")
        tokens.append(match.group())
        position = match.end()
    if len(tokens) == (1 if nterm else 0):
        raise ValueError(f"empty peptide: {value!r}")
    if any(token not in tokenizer.index for token in tokens):
        unknown = [token for token in tokens if token not in tokenizer.index]
        raise ValueError(f"token absent from tokenizer: {unknown!r} in {value!r}")
    # ``tokenize`` is a final guard against an implementation/config mismatch.
    tokenizer.tokenize("".join(tokens))
    return "".join(tokens)


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("peak_file", pa.string()),
            pa.field("scan_id", pa.int64()),
            pa.field("ms_level", pa.uint8()),
            pa.field("precursor_mz", pa.float64()),
            pa.field("precursor_charge", pa.int16()),
            pa.field("mz_array", pa.list_(pa.float32())),
            pa.field("intensity_array", pa.list_(pa.float32())),
            pa.field("title", pa.string()),
            pa.field("seq", pa.string()),
            pa.field("source_modified_sequence", pa.string()),
            pa.field("species", pa.string()),
            pa.field("raw_file", pa.string()),
            pa.field("source_row_index", pa.int64()),
            pa.field("quality_qvalue", pa.float64()),
            pa.field("split", pa.string()),
        ]
    )


def _compatible_indices(
    table: pa.Table,
    *,
    split: str,
    validation_raw_files: set[str],
    tokenizer: PeptideTokenizer,
    rejected: Counter[str],
) -> tuple[list[int], list[str]]:
    """Return retained source rows and normalized labels without touching peaks."""
    qualified = _valid_indices(table, validation_raw_files=validation_raw_files, split=split)
    labels = table["modified_sequence"].to_pylist()
    indices: list[int] = []
    sequences: list[str] = []
    for index in qualified:
        try:
            sequence = _canonical_numeric_sequence(labels[index], tokenizer)
        except ValueError as exc:
            rejected[str(exc).split(":", 1)[0]] += 1
            continue
        indices.append(index)
        sequences.append(sequence)
    return indices, sequences


def _output_table(source: pa.Table, *, sequences: list[str], source_indices: list[int], species: str, split: str) -> pa.Table:
    """Project Arrow columns directly so peak arrays never pass through Python."""
    count = source.num_rows
    projected = pa.table(
        {
            "peak_file": source["raw_file"],
            "scan_id": source["scan"],
            "ms_level": pa.array([2] * count, type=pa.uint8()),
            "precursor_mz": source["precursor_mass"],
            "precursor_charge": source["precursor_charge"],
            "mz_array": source["mz_array"],
            "intensity_array": source["intensity_array"],
            "title": source["title"],
            "seq": pa.array(sequences, type=pa.string()),
            "source_modified_sequence": source["modified_sequence"],
            "species": pa.array([species] * count, type=pa.string()),
            "raw_file": source["raw_file"],
            "source_row_index": pa.array(source_indices, type=pa.int64()),
            "quality_qvalue": source["Qvalue"],
            "split": pa.array([split] * count, type=pa.string()),
        }
    )
    return projected.cast(_schema(), safe=True)


def _write_split(
    paths: list[Path],
    destination: Path,
    *,
    split: str,
    seed: int,
    tokenizer: PeptideTokenizer,
    batch_size: int,
) -> tuple[dict[str, Any], Counter[str]]:
    created = False
    rows_written = 0
    rows_qualified = 0
    sequence_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    species_summary: dict[str, Any] = {}
    for path in tqdm(paths, desc=f"Materialize Kingdoms de novo {split}", unit="species"):
        species = path.stem
        table = pq.read_table(path, columns=[*SOURCE_COLUMNS, "scan", "title"])
        raw_files = {str(value) for value in table["raw_file"].to_pylist() if value}
        validation_raw_files = _validation_acquisitions(species, raw_files, seed)
        indices, sequences = _compatible_indices(
            table,
            split=split,
            validation_raw_files=validation_raw_files,
            tokenizer=tokenizer,
            rejected=rejected,
        )
        rows_qualified += len(_valid_indices(table, validation_raw_files=validation_raw_files, split=split))
        for offset in range(0, len(indices), batch_size):
            chunk_indices = indices[offset : offset + batch_size]
            chunk_sequences = sequences[offset : offset + batch_size]
            source = table.take(pa.array(chunk_indices, type=pa.int64()))
            output = _output_table(
                source,
                sequences=chunk_sequences,
                source_indices=chunk_indices,
                species=species,
                split=split,
            )
            lance.write_dataset(
                output,
                str(destination),
                mode="append" if created else "create",
                max_rows_per_file=100_000,
                max_rows_per_group=4096,
            )
            created = True
            rows_written += output.num_rows
        sequence_counts.update(sequences)
        species_summary[species] = {
            "source_metadata_sha256": _metadata_fingerprint(path),
            "validation_raw_files": sorted(validation_raw_files),
            "quality_qualified_rows": len(_valid_indices(table, validation_raw_files=validation_raw_files, split=split)),
            "tokenizer_compatible_rows": len(indices),
        }
    if not created:
        raise RuntimeError(f"No rows written to {destination}")
    return {
        "rows_quality_qualified": rows_qualified,
        "rows_written": rows_written,
        "unique_peptidoforms": len(sequence_counts),
        "species": species_summary,
    }, rejected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    paths = sorted(source.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No source Parquets found under {source}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}; use --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(args.tokenizer_manifest)

    summaries: dict[str, Any] = {}
    rejected_total: Counter[str] = Counter()
    for split, filename in (("validation", "validation.lance"), ("test", "test.lance")):
        summary, rejected = _write_split(paths, output / filename, split=split, seed=args.seed, tokenizer=tokenizer, batch_size=args.batch_size)
        summaries[split] = summary
        rejected_total.update(rejected)
    val_sequences = set(lance.dataset(str(output / "validation.lance")).to_table(columns=["seq"])["seq"].to_pylist())
    test_sequences = set(lance.dataset(str(output / "test.lance")).to_table(columns=["seq"])["seq"].to_pylist())
    manifest = {
        "script_version": SCRIPT_VERSION,
        "source_root": str(source),
        "source_split_protocol": "exact Kingdoms run-disjoint validation/test assignment from create_kingdoms_paper_benchmarks.py",
        "seed": args.seed,
        "tokenizer_manifest": str(args.tokenizer_manifest.resolve()),
        "tokenizer_manifest_sha256": _sha256(args.tokenizer_manifest),
        "sequence_notation": "canonical numeric mass-delta tokens accepted by the PA1.1/MSKB tokenizer",
        "quality_gate": "Qvalue <= 0.01 and chimeric == false; positive charge and precursor m/z",
        "splits": summaries,
        "validation_test_peptidoform_overlap": len(val_sequences & test_sequences),
        "rejected_rows_by_reason": dict(sorted(rejected_total.items())),
        "warning": "Splits are run-disjoint, not peptidoform-disjoint. This preserves the existing Kingdoms validation/test definition.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote Kingdoms de novo benchmarks: {output}")


if __name__ == "__main__":
    main()
