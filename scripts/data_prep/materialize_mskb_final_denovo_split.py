#!/usr/bin/env python3
"""Materialize a standalone official MSKB-final de novo train/val/test corpus.

The official MSKB-final source has released ``train`` and ``test`` splits but
no validation split. This script deterministically carves validation
*peptidoforms* from official train and removes all spectra of those identities
from derived train. Official test is copied without reading or filtering it;
the copied Lance directory is byte-identical to the released source directory.

Identity definition: canonical Casanovo numeric-mass-delta peptidoform. It
normalizes whitespace/underscore spelling and known named modifications before
selection. Rows retain their original ``seq`` values so Casanovo-compatible
training still sees the official labels exactly as released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import lance
import pyarrow as pa
from tqdm import tqdm


SCRIPT_VERSION = "mskb_final_denovo_split_v1"
DEFAULT_SOURCE_ROOT = Path(
    "/path/to/data/casanovo_official/v5_mskb_final/lance_no_peak_cap"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/path/to/data/denovo_mskb_final/lance_peptidoform_val10k_seed42"
)
KNOWN_EXT_RE = re.compile(r"\.(?:raw|mzml|mzxml|mgf)$", re.IGNORECASE)
RESIDUE_TOKEN_RE = re.compile(r"(?P<aa>[A-Z])(?P<mods>(?:[+-]\d+(?:\.\d+)?)*)")
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-target-spectra",
        type=int,
        default=10_000,
        help="Target before whole-peptidoform rounding; default is 100 batches at batch size 100.",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _canonical_peptidoform(value: Any) -> str:
    """Canonical identity only; do not replace the released row label."""
    text = _as_text(value).replace(" ", "").replace("_", "")
    text = re.sub(r"M\(ox\)", "M+15.995", text, flags=re.IGNORECASE)
    text = re.sub(r"M\[Oxidation\]", "M+15.995", text, flags=re.IGNORECASE)
    text = re.sub(r"C\[Carbamidomethyl\]", "C+57.021", text, flags=re.IGNORECASE)
    text = re.sub(r"([NQ])\[Deamidated\]", r"\1+0.984", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Acetyl\]-", "+42.011", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Carbamyl\]-", "+43.006", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Ammonia-loss\]-", "-17.027", text, flags=re.IGNORECASE)
    text = text.upper()
    if not text or any(marker in text for marker in "[](){};"):
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
    encoded = "\x1f".join(_as_text(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _iter_batches(path: Path, *, columns: list[str] | None, batch_size: int, desc: str) -> Iterator[pa.RecordBatch]:
    dataset = lance.dataset(str(path))
    yield from tqdm(dataset.to_batches(columns=columns, batch_size=batch_size), desc=desc, unit="batch", mininterval=1.0)


def _select_validation(train_lance: Path, seed: int, target: int, batch_size: int) -> tuple[set[str], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for batch in _iter_batches(train_lance, columns=["seq"], batch_size=batch_size, desc="Count official train peptidoforms"):
        for sequence in batch.column(0).to_pylist():
            counts[_canonical_peptidoform(sequence)] += 1
    selected: set[str] = set()
    selected_rows = 0
    for peptidoform in sorted(counts, key=lambda value: (_stable_u63("mskb_final_denovo_validation", seed, value), value)):
        if selected_rows >= target:
            break
        selected.add(peptidoform)
        selected_rows += counts[peptidoform]
    return selected, {
        "official_train_unique_peptidoforms": len(counts),
        "validation_unique_peptidoforms": len(selected),
        "validation_rows": selected_rows,
        "train_unique_peptidoforms": len(counts) - len(selected),
        "train_rows": sum(rows for key, rows in counts.items() if key not in selected),
    }


def _write_split(source: Path, destination: Path, validation: set[str], wanted_validation: bool, batch_size: int) -> int:
    written = 0
    created = False
    for batch in _iter_batches(source, columns=None, batch_size=batch_size, desc=f"Write {'val' if wanted_validation else 'train'}"):
        membership = [_canonical_peptidoform(value) in validation for value in batch.column(batch.schema.get_field_index("seq")).to_pylist()]
        indices = [index for index, is_validation in enumerate(membership) if is_validation == wanted_validation]
        if not indices:
            continue
        table = pa.Table.from_batches([batch.take(pa.array(indices, type=pa.int64()))])
        lance.write_dataset(table, str(destination), mode="append" if created else "create", max_rows_per_file=100_000, max_rows_per_group=4096)
        created = True
        written += table.num_rows
    if not created:
        raise RuntimeError(f"No rows selected for {destination.name}")
    return written


def _write_manifest(output: Path, payload: dict[str, Any]) -> None:
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.validation_target_spectra < 1 or args.batch_size < 1:
        raise ValueError("validation-target-spectra and batch-size must be positive")
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    train_source = source / "train.lance"
    test_source = source / "test.lance"
    for path in (train_source, test_source):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Refusing existing output: {output}; pass --force to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    validation, counts = _select_validation(train_source, args.seed, args.validation_target_spectra, args.batch_size)
    validation_path = output / "validation_peptidoforms.txt"
    validation_path.write_text("".join(f"{identity}\n" for identity in sorted(validation)))

    train_written = _write_split(train_source, output / "train.lance", validation, False, args.batch_size)
    val_written = _write_split(train_source, output / "val.lance", validation, True, args.batch_size)
    # Preserve the official test directory as bytes, not merely equivalent rows.
    shutil.copytree(test_source, output / "test.lance")

    train_peptides = set()
    for batch in _iter_batches(output / "train.lance", columns=["seq"], batch_size=args.batch_size, desc="Audit train peptidoforms"):
        train_peptides.update(_canonical_peptidoform(value) for value in batch.column(0).to_pylist())
    test_peptides = set()
    for batch in _iter_batches(output / "test.lance", columns=["seq"], batch_size=args.batch_size, desc="Audit official test peptidoforms"):
        test_peptides.update(_canonical_peptidoform(value) for value in batch.column(0).to_pylist())
    if train_peptides & validation or train_peptides & test_peptides or validation & test_peptides:
        raise RuntimeError("Peptidoform leakage detected after materialization.")
    if lance.dataset(str(output / "test.lance")).count_rows() != lance.dataset(str(test_source)).count_rows():
        raise RuntimeError("Copied official test row count differs from source.")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "source_root": str(source),
        "seed": args.seed,
        "validation_target_spectra": args.validation_target_spectra,
        "identity_definition": "canonical Casanovo numeric-mass-delta peptidoform; charge is not part of identity",
        "selection": "one seeded stable hash ordering of official-train identities; retain all spectra for selected identities",
        "official_test": "copied byte-for-byte from the released source; never used to select validation",
        "counts": {**counts, "train_rows_written": train_written, "val_rows_written": val_written, "official_test_rows": lance.dataset(str(test_source)).count_rows()},
        "disjointness": {"train_val_peptidoforms": 0, "train_test_peptidoforms": 0, "val_test_peptidoforms": 0},
        "hashes": {
            "source_train_tree_sha256": _tree_hash(train_source),
            "source_test_tree_sha256": _tree_hash(test_source),
            "copied_test_tree_sha256": _tree_hash(output / "test.lance"),
            "validation_peptidoforms_sha256": _sha256_file(validation_path),
        },
    }
    _write_manifest(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote standalone MSKB-final de novo corpus: {output}")


if __name__ == "__main__":
    main()
