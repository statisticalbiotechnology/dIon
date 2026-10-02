#!/usr/bin/env python3
"""Audit a dIon-de-novo-labeled-v1 (DNLv1) candidate without writing any Lance dataset.

dIon-de-novo-labeled-v1 (DNLv1) reuses final KitchenSink v4 rows from every non-mskb_final
source. It replaces only the legacy ``mskb_v5_seed`` component with the
official mskb_final train/test source: official test remains test, while a
deterministic peptidoform-level validation set is carved from official train.

The report quantifies v4 rows that conflict with the new held-out mskb_final
membership under the original v3 source-resolution policy. No dataset rows are
created, modified, or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import lance
from tqdm import tqdm


SCRIPT_VERSION = "dnlv1_audit_v1"
SPLITS = ("train", "val", "test")
DEFAULT_V4_ROOT = Path("/path/to/data/denovo_kitchensink_v4/lance")
DEFAULT_MSKB_ROOT = Path(
    "/path/to/data/casanovo_official/v5_mskb_final/lance_no_peak_cap"
)
DEFAULT_REPORT = Path("/tmp/dnlv1_audit.json")

KNOWN_EXT_RE = re.compile(r"\.(?:raw|mzml|mzxml|mgf)$", re.IGNORECASE)
PXD_PREFIX_RE = re.compile(r"^pxd\d+_", re.IGNORECASE)
TITLE_SCAN_RE = re.compile(r"^(?P<run>.+?):scan:(?P<scan>\d+)$", re.IGNORECASE)
RESIDUE_TOKEN_RE = re.compile(r"(?P<aa>[A-Z])(?P<mods>(?:[+-]\d+(?:\.\d+)?)*)")
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--mskb-root", type=Path, default=DEFAULT_MSKB_ROOT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-target-spectra", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _canonicalize_label(value: Any) -> str:
    """Match the v3 canonical peptidoform representation."""
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
        raise ValueError(f"Unsupported mskb_final sequence notation: {value!r}")
    prefix = re.match(r"^(?:[+-]\d+(?:\.\d+)?)*", text)
    assert prefix is not None
    position = prefix.end()
    residues = 0
    while position < len(text):
        match = RESIDUE_TOKEN_RE.match(text, position)
        if match is None or match.group("aa") not in VALID_RESIDUES:
            raise ValueError(f"Unsupported mskb_final sequence notation: {value!r}")
        residues += 1
        position = match.end()
    if residues == 0:
        raise ValueError(f"Empty mskb_final sequence: {value!r}")
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


def _base_physical_key(run_file: Any, scan: Any, charge: Any) -> str:
    try:
        scan_value = int(scan)
        charge_value = int(charge)
    except (TypeError, ValueError):
        return ""
    run = _normalize_run(run_file)
    return f"{run}|scan={scan_value}|z={charge_value}" if run else ""


def _parse_title(title: Any) -> tuple[str, int]:
    match = TITLE_SCAN_RE.match(_as_text(title))
    if match is None:
        raise ValueError(f"Unparseable official mskb_final title: {title!r}")
    return match.group("run"), int(match.group("scan"))


def _iter_mskb_rows(
    root: Path,
    split: str,
    batch_size: int,
) -> Iterator[tuple[str, str]]:
    dataset = lance.dataset(str(root / f"{split}.lance"))
    columns = ["title", "seq", "precursor_charge"]
    for batch in tqdm(
        dataset.to_batches(columns=columns, batch_size=batch_size),
        desc=f"official mskb_final {split}",
        unit="batch",
        mininterval=2.0,
    ):
        titles = batch.column(0).to_pylist()
        sequences = batch.column(1).to_pylist()
        charges = batch.column(2).to_pylist()
        for title, sequence, charge in zip(titles, sequences, charges):
            run, scan = _parse_title(title)
            yield _canonicalize_label(sequence), _base_physical_key(run, scan, charge)


def _iter_v4_non_mskb_rows(root: Path, split: str, batch_size: int):
    dataset = lance.dataset(str(root / f"{split}.lance"))
    columns = ["seq", "source_id", "source_dataset", "run_file", "physical_scan_id", "precursor_charge"]
    for batch in tqdm(
        dataset.to_batches(columns=columns, batch_size=batch_size),
        desc=f"v4 non-mskb {split}",
        unit="batch",
        mininterval=2.0,
    ):
        arrays = [batch.column(index).to_pylist() for index in range(len(columns))]
        for seq, source_id, source_dataset, run_file, scan, charge in zip(*arrays):
            if str(source_id) == "mskb_v5_seed" or str(source_dataset) == "mskb_v5_seed":
                continue
            yield (
                str(seq),
                str(source_id),
                str(source_dataset),
                _base_physical_key(run_file, scan, charge),
            )


def _split_validation_groups(
    counts: dict[str, int],
    target_spectra: int,
    seed: int,
) -> tuple[set[str], int]:
    ordered = sorted(
        counts,
        key=lambda seq: (_stable_u63("dnlv1_mskb_final_validation", seed, seq), seq),
    )
    selected: set[str] = set()
    selected_rows = 0
    for seq in ordered:
        if selected_rows >= target_spectra:
            break
        selected.add(seq)
        selected_rows += counts[seq]
    return selected, selected_rows


def main() -> None:
    args = parse_args()
    if args.validation_target_spectra < 1 or args.batch_size < 1:
        raise ValueError("Validation target and batch size must be positive.")
    v4_root = args.v4_root.resolve()
    mskb_root = args.mskb_root.resolve()
    for root, label in ((v4_root, "v4"), (mskb_root, "mskb_final")):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {label} root: {root}")
    for split in SPLITS:
        if not (v4_root / f"{split}.lance").is_dir():
            raise FileNotFoundError(v4_root / f"{split}.lance")
    for split in ("train", "test"):
        if not (mskb_root / f"{split}.lance").is_dir():
            raise FileNotFoundError(mskb_root / f"{split}.lance")

    # Official test has absolute priority. Any official-train peptidoform found
    # there cannot be selected for the new validation or training membership.
    official_test_sequences: set[str] = set()
    official_base_keys: set[str] = set()
    official_counts = Counter()
    for seq, base_key in _iter_mskb_rows(mskb_root, "test", args.batch_size):
        official_test_sequences.add(seq)
        official_base_keys.add(base_key)
        official_counts["test_rows"] += 1

    eligible_train_counts: dict[str, int] = defaultdict(int)
    official_train_overlap_test = Counter()
    official_train_overlap_test_sequences: set[str] = set()
    for seq, base_key in _iter_mskb_rows(mskb_root, "train", args.batch_size):
        official_counts["train_rows"] += 1
        official_base_keys.add(base_key)
        if seq in official_test_sequences:
            official_train_overlap_test["rows"] += 1
            official_train_overlap_test_sequences.add(seq)
            continue
        eligible_train_counts[seq] += 1
    official_counts["train_peptidoforms_overlapping_official_test"] = len(
        official_train_overlap_test_sequences
    )
    official_counts["train_rows_overlapping_official_test"] = official_train_overlap_test["rows"]

    validation_sequences, validation_rows = _split_validation_groups(
        eligible_train_counts,
        args.validation_target_spectra,
        args.seed,
    )
    train_sequences = set(eligible_train_counts) - validation_sequences
    mskb_split_by_sequence = {seq: "test" for seq in official_test_sequences}
    mskb_split_by_sequence.update({seq: "val" for seq in validation_sequences})
    mskb_split_by_sequence.update({seq: "train" for seq in train_sequences})

    v4_non_mskb = Counter()
    v4_legacy_mskb = Counter()
    sequence_conflicts = Counter()
    physical_conflicts = Counter()
    for split in SPLITS:
        dataset = lance.dataset(str(v4_root / f"{split}.lance"))
        v4_rows_total = dataset.count_rows()
        for seq, source_id, source_dataset, base_key in _iter_v4_non_mskb_rows(
            v4_root, split, args.batch_size
        ):
            v4_non_mskb[split] += 1
            source_key = f"{source_id}|{source_dataset}"
            mskb_split = mskb_split_by_sequence.get(seq)
            if mskb_split in {"val", "test"}:
                sequence_conflicts[(split, mskb_split, source_key)] += 1
            if base_key and base_key in official_base_keys:
                physical_conflicts[(split, source_key)] += 1
        v4_legacy_mskb[split] = v4_rows_total - v4_non_mskb[split]

    report = {
        "script_version": SCRIPT_VERSION,
        "mode": "investigate_only_no_lance_written",
        "v4_root": str(v4_root),
        "official_mskb_final_root": str(mskb_root),
        "policy": {
            "source_replacement": "replace only legacy mskb_v5_seed rows; retain final v4 non-mskb rows as candidates",
            "official_test": "all official mskb_final test rows are assigned to v5 test",
            "validation": {
                "seed": args.seed,
                "unit": "canonical peptidoform",
                "target_spectra": args.validation_target_spectra,
                "selection": "stable hash-ranked peptidoforms accumulated to target",
            },
            "global_cleanup_reference": "v3 test>val>train peptidoform precedence and duplicate cleanup",
        },
        "official_mskb_final": {
            **dict(official_counts),
            "test_unique_peptidoforms": len(official_test_sequences),
            "eligible_train_unique_peptidoforms": len(eligible_train_counts),
            "validation_unique_peptidoforms": len(validation_sequences),
            "validation_rows": validation_rows,
            "train_unique_peptidoforms_after_validation": len(train_sequences),
            "train_rows_after_validation": sum(eligible_train_counts[seq] for seq in train_sequences),
            "official_base_physical_keys": len(official_base_keys),
        },
        "current_v4": {
            "legacy_mskb_v5_seed_rows": dict(v4_legacy_mskb),
            "non_mskb_candidate_rows": dict(v4_non_mskb),
        },
        "non_mskb_rows_removed_by_new_mskb_membership": {
            "peptidoform_heldout_conflicts": [
                {
                    "existing_v4_split": split,
                    "new_mskb_final_split": mskb_split,
                    "source": source,
                    "rows": rows,
                }
                for (split, mskb_split, source), rows in sorted(sequence_conflicts.items())
            ],
            "physical_key_conflicts": [
                {
                    "existing_v4_split": split,
                    "source": source,
                    "rows": rows,
                }
                for (split, source), rows in sorted(physical_conflicts.items())
            ],
            "union_count_note": "Per-row union is computed during v5 materialization; this audit reports each invariant independently.",
        },
        "next_step": "Materialize a new v5 candidate root, then run the preserved v3 cleanup and full physical/exact-peak/peptidoform audit before promotion.",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote dIon-de-novo-labeled-v1 (DNLv1) audit report: {args.output_report}")


if __name__ == "__main__":
    main()
