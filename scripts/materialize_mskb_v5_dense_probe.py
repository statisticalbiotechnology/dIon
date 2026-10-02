#!/usr/bin/env python3
"""Build a small, frozen-token de novo probe from official MKB-v5 training.

The probe is intentionally a development monitor, not a final de novo
benchmark. It is drawn only from the official Casanovo MKB-v5 train split and
uses raw-file identities recovered from ``title`` (``<raw_file>:scan:<id>``).

``--mode investigate`` is read-only and prints the proposed split audit.
``--mode materialize`` writes train/val/test Lance datasets plus a manifest,
but only after verifying both run and peptide disjointness.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
import json
from pathlib import Path
import shutil
from typing import Iterable

import lance
import pyarrow as pa


DEFAULT_INPUT = (
    "/path/to/data/casanovo_official/v5_mskb_final/"
    "lance_no_peak_cap/custom_splits_seed42/train.lance"
)
DEFAULT_OUTPUT = "/path/to/data/denovo_mskb_v5_dense_probe_16aa_short_v2"
# Match the successful DeNovoMNIST probe alphabet while retaining a strict
# run- and peptide-disjoint split from the MKB-v5 training source.
DEFAULT_VOCAB = "ADEFGIKLNPQRSTVY"
TITLE_SCAN_MARKER = ":scan:"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CandidateRow:
    row_id: int
    seq: str
    run_file: str
    precursor_charge: int

    @property
    def length(self) -> int:
        return len(self.seq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("investigate", "materialize"),
        default="investigate",
        help="Investigate is read-only; materialize writes the selected Lance splits.",
    )
    parser.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB,
        help="Unique unmodified one-letter amino-acid tokens to retain.",
    )
    parser.add_argument("--min-length", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("--train-rows", type=int, default=7000)
    parser.add_argument("--val-rows", type=int, default=900)
    parser.add_argument("--test-rows", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of an existing output root in materialize mode.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str:
    vocab = args.vocab.strip().upper()
    if not vocab or not vocab.isalpha() or not vocab.isascii():
        raise ValueError("--vocab must be a non-empty ASCII one-letter-token string")
    if len(set(vocab)) != len(vocab):
        raise ValueError("--vocab must not contain duplicate tokens")
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("Require 1 <= --min-length <= --max-length")
    for name in ("train_rows", "val_rows", "test_rows"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return vocab


def run_from_title(title: str) -> str:
    if not title or TITLE_SCAN_MARKER not in title:
        raise ValueError(
            "Cannot derive raw-file provenance: expected title of the form "
            f"'<raw_file>{TITLE_SCAN_MARKER}<scan>', got {title!r}"
        )
    return title.split(TITLE_SCAN_MARKER, 1)[0]


def stable_key(seed: int, *parts: object) -> int:
    text = ":".join(str(part) for part in (seed, *parts))
    return int.from_bytes(blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def largest_remainder_quotas(
    length_counts: Counter[int], target_rows: int
) -> dict[int, int]:
    total = sum(length_counts.values())
    if total < target_rows:
        raise ValueError(f"Only {total} candidate rows available for target {target_rows}")

    raw = {
        length: target_rows * count / total for length, count in length_counts.items()
    }
    quotas = {length: int(value) for length, value in raw.items()}
    remainder = target_rows - sum(quotas.values())
    for length, _ in sorted(
        raw.items(), key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True
    )[:remainder]:
        quotas[length] += 1
    return dict(sorted(quotas.items()))


def scan_candidates(
    dataset: lance.LanceDataset,
    *,
    vocab: str,
    min_length: int,
    max_length: int,
    batch_size: int,
) -> tuple[list[CandidateRow], dict[str, int]]:
    allowed = frozenset(vocab)
    candidates: list[CandidateRow] = []
    counters = Counter()
    row_offset = 0

    scanner = dataset.scanner(
        columns=["seq", "title", "precursor_charge"], batch_size=batch_size
    )
    for batch in scanner.to_batches():
        data = batch.to_pydict()
        for index, (seq, title, charge) in enumerate(
            zip(data["seq"], data["title"], data["precursor_charge"])
        ):
            counters["input_rows"] += 1
            if not seq:
                counters["excluded_empty_sequence"] += 1
                continue
            if not min_length <= len(seq) <= max_length:
                counters["excluded_length"] += 1
                continue
            if not frozenset(seq) <= allowed:
                counters["excluded_vocab_or_modification"] += 1
                continue
            candidates.append(
                CandidateRow(
                    row_id=row_offset + index,
                    seq=seq,
                    run_file=run_from_title(title),
                    precursor_charge=int(charge),
                )
            )
            counters["eligible_rows"] += 1
        row_offset += batch.num_rows

    if row_offset != dataset.count_rows():
        raise RuntimeError(
            f"Scanned {row_offset} rows but dataset reports {dataset.count_rows()} rows"
        )
    return candidates, dict(counters)


def select_holdout(
    *,
    split: str,
    target_rows: int,
    candidates_by_run: dict[str, list[CandidateRow]],
    length_quotas: dict[int, int],
    used_runs: set[str],
    seed: int,
) -> list[CandidateRow]:
    """Select rows from complete, held-out runs while matching length quotas."""
    selected: list[CandidateRow] = []
    remaining = dict(length_quotas)
    ordered_runs = sorted(
        (run for run in candidates_by_run if run not in used_runs),
        key=lambda run: stable_key(seed, split, "run", run),
    )

    for run_file in ordered_runs:
        if not any(remaining.values()):
            break
        rows = sorted(
            candidates_by_run[run_file],
            key=lambda row: stable_key(seed, split, "row", row.row_id),
        )
        chosen = [row for row in rows if remaining.get(row.length, 0) > 0]
        if not chosen:
            continue

        # Reserve the complete raw file from the other splits, even if only a
        # subset of its singleton-peptide rows is needed to satisfy the quota.
        used_runs.add(run_file)
        for row in chosen:
            if remaining[row.length] <= 0:
                continue
            selected.append(row)
            remaining[row.length] -= 1

    if len(selected) != target_rows or any(remaining.values()):
        raise ValueError(
            f"Could not select {target_rows} {split} rows from run-unique peptides; "
            f"selected {len(selected)}, remaining length quotas={remaining}"
        )
    return selected


def select_train(
    *,
    target_rows: int,
    candidates: Iterable[CandidateRow],
    excluded_runs: set[str],
    excluded_peptides: set[str],
    length_quotas: dict[int, int],
    seed: int,
) -> list[CandidateRow]:
    by_length: dict[int, list[CandidateRow]] = defaultdict(list)
    for row in candidates:
        if row.run_file in excluded_runs or row.seq in excluded_peptides:
            continue
        by_length[row.length].append(row)

    selected: list[CandidateRow] = []
    for length, quota in length_quotas.items():
        rows = sorted(
            by_length[length], key=lambda row: stable_key(seed, "train", row.row_id)
        )
        if len(rows) < quota:
            raise ValueError(
                f"Only {len(rows)} eligible training rows of length {length}; need {quota}"
            )
        selected.extend(rows[:quota])
    if len(selected) != target_rows:
        raise RuntimeError(f"Selected {len(selected)} training rows, expected {target_rows}")
    return selected


def assert_disjoint(selected: dict[str, list[CandidateRow]]) -> dict[str, object]:
    run_sets = {split: {row.run_file for row in rows} for split, rows in selected.items()}
    peptide_sets = {split: {row.seq for row in rows} for split, rows in selected.items()}
    row_sets = {split: {row.row_id for row in rows} for split, rows in selected.items()}
    leakage = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            for name, values in (("runs", run_sets), ("peptides", peptide_sets), ("rows", row_sets)):
                overlap = values[left] & values[right]
                leakage[f"{left}_{right}_{name}"] = len(overlap)
                if overlap:
                    examples = sorted(overlap)[:5]
                    raise RuntimeError(
                        f"{name} leak between {left} and {right}: {examples}"
                    )
    return {
        "run_counts": {split: len(values) for split, values in run_sets.items()},
        "peptide_counts": {split: len(values) for split, values in peptide_sets.items()},
        "row_counts": {split: len(values) for split, values in row_sets.items()},
        "leakage": leakage,
    }


def split_summary(rows: list[CandidateRow], target_quotas: dict[int, int]) -> dict[str, object]:
    lengths = Counter(row.length for row in rows)
    charges = Counter(row.precursor_charge for row in rows)
    return {
        "rows": len(rows),
        "unique_peptides": len({row.seq for row in rows}),
        "unique_runs": len({row.run_file for row in rows}),
        "length_counts": dict(sorted(lengths.items())),
        "target_length_counts": target_quotas,
        "charge_counts": dict(sorted(charges.items())),
    }


def build_selection(args: argparse.Namespace, vocab: str) -> tuple[dict[str, list[CandidateRow]], dict[str, object]]:
    dataset = lance.dataset(args.input_lance)
    candidates, scan_counters = scan_candidates(
        dataset,
        vocab=vocab,
        min_length=args.min_length,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    if not candidates:
        raise ValueError("No rows matched the requested reduced-vocabulary filter")

    peptide_runs: dict[str, set[str]] = defaultdict(set)
    all_lengths = Counter()
    for row in candidates:
        peptide_runs[row.seq].add(row.run_file)
        all_lengths[row.length] += 1
    singleton_peptides = {
        peptide for peptide, runs in peptide_runs.items() if len(runs) == 1
    }
    singleton_rows_by_run: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in candidates:
        if row.seq in singleton_peptides:
            singleton_rows_by_run[row.run_file].append(row)

    val_quotas = largest_remainder_quotas(all_lengths, args.val_rows)
    test_quotas = largest_remainder_quotas(all_lengths, args.test_rows)
    train_quotas = largest_remainder_quotas(all_lengths, args.train_rows)
    used_runs: set[str] = set()
    val_rows = select_holdout(
        split="val",
        target_rows=args.val_rows,
        candidates_by_run=singleton_rows_by_run,
        length_quotas=val_quotas,
        used_runs=used_runs,
        seed=args.seed,
    )
    test_rows = select_holdout(
        split="test",
        target_rows=args.test_rows,
        candidates_by_run=singleton_rows_by_run,
        length_quotas=test_quotas,
        used_runs=used_runs,
        seed=args.seed,
    )
    holdout_peptides = {row.seq for row in val_rows} | {row.seq for row in test_rows}
    train_rows = select_train(
        target_rows=args.train_rows,
        candidates=candidates,
        excluded_runs=used_runs,
        excluded_peptides=holdout_peptides,
        length_quotas=train_quotas,
        seed=args.seed,
    )
    selected = {"train": train_rows, "val": val_rows, "test": test_rows}
    disjointness = assert_disjoint(selected)
    audit = {
        "input_lance": str(Path(args.input_lance).resolve()),
        "input_rows": dataset.count_rows(),
        "filter": {
            "vocab": vocab,
            "vocab_alphabetical": "".join(sorted(vocab)),
            "min_length": args.min_length,
            "max_length": args.max_length,
            "modified_tokens": "excluded; sequences are never normalized or rewritten",
        },
        "seed": args.seed,
        "scan_counters": scan_counters,
        "candidate": {
            "rows": len(candidates),
            "unique_peptides": len(peptide_runs),
            "unique_runs": len({row.run_file for row in candidates}),
            "length_counts": dict(sorted(all_lengths.items())),
            "singleton_run_peptides": len(singleton_peptides),
            "singleton_run_rows": sum(len(rows) for rows in singleton_rows_by_run.values()),
        },
        "splits": {
            "train": split_summary(train_rows, train_quotas),
            "val": split_summary(val_rows, val_quotas),
            "test": split_summary(test_rows, test_quotas),
        },
        "disjointness": disjointness,
        "selection_policy": {
            "holdouts": (
                "Validation/test rows come only from peptides observed in exactly one "
                "eligible raw file. Every raw file selected for either holdout is "
                "excluded entirely from train."
            ),
            "training": (
                "Training rows are sampled from the remaining raw files and exclude "
                "all held-out peptide identities."
            ),
            "stratification": "Exact per-split peptide-length quotas by largest remainder.",
        },
    }
    return selected, audit


def table_for_rows(dataset: lance.LanceDataset, rows: list[CandidateRow], split: str) -> pa.Table:
    indices = [row.row_id for row in rows]
    table = dataset.take(indices)
    if table.num_rows != len(rows):
        raise RuntimeError(f"Fetched {table.num_rows} rows for {split}, expected {len(rows)}")
    fetched_sequences = table.column("seq").to_pylist()
    fetched_runs = [run_from_title(title) for title in table.column("title").to_pylist()]
    if fetched_sequences != [row.seq for row in rows] or fetched_runs != [
        row.run_file for row in rows
    ]:
        raise RuntimeError(
            f"Lance take() did not preserve requested source-row alignment for {split}"
        )
    return table.append_column("probe_source_row", pa.array(indices, type=pa.int64())).append_column(
        "probe_run_file", pa.array([row.run_file for row in rows], type=pa.string())
    ).append_column("probe_split", pa.array([split] * len(rows), type=pa.string()))


def materialize(args: argparse.Namespace, selected: dict[str, list[CandidateRow]], audit: dict[str, object]) -> None:
    root = Path(args.output_root)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    dataset = lance.dataset(args.input_lance)
    for split in SPLITS:
        lance.write_dataset(table_for_rows(dataset, selected[split], split), root / f"{split}.lance", mode="create")
    audit["output_root"] = str(root.resolve())
    audit["mode"] = "materialize"
    (root / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Wrote MKB-v5 dense probe: {root}")


def main() -> None:
    args = parse_args()
    vocab = validate_args(args)
    selected, audit = build_selection(args, vocab)
    audit["mode"] = args.mode
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.mode == "materialize":
        materialize(args, selected, audit)


if __name__ == "__main__":
    main()
