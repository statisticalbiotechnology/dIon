#!/usr/bin/env python3
"""Strict post-materialization cleanup for the General vNext Lance corpus.

This script preserves all output/provenance columns while enforcing two final
invariants that cannot be guaranteed from heterogeneous source identifiers:

1. A canonical, sequence-disjoint split for every peptidoform. Existing test
   rows take precedence over validation, which takes precedence over training.
   Lower-priority occurrences are removed rather than moved between splits.
2. One representative for every byte-identical m/z peak list. Intensities are
   deliberately excluded because identical spectra can carry source-specific
   intensity normalization. Source priority decides the canonical row.

A complete temporary Lance root is strictly audited before it replaces the
input root. The old materialization is kept as ``lance_pre_cleanup``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from tqdm import tqdm

import lance


SPLITS = ("train", "val", "test")
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\\n")


def _peak_fingerprint(mz_array: Any) -> bytes:
    mz = np.asarray(mz_array, dtype=np.float32)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([len(mz)], dtype=np.int64).tobytes())
    digest.update(mz.tobytes())
    return digest.digest()


def _directory_size_gib(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total / (1024 ** 3)


def _assert_storage_budget(output_root: Path, max_gib: float) -> None:
    size_gib = _directory_size_gib(output_root)
    print(f"[storage] output={output_root} size={size_gib:.2f} GiB budget={max_gib:.2f} GiB")
    if size_gib > max_gib:
        raise RuntimeError(f"Output budget exceeded: {size_gib:.2f} GiB > {max_gib:.2f} GiB")


def _source_priorities(config: dict[str, Any]) -> dict[str, int]:
    priorities = {"mskb_v5_seed": 0}
    for source in config["sources"]:
        value = source.get("source_priority")
        if value is not None:
            priorities[str(source["id"])] = int(value)
    return priorities


def _audit(lance_root: Path, batch_size: int) -> dict[str, Any]:
    physical_seen: set[str] = set()
    mz_seen: set[bytes] = set()
    seq_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    rows_by_split: dict[str, int] = {}
    physical_duplicates = 0
    mz_duplicates = 0
    for split in SPLITS:
        path = lance_root / f"{split}.lance"
        if not path.exists():
            raise FileNotFoundError(f"Missing split during audit: {path}")
        dataset = lance.dataset(str(path))
        rows_by_split[split] = int(dataset.count_rows())
        for batch in tqdm(
            dataset.to_batches(columns=["physical_key", "seq", "mz_array"], batch_size=batch_size),
            desc=f"audit {split}",
            unit="batch",
            mininterval=2.0,
        ):
            physical_values = batch.column(0).to_pylist()
            seq_values = batch.column(1).to_pylist()
            mz_values = batch.column(2).to_pylist()
            for physical, seq, mz_array in zip(physical_values, seq_values, mz_values):
                if physical in physical_seen:
                    physical_duplicates += 1
                physical_seen.add(physical)
                fingerprint = _peak_fingerprint(mz_array)
                if fingerprint in mz_seen:
                    mz_duplicates += 1
                mz_seen.add(fingerprint)
                seq_by_split[split].add(seq)
    overlaps = {
        "train_val_peptidoforms": len(seq_by_split["train"] & seq_by_split["val"]),
        "train_test_peptidoforms": len(seq_by_split["train"] & seq_by_split["test"]),
        "val_test_peptidoforms": len(seq_by_split["val"] & seq_by_split["test"]),
    }
    return {
        "rows_by_split": rows_by_split,
        "unique_peptidoforms_by_split": {split: len(values) for split, values in seq_by_split.items()},
        "duplicate_physical_rows": physical_duplicates,
        "duplicate_exact_mz_peaklist_rows": mz_duplicates,
        "peptidoform_overlaps": overlaps,
    }


def _iter_batches(dataset: lance.LanceDataset, columns: list[str], batch_size: int, desc: str):
    return tqdm(
        dataset.to_batches(columns=columns, batch_size=batch_size),
        desc=desc,
        unit="batch",
        mininterval=2.0,
    )


def audit_existing(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    tmp_root = output_root / "lance_cleanup_tmp"
    input_root = args.input_root.resolve()
    backup_root = output_root / "lance_pre_cleanup"
    if not tmp_root.exists():
        raise FileNotFoundError(f"Missing staged cleanup root: {tmp_root}")
    if args.replace and backup_root.exists():
        raise FileExistsError(f"Backup root already exists: {backup_root}; refusing to overwrite")

    audit = _audit(tmp_root, args.batch_size)
    summary = {
        "temporary_lance_root": str(tmp_root),
        "temporary_audit": audit,
        "audit_only": True,
    }
    _write_json(output_root / "cleanup_summary.json", summary)
    if (
        audit["duplicate_physical_rows"]
        or audit["duplicate_exact_mz_peaklist_rows"]
        or any(audit["peptidoform_overlaps"].values())
    ):
        raise RuntimeError(f"Staged cleanup audit failed: {json.dumps(audit, sort_keys=True)}")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.replace:
        print("[cleanup] Staged root passed audit; rerun with --audit-existing --replace to promote it.")
        return

    if not input_root.exists():
        raise FileNotFoundError(f"Current Lance root missing: {input_root}")
    os.rename(input_root, backup_root)
    os.rename(tmp_root, input_root)
    summary["promoted_lance_root"] = str(input_root)
    summary["backup_lance_root"] = str(backup_root)
    _write_json(output_root / "cleanup_summary.json", summary)
    print(f"[cleanup] Promoted {input_root}; backup retained at {backup_root}")


def cleanup(args: argparse.Namespace) -> None:
    if args.audit_existing:
        audit_existing(args)
        return

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    split_paths = {split: input_root / f"{split}.lance" for split in SPLITS}
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input Lance split(s): {missing}")
    if input_root.name != "lance":
        raise ValueError(f"Expected input root ending in 'lance', got {input_root}")

    config = json.loads(args.config.read_text())
    source_priority = _source_priorities(config)
    tmp_root = output_root / "lance_cleanup_tmp"
    backup_root = output_root / "lance_pre_cleanup"
    if tmp_root.exists():
        if not args.force:
            raise FileExistsError(f"Temporary root exists: {tmp_root}; use --force to replace it")
        shutil.rmtree(tmp_root)
    if backup_root.exists():
        if not args.force:
            raise FileExistsError(f"Backup root exists: {backup_root}; use --force to replace it")
        shutil.rmtree(backup_root)

    # Pass 1: assign each peptidoform one canonical split. This is an in-memory
    # map of labels only; all spectra remain on disk until the write pass.
    best_split_by_seq: dict[str, int] = {}
    for split in SPLITS:
        dataset = lance.dataset(str(split_paths[split]))
        rank = SPLIT_RANK[split]
        for batch in _iter_batches(dataset, ["seq"], args.batch_size, f"split policy {split}"):
            for seq in batch.column(0).to_pylist():
                previous = best_split_by_seq.get(seq)
                if previous is None or rank < previous:
                    best_split_by_seq[seq] = rank

    # Pass 2: determine the canonical row of every exact m/z peak list among
    # spectra surviving the split policy. The row offset is stable within this
    # Lance snapshot and prevents collapsing distinct rows with equal metadata.
    winner_by_fingerprint: dict[bytes, tuple[int, int, str, int]] = {}
    candidate_rows = 0
    duplicate_candidates = 0
    for split in SPLITS:
        dataset = lance.dataset(str(split_paths[split]))
        rank = SPLIT_RANK[split]
        row_number = 0
        for batch in _iter_batches(dataset, ["seq", "source_id", "mz_array"], args.batch_size, f"peak winners {split}"):
            seq_values = batch.column(0).to_pylist()
            source_values = batch.column(1).to_pylist()
            mz_values = batch.column(2).to_pylist()
            for offset, (seq, source_id, mz_array) in enumerate(zip(seq_values, source_values, mz_values)):
                index = row_number + offset
                if best_split_by_seq[seq] != rank:
                    continue
                candidate_rows += 1
                fingerprint = _peak_fingerprint(mz_array)
                candidate = (source_priority.get(source_id, 10_000), rank, source_id, index)
                previous = winner_by_fingerprint.get(fingerprint)
                if previous is not None:
                    duplicate_candidates += 1
                if previous is None or candidate < previous:
                    winner_by_fingerprint[fingerprint] = candidate
            row_number += batch.num_rows

    # Pass 3: retain only scalar metadata in Python and filter full Arrow rows
    # in-place. Converting every peak array through ``to_pylist`` is several
    # orders slower on the fragmented pre-cleanup materialization.
    schema = lance.dataset(str(split_paths["train"])).schema
    buffers: dict[str, list[pa.Table]] = defaultdict(list)
    buffered_rows: Counter[str] = Counter()
    created: set[str] = set()
    written: Counter[str] = Counter()
    dropped_by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    dropped_by_source: Counter[str] = Counter()

    def flush(split: str) -> None:
        tables = buffers[split]
        if not tables:
            return
        path = tmp_root / f"{split}.lance"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.concat_tables(tables)
        lance.write_dataset(
            table,
            str(path),
            mode="append" if split in created else "create",
            max_rows_per_file=100_000,
            max_rows_per_group=4096,
        )
        created.add(split)
        written[split] += table.num_rows
        buffers[split] = []
        buffered_rows[split] = 0

    columns = list(schema.names)
    for split in SPLITS:
        dataset = lance.dataset(str(split_paths[split]))
        rank = SPLIT_RANK[split]
        row_number = 0
        for batch in _iter_batches(dataset, columns, args.batch_size, f"write {split}"):
            seq_values = batch.column(batch.schema.get_field_index("seq")).to_pylist()
            source_values = batch.column(batch.schema.get_field_index("source_id")).to_pylist()
            mz_values = batch.column(batch.schema.get_field_index("mz_array")).to_pylist()
            keep: list[bool] = []
            for offset, (seq, source_id, mz_array) in enumerate(zip(seq_values, source_values, mz_values)):
                index = row_number + offset
                if best_split_by_seq[seq] != rank:
                    dropped_by_split[split]["peptidoform_split_policy"] += 1
                    dropped_by_source[source_id] += 1
                    keep.append(False)
                    continue
                fingerprint = _peak_fingerprint(mz_array)
                candidate = (
                    source_priority.get(source_id, 10_000),
                    rank,
                    source_id,
                    index,
                )
                if winner_by_fingerprint[fingerprint] != candidate:
                    dropped_by_split[split]["exact_mz_peaklist_duplicate"] += 1
                    dropped_by_source[source_id] += 1
                    keep.append(False)
                    continue
                keep.append(True)
            if any(keep):
                filtered = pa.Table.from_batches([batch]).filter(pa.array(keep, type=pa.bool_()))
                buffers[split].append(filtered)
                buffered_rows[split] += filtered.num_rows
                if buffered_rows[split] >= args.rows_per_write:
                    flush(split)
            row_number += batch.num_rows
        flush(split)

    _assert_storage_budget(output_root, float(config["max_output_gib"]))
    temporary_audit = _audit(tmp_root, args.batch_size)
    summary = {
        "input_lance_root": str(input_root),
        "temporary_lance_root": str(tmp_root),
        "policy": {
            "split_priority": ["test", "val", "train"],
            "exact_peak_representative_order": ["source_priority", "split_priority", "source_id", "lance_row_index"],
            "source_priorities": source_priority,
        },
        "unique_peptidoforms_global": len(best_split_by_seq),
        "candidate_rows_after_split_policy": candidate_rows,
        "exact_peak_collision_rows_before_collapse": duplicate_candidates,
        "written_rows": dict(written),
        "dropped_by_split": {split: dict(values) for split, values in dropped_by_split.items()},
        "dropped_by_source": dict(dropped_by_source),
        "temporary_audit": temporary_audit,
    }
    _write_json(output_root / "cleanup_summary.json", summary)
    if (
        temporary_audit["duplicate_physical_rows"]
        or temporary_audit["duplicate_exact_mz_peaklist_rows"]
        or any(temporary_audit["peptidoform_overlaps"].values())
    ):
        raise RuntimeError(f"Cleanup audit failed; original Lance root retained: {json.dumps(temporary_audit, sort_keys=True)}")
    if not args.replace:
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("[cleanup] Temporary root passed audit; rerun with --replace to promote it.")
        return

    os.rename(input_root, backup_root)
    os.rename(tmp_root, input_root)
    summary["promoted_lance_root"] = str(input_root)
    summary["backup_lance_root"] = str(backup_root)
    _write_json(output_root / "cleanup_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--rows-per-write", type=int, default=20_000)
    p.add_argument("--replace", action="store_true", help="Promote the strictly-audited temporary root.")
    p.add_argument("--audit-existing", action="store_true", help="Audit an existing lance_cleanup_tmp without rebuilding it.")
    p.add_argument("--force", action="store_true", help="Replace stale temporary/backup cleanup roots.")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.batch_size <= 0 or args.rows_per_write <= 0:
        raise ValueError("batch sizes must be positive")
    cleanup(args)


if __name__ == "__main__":
    main()
