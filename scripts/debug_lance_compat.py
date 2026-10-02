#!/usr/bin/env python3
"""
Debug script to validate Lance dataset compatibility across environments.

Usage (in dia-unmixing env):
  python scripts/debug_lance_compat.py --max-batches 1 --batch-size 64
  python scripts/debug_lance_compat.py --paths /path/to/a.lance /path/to/b.lance
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Set

import lance


RE_LANCE = re.compile(r"(/[^\\s'\\\"]+\\.lance)")


def find_lance_paths_from_configs(config_root: Path) -> List[str]:
    paths: Set[str] = set()
    for p in config_root.rglob("*.yaml"):
        try:
            text = p.read_text()
        except Exception:
            continue
        for m in RE_LANCE.findall(text):
            paths.add(m)
    return sorted(paths)


DEFAULT_DATASETS: List[str] = [
    # Explicit .lance datasets
    "/path/to/data/lance_datasets/filtered_data_10_central.lance",
    "/path/to/data/yeast-files/yeast_validation.lance",
    "/path/to/data/yeast-files/yeast_test.lance",
    "/path/to/data/foundational_dataset/combined.lance",
    "/Users/anon/Datasets/instanovo_splits_subset/train/indexed.lance",
    "/Users/anon/Documents/Datasets/instanovo_data_subset/indexed.lance",
    # Data roots used by LanceDataModule (may contain train/val/test.lance)
    "/path/to/data/bacteria_PXD010000__PXD010613/no_labels/",
    "/path/to/data/foundational_dataset/msconvert/",
    "/path/to/data/foundational_dataset/combined/",
    "/path/to/data/MassIVE_KB/",
    "/path/to/data/InstaNovo_SPLITS_full",
    "/path/to/data/InstaNovo_larger_subset/",
    "/Users/anon/Documents/Datasets/instanovo_data_subset",
    # Discovered existing datasets under /path/to/data
    "/path/to/data/InstaNovo_SPLITS_full/train/indexed.lance",
    "/path/to/data/InstaNovo_SPLITS_full/val/indexed.lance",
    "/path/to/data/InstaNovo_SPLITS_full/test/indexed.lance",
    "/path/to/data/InstaNovo_SPLITS_larger_subset/train/indexed.lance",
    "/path/to/data/InstaNovo_SPLITS_larger_subset/val/indexed.lance",
    "/path/to/data/InstaNovo_SPLITS_larger_subset/test/indexed.lance",
    "/path/to/data/InstaNovo_dataset/foundational_model/indexed.lance",
    "/path/to/data/InstaNovo_larger_subset/indexed.lance",
    "/path/to/data/MassIVE_KB/indexed.lance",
    "/path/to/data/MassIVE_KB/train.lance",
    "/path/to/data/MassIVE_KB/val.lance",
    "/path/to/data/MassIVE_KB/test.lance",
    "/path/to/data/SQA_dataset/dummy.lance",
    "/path/to/data/SQA_dataset/dummy/train.lance",
    "/path/to/data/SQA_dataset/dummy/val.lance",
    "/path/to/data/SQA_dataset/dummy/test.lance",
    "/path/to/data/bacteria_PXD010000__PXD010613/no_labels/train.lance",
    "/path/to/data/bacteria_PXD010000__PXD010613/no_labels/val.lance",
    "/path/to/data/bacteria_PXD010000__PXD010613/no_labels/test.lance",
    "/path/to/data/foundational_dataset/combined/train.lance",
    "/path/to/data/foundational_dataset/combined/val.lance",
    "/path/to/data/foundational_dataset/msconvert/train.lance",
    "/path/to/data/foundational_dataset/msconvert/val.lance",
    "/path/to/data/instanovo_data_subset/indexed.lance",
]


def iter_lance_paths(cli_paths: List[str]) -> List[str]:
    if cli_paths:
        return cli_paths
    return DEFAULT_DATASETS


def try_read_dataset(path: str, batch_size: int, max_batches: int) -> str:
    print(f"\n==> {path}")
    if not os.path.exists(path):
        print("  [missing] path does not exist")
        return "missing"
    try:
        ds = lance.dataset(path)
        print(f"  rows: {ds.count_rows()}")
        print(f"  schema: {ds.schema}")
        batch_iter = ds.to_batches(batch_size=batch_size)
        for i, batch in enumerate(batch_iter):
            print(f"  batch {i}: rows={batch.num_rows}")
            if i + 1 >= max_batches:
                break
    except Exception as e:
        print(f"  [error] {type(e).__name__}: {e}")
        return "error"
    return "ok"


def expand_lance_path(path: str) -> List[str]:
    # If it's already a .lance dataset, keep as-is.
    if path.endswith(".lance"):
        return [path]
    # If it's a dir, try common Lance layouts.
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "indexed.lance"),
            os.path.join(path, "train.lance"),
            os.path.join(path, "val.lance"),
            os.path.join(path, "test.lance"),
        ]
        return candidates
    return [path]


def main() -> int:
    p = argparse.ArgumentParser(description="Validate Lance dataset readability.")
    p.add_argument(
        "--paths",
        nargs="*",
        default=[],
        help="Explicit .lance paths to test. If omitted, scans active configs for .lance paths.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=1)
    args = p.parse_args()

    paths = iter_lance_paths(args.paths)
    if not paths:
        print("No .lance paths found.")
        return 1

    expanded: List[str] = []
    for p in paths:
        expanded.extend(expand_lance_path(p))

    seen = set()
    stats = {"ok": 0, "missing": 0, "error": 0}
    ok_paths: List[str] = []
    missing_paths: List[str] = []
    error_paths: List[str] = []

    for path in expanded:
        if path in seen:
            continue
        seen.add(path)
        result = try_read_dataset(path, args.batch_size, args.max_batches)
        stats[result] += 1
        if result == "ok":
            ok_paths.append(path)
        elif result == "missing":
            missing_paths.append(path)
        elif result == "error":
            error_paths.append(path)

    print("\n=== Summary ===")
    print(f"ok: {stats['ok']}")
    print(f"missing: {stats['missing']}")
    print(f"error: {stats['error']}")
    if ok_paths:
        print("\nOK paths:")
        for p in ok_paths:
            print(f"- {p}")
    if missing_paths:
        print("\nMissing paths:")
        for p in missing_paths:
            print(f"- {p}")
    if error_paths:
        print("\nError paths:")
        for p in error_paths:
            print(f"- {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
