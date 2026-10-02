#!/usr/bin/env python3
"""
Validate Lance compatibility using SafeLanceDataset + safe DataLoader.

Usage (dia-unmixing env):
  python scripts/debug_lance_safe_loader.py --max-batches 1 --batch-size 64 --num-workers 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List
from functools import partial

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Add dia-unmixing src for collate import.
DIA_SRC = Path("/path/to/work/dia-unmixing/src")
sys.path.append(str(DIA_SRC))

from scripts.debug_lance_compat import DEFAULT_DATASETS, expand_lance_path  # noqa: E402
from data.collate import pad_peaks  # noqa: E402


def make_safe_loader(dataset, *, batch_size: int, num_workers: int, collate_fn):
    if num_workers <= 0:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn
        )
    from lance.torch.data import get_safe_loader
    return get_safe_loader(
        dataset, batch_size=batch_size, num_workers=int(num_workers), collate_fn=collate_fn
    )


def try_safe_dataset(
    path: str,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    collate_fn,
) -> str:
    print(f"\n==> {path}")
    if not os.path.exists(path):
        print("  [missing] path does not exist")
        return "missing"
    try:
        from lance.torch.data import SafeLanceDataset
        ds = SafeLanceDataset(path)
        print(f"  rows: {len(ds)}")
        loader = make_safe_loader(
            ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
        )
        for i, batch in enumerate(loader):
            print(f"  batch {i}: rows={batch['mz_array'].shape[0]}")
            if i + 1 >= max_batches:
                break
    except Exception as e:
        print(f"  [error] {type(e).__name__}: {e}")
        return "error"
    return "ok"


def main() -> int:
    p = argparse.ArgumentParser(description="Validate SafeLanceDataset loader.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-peaks", type=int, default=300)
    p.add_argument("--filter-method", type=str, default="default")
    args = p.parse_args()

    expanded: List[str] = []
    for pth in DEFAULT_DATASETS:
        expanded.extend(expand_lance_path(pth))

    collate_fn = partial(
        pad_peaks,
        required_fields=["mz_array", "intensity_array"],
        max_peaks=int(args.max_peaks),
        filter_method=str(args.filter_method),
    )

    seen = set()
    stats = {"ok": 0, "missing": 0, "error": 0}
    ok_paths: List[str] = []
    missing_paths: List[str] = []
    error_paths: List[str] = []

    for path in expanded:
        if path in seen:
            continue
        seen.add(path)
        result = try_safe_dataset(
            path, args.batch_size, args.max_batches, args.num_workers, collate_fn
        )
        stats[result] += 1
        if result == "ok":
            ok_paths.append(path)
        elif result == "missing":
            missing_paths.append(path)
        else:
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
