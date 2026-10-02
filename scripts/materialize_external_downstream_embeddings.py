"""Convert complete external embeddings into the standard downstream cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-cache", type=Path, required=True)
    for split in SPLITS:
        parser.add_argument(f"--{split}-embeddings", type=Path)
    return parser.parse_args()


def _materialize_split(path: Path, output_path: Path) -> tuple[int, int, dict[str, object]]:
    artifact = np.load(path, allow_pickle=False)
    required = {"export_row_index", "embedding"}
    missing = sorted(required - set(artifact.files))
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")
    indices = np.asarray(artifact["export_row_index"], dtype=np.int64)
    embeddings = np.asarray(artifact["embedding"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != indices.size:
        raise ValueError(f"{path} has malformed embedding/index arrays.")
    if indices.size == 0:
        raise ValueError(f"{path} contains no external embeddings.")
    source_manifest = path.with_suffix(".manifest.json")
    expected_count = indices.size
    if source_manifest.exists():
        external_metadata = json.loads(source_manifest.read_text())
        expected_count = int(external_metadata.get("input_count", expected_count))
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    embeddings = embeddings[order]
    expected = np.arange(expected_count, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError(
            f"{path} is incomplete or has duplicate row indices: expected every index "
            f"in [0, {expected_count}), got {indices.size} retained rows. "
            "Do not train a downstream head on an implicitly altered split."
        )
    torch.save(
        {"index": torch.from_numpy(indices), "features": torch.from_numpy(embeddings)},
        output_path,
    )
    source_manifest = path.with_suffix(".manifest.json")
    return indices.size, embeddings.shape[1], {
        "embeddings": str(path.resolve()),
        "embeddings_sha256": _sha256(path),
        "external_manifest": str(source_manifest.resolve()) if source_manifest.exists() else None,
    }


def main() -> None:
    args = _parse_args()
    supplied = {split: getattr(args, f"{split}_embeddings") for split in SPLITS}
    if not any(supplied.values()):
        raise ValueError("Supply at least one --<split>-embeddings artifact.")
    args.output_cache.mkdir(parents=True, exist_ok=True)
    dimensions: set[int] = set()
    split_manifest: dict[str, object] = {}
    for split, source in supplied.items():
        if source is None:
            continue
        count, dimension, metadata = _materialize_split(source, args.output_cache / f"{split}.pt")
        dimensions.add(dimension)
        split_manifest[split] = {"rows": count, **metadata}
        print(f"Materialized {split}: {count:,} rows, dimension {dimension}.")
    if len(dimensions) != 1:
        raise ValueError(f"External embedding dimensions disagree across splits: {sorted(dimensions)}")
    manifest = {
        "code_version": "external_downstream_cache_v1",
        "embedding_dimension": dimensions.pop(),
        "index_contract": "each split contains every dIon dataset index exactly once",
        "splits": split_manifest,
    }
    (args.output_cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote downstream external embedding cache: {args.output_cache}")


if __name__ == "__main__":
    main()
