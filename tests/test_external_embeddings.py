"""Contracts for frozen, externally materialized downstream embeddings."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.materialize_external_downstream_embeddings import main as materialize_main
from src.external_embeddings import ExternalEmbeddingSource


class ExternalEmbeddingCacheTests(unittest.TestCase):
    def _artifact(self, root: Path, split: str, indices: np.ndarray) -> Path:
        path = root / f"{split}.npz"
        np.savez_compressed(
            path,
            export_row_index=indices,
            embedding=np.ones((len(indices), 5), dtype=np.float32),
        )
        path.with_suffix(".manifest.json").write_text(json.dumps({"input_count": 3}))
        return path

    def _run_materializer(self, cache: Path, artifacts: dict[str, Path]) -> None:
        argv = ["materialize", "--output-cache", str(cache)]
        for split, path in artifacts.items():
            argv.extend([f"--{split}-embeddings", str(path)])
        previous = sys.argv
        try:
            sys.argv = argv
            materialize_main()
        finally:
            sys.argv = previous

    def test_complete_split_indexed_cache_exposes_embedding_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = {
                split: self._artifact(root, split, np.arange(3))
                for split in ("train", "val", "test")
            }
            cache = root / "cache"
            self._run_materializer(cache, artifacts)
            source = ExternalEmbeddingSource(cache)
            self.assertEqual(source.running_units, 5)
            payload = torch.load(cache / "val.pt", weights_only=True)
            self.assertTrue(torch.equal(payload["index"], torch.arange(3)))
            self.assertEqual(tuple(payload["features"].shape), (3, 5))

    def test_materializer_rejects_preprocessing_dropped_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "incomplete or has duplicate"):
                self._run_materializer(
                    root / "cache",
                    {"train": self._artifact(root, "train", np.array([0, 2]))},
                )


if __name__ == "__main__":
    unittest.main()
