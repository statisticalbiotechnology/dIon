"""Frozen, externally materialized spectrum embeddings for downstream heads.

This module deliberately contains no external model implementation.  An
external runner materializes embeddings keyed by the ordinary dataset row
index; downstream Lightning tasks then reuse their existing split-aware cache
and train only the established task head.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class ExternalEmbeddingSource(nn.Module):
    """Describe a complete on-disk frozen embedding cache.

    The SQA-derived wrappers perform the actual index lookup because dataset
    indices are scoped to a split.  This module exists only to provide the
    normal encoder contract (`running_units`) to head construction and model
    bookkeeping.
    """

    requires_external_embedding_cache = True
    use_mass = False
    use_charge = False
    use_energy = False

    def __init__(self, cache_dir: str | Path) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"External embedding cache is missing its manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        dimension = manifest.get("embedding_dimension")
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError(
                f"{manifest_path} must define a positive embedding_dimension."
            )
        self.running_units = dimension

    def forward(self, *args, **kwargs):  # pragma: no cover - wrapper owns split lookup
        del args, kwargs
        raise RuntimeError(
            "ExternalEmbeddingSource is index-resolved by the downstream wrapper; "
            "call the standard downstream task rather than this module directly."
        )
