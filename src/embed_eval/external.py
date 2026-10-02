"""Load aligned embeddings exported by an external spectrum embedder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.embed_eval.extraction import CachedEmbeddings


def load_external_embedding_cache(
    artifact_path: str | Path,
    *,
    peptide_id_key: str,
    partition_id_key: str,
    spectrum_id_key: str | None = None,
) -> CachedEmbeddings:
    """Load a validated NPZ embedding artifact into the shared cache format.

    The artifact must preserve row-aligned metadata from the exported raw
    benchmark. This keeps external baselines on the exact spectra that passed
    their own published preprocessing.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"External embedding artifact not found: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "embedding",
            "precursor_mz",
            "precursor_charge",
            peptide_id_key,
            partition_id_key,
        }
        if spectrum_id_key is not None:
            required.add(spectrum_id_key)
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"{path} is missing external embedding arrays: {missing}")
        values = np.asarray(artifact["embedding"], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("External embedding array must be non-empty with shape [N, D].")
        if not np.isfinite(values).all():
            raise ValueError("External embedding array contains non-finite values.")
        size = values.shape[0]
        for key in required - {"embedding"}:
            if len(artifact[key]) != size:
                raise ValueError(
                    f"External embedding metadata {key!r} does not align with embeddings."
                )
        peptide_ids = [str(value) for value in artifact[peptide_id_key].tolist()]
        partition_ids = [str(value) for value in artifact[partition_id_key].tolist()]
        spectrum_ids = (
            [str(value) for value in artifact[spectrum_id_key].tolist()]
            if spectrum_id_key is not None
            else [str(index) for index in range(size)]
        )
        if len(set(spectrum_ids)) != len(spectrum_ids):
            raise ValueError("External embedding spectrum IDs must be unique.")
        return CachedEmbeddings(
            values=torch.from_numpy(values),
            peptide_ids=peptide_ids,
            partition_ids=partition_ids,
            spectrum_ids=spectrum_ids,
            precursor_mz=torch.from_numpy(
                np.asarray(artifact["precursor_mz"], dtype=np.float32)
            ),
            precursor_charges=torch.from_numpy(
                np.asarray(artifact["precursor_charge"], dtype=np.float32)
            ),
        )
