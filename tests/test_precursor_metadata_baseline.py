"""Tests for the label-free precursor metadata evaluation baseline."""

from types import SimpleNamespace

import pyarrow as pa
import torch

from src.embed_eval.extraction import extract_precursor_metadata_embeddings


def test_precursor_metadata_baseline_uses_mass_and_charge_without_labels():
    dataset = SimpleNamespace(
        table=pa.table(
            {
                "precursor_mz": [500.0, 500.0, 750.0],
                "precursor_charge": [2, 3, 2],
                "peptide_ion_id": ["ion_a", "ion_b", "ion_c"],
                "species": ["species_a", "species_a", "species_b"],
                "spectrum_id": ["scan_1", "scan_2", "scan_3"],
            }
        )
    )

    cache = extract_precursor_metadata_embeddings(dataset)

    assert cache.values.shape == (3, 2)
    assert torch.allclose(cache.values.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert cache.peptide_ids == ["ion_a", "ion_b", "ion_c"]
    assert cache.partition_ids == ["species_a", "species_a", "species_b"]
    assert cache.spectrum_ids == ["scan_1", "scan_2", "scan_3"]
