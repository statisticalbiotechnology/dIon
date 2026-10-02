from types import SimpleNamespace

import torch

from src.embed_eval.baselines import binned_spectrum_details, similarity_metadata
from src.embed_eval.loading import load_peak_only_binned_spectrum_embedder


def _args(**overrides):
    values = {
        "use_mass": 0,
        "use_charge": 0,
        "max_mz": 2500.0,
        "max_charge": 10,
        "intensity_scaling": "minmax",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_peak_only_binner_ignores_precursor_metadata():
    _, embedder = load_peak_only_binned_spectrum_embedder(_args())
    spectra = torch.tensor([[[100.0, 0.2], [101.0, 0.8], [0.0, 0.0]]])
    mask = torch.tensor([[False, False, True]])
    first = embedder(spectra, mask, torch.tensor([500.0]), torch.tensor([2]))
    second = embedder(spectra, mask, torch.tensor([1500.0]), torch.tensor([9]))
    assert first.shape == (1, 1024)
    assert torch.equal(first, second)


def test_peak_only_binner_rejects_metadata_enabled_configuration():
    try:
        load_peak_only_binned_spectrum_embedder(_args(use_mass=1))
    except ValueError as exc:
        assert "use_mass 0" in str(exc)
    else:
        raise AssertionError("expected metadata-enabled binned config to fail")


def test_binned_provenance_and_spectral_angle_equivalence():
    details = binned_spectrum_details(_args())
    assert details["metadata_features"] == []
    assert details["bin_count"] == 1024
    assert details["bin_width_da"] == 2500.0 / 1024
    similarity = similarity_metadata("cosine")
    assert similarity["equivalent_normalized_spectral_angle"]["ranking_metrics_identical_to_cosine"]
