import torch

from src.callbacks.linprobe_callback import EndAAProbeCallback


class _RecordingEmbedder(torch.nn.Module):
    use_mass = True
    use_charge = True

    def __init__(self):
        super().__init__()
        self.received_mass = None

    def forward(self, spectra, pad_mask, mass, charge):
        del pad_mask, charge
        self.received_mass = mass.detach().clone()
        return torch.zeros((spectra.shape[0], 2), device=spectra.device)


def _batch():
    return {
        "mz_array": torch.tensor([[100.0, 200.0], [300.0, 0.0]]),
        "intensity_array": torch.tensor([[1.0, 0.5], [1.0, 0.0]]),
        "peak_lengths": torch.tensor([2, 1]),
        "precursor_mz": torch.tensor([500.0, 600.0]),
        "precursor_mass": torch.tensor([1000.0, 1800.0]),
        "precursor_charge": torch.tensor([2, 3]),
        "labels": torch.tensor([0, 1]),
    }


def _callback(mass_input):
    callback = object.__new__(EndAAProbeCallback)
    callback.mass_input = mass_input
    callback.precursor_conditioning = "conditioned"
    return callback


def test_cache_embeddings_uses_training_matched_precursor_mass():
    batch = _batch()
    embedder = _RecordingEmbedder()
    _callback("precursor_mass")._cache_embeddings(
        [batch], embedder, torch.device("cpu"), split_name="test", show_progress=False
    )
    assert torch.equal(embedder.received_mass, batch["precursor_mass"])



def test_cache_embeddings_supports_null_precursor_conditioning():
    batch = _batch()
    embedder = _RecordingEmbedder()
    callback = _callback("precursor_mass")
    callback.precursor_conditioning = "null"
    callback._cache_embeddings(
        [batch], embedder, torch.device("cpu"), split_name="test", show_progress=False
    )
    assert torch.equal(embedder.received_mass, torch.zeros_like(batch["precursor_mass"]))

def test_cache_embeddings_supports_explicit_legacy_precursor_mz():
    batch = _batch()
    embedder = _RecordingEmbedder()
    _callback("precursor_mz")._cache_embeddings(
        [batch], embedder, torch.device("cpu"), split_name="test", show_progress=False
    )
    assert torch.equal(embedder.received_mass, batch["precursor_mz"])
