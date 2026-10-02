from types import SimpleNamespace

import torch

from src.callbacks.dense_denovo_probe_callback import DenseDeNovoProbeCallback
from src.data.unified_tokenizer import PeptideTokenizer, UNIFIED_RESIDUES


class _RecordingEncoder(torch.nn.Module):
    use_mass = True
    use_charge = True

    def __init__(self):
        super().__init__()
        self.mass = None
        self.charge = None

    def forward(self, spectra, key_padding_mask, mass, charge):
        self.mass = mass.detach().clone()
        self.charge = charge.detach().clone()
        return {
            "emb": torch.ones((spectra.shape[0], spectra.shape[1], 8), device=spectra.device),
            "mask": key_padding_mask,
        }


def _probe(conditioning="conditioned"):
    probe = object.__new__(DenseDeNovoProbeCallback)
    probe.global_args = SimpleNamespace(max_charge=10, max_peaks=4)
    probe.encoder_precursor_conditioning = conditioning
    probe.cache_dtype = torch.float16
    probe.max_peptide_length = 3
    probe.tokenizer = PeptideTokenizer(
        residues={token: UNIFIED_RESIDUES[token] for token in "ADE"},
        add_unk_token=False,
    )
    return probe


def _batch():
    tokenizer = _probe().tokenizer
    return {
        "mz_array": torch.tensor([[100.0, 200.0], [300.0, 0.0]]),
        "intensity_array": torch.tensor([[1.0, 0.5], [1.0, 0.0]]),
        "peak_lengths": torch.tensor([[2], [1]], dtype=torch.int32),
        "precursor_mz": torch.tensor([500.0, 600.0]),
        "precursor_mass": torch.tensor([1000.0, 1800.0]),
        "precursor_charge": torch.tensor([2, 3]),
        "intseq": torch.tensor(
            [[tokenizer.index["A"], tokenizer.index["D"]], [tokenizer.index["E"], tokenizer.pad_token_id]]
        ),
        "peptide_lengths": torch.tensor([[2], [1]], dtype=torch.int32),
    }


def test_dense_cache_uses_conditioned_precursor_and_fixed_shapes():
    probe = _probe("conditioned")
    encoder = _RecordingEncoder()
    cached = probe._cache_split([_batch()], encoder, torch.device("cpu"), split_name="test", show_progress=False)
    memory, memory_mask, precursors, tokens, targets = cached
    assert torch.equal(encoder.mass, torch.tensor([1000.0, 1800.0]))
    assert torch.equal(encoder.charge, torch.tensor([2, 3]))
    assert memory.shape == (2, 4, 8)
    assert memory.dtype == torch.float16
    assert memory_mask.tolist() == [[False, False, True, True], [False, True, True, True]]
    assert tokens.shape == (2, 3)
    assert targets.shape == (2, 4)
    assert precursors.shape == (2, 3)


def test_dense_cache_uses_explicit_hybrid_null_precursor():
    probe = _probe("null")
    encoder = _RecordingEncoder()
    probe._cache_split([_batch()], encoder, torch.device("cpu"), split_name="test", show_progress=False)
    assert torch.equal(encoder.mass, torch.zeros(2))
    assert torch.equal(encoder.charge, torch.zeros(2))


def test_tiny_decoder_trains_on_cached_memory():
    probe = _probe()
    probe.decoder_cfg = {"n_head": 2, "dim_feedforward": 16, "n_layers": 1}
    probe.training_cfg = {
        "batch_size": 2,
        "learning_rate": 0.01,
        "epochs": 2,
    }
    probe.seed = 0
    batch = _batch()
    encoder = _RecordingEncoder()
    cached_split = probe._cache_split([batch], encoder, torch.device("cpu"), split_name="test", show_progress=False)
    cached = {split: cached_split for split in ("train", "val", "test")}
    decoder, epochs = probe._train_decoder(cached, torch.device("cpu"), show_progress=False)
    loss = probe._teacher_forced_loss(decoder, cached_split, torch.device("cpu"), batch_size=2)
    assert epochs == 2
    assert loss > 0
