import unittest

import torch
from torch.utils.data import Dataset

from src.embed_eval.extraction import extract_embeddings


class _OneSpectrumDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


def _collate(_):
    return {
        "mz_array": torch.tensor([[100.0, 200.0]]),
        "intensity_array": torch.tensor([[0.5, 1.0]]),
        "peak_lengths": torch.tensor([2]),
        "precursor_mass": torch.tensor([1000.0]),
        "precursor_charge": torch.tensor([2]),
        "precursor_mz": torch.tensor([500.0]),
        "peptide_id": ["PEPTIDE/2"],
        "partition_id": ["test"],
        "spectrum_id": ["spectrum-1"],
    }


class _RecordingEmbedder(torch.nn.Module):
    use_mass = True
    use_charge = True

    def forward(self, spectra, padding_mask, mass, charge):
        self.last_mass = mass.detach().clone()
        self.last_charge = charge.detach().clone()
        return torch.stack([mass, charge.float()], dim=1)


class EmbeddingExtractionTests(unittest.TestCase):
    def test_explicit_null_conditioning_requires_no_module_global_args(self):
        embedder = _RecordingEmbedder()

        cache = extract_embeddings(
            embedder,
            _OneSpectrumDataset(),
            collate_fn=_collate,
            batch_size=1,
            num_workers=0,
            device=torch.device("cpu"),
            precursor_conditioning="null",
        )

        self.assertTrue(torch.equal(embedder.last_mass, torch.zeros(1)))
        self.assertTrue(torch.equal(embedder.last_charge, torch.zeros(1)))
        self.assertTrue(torch.equal(cache.values, torch.zeros((1, 2))))


if __name__ == "__main__":
    unittest.main()
