import tempfile
from pathlib import Path

import lance
import pyarrow as pa
import torch
from torch import nn

from src.data.peptide_identity_batch_sampler import PeptideIdentityBatchSampler
from src.metric_learning import (
    MetricLearningEmbedder,
    MetricProjectionHead,
    supervised_contrastive_loss,
)


def test_supcon_excludes_self_and_uses_other_labels_as_negatives():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss, valid_fraction = supervised_contrastive_loss(embeddings, ["A", "A", "B"], temperature=1.0)
    # For A anchors: -log(exp(1) / (exp(1) + exp(0))); B has no positive.
    expected = -torch.log(torch.exp(torch.tensor(1.0)) / (torch.exp(torch.tensor(1.0)) + 1.0))
    assert torch.allclose(loss, expected)
    assert torch.allclose(valid_fraction, torch.tensor(2 / 3))
    loss.backward()
    assert embeddings.grad is not None


def test_supcon_handles_multiple_positives_and_rejects_empty_positive_batch():
    embeddings = torch.randn(4, 8, requires_grad=True)
    loss, valid_fraction = supervised_contrastive_loss(embeddings, ["A", "A", "A", "B"])
    assert torch.isfinite(loss)
    assert valid_fraction.item() == 0.75
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()

    try:
        supervised_contrastive_loss(torch.randn(3, 4), ["A", "B", "C"])
    except ValueError as exc:
        assert "no valid anchors" in str(exc)
    else:
        raise AssertionError("Expected no-positive SupCon batch to fail.")


def test_metric_head_is_normalized_and_propagates_to_encoder():
    class ToyEncoder(nn.Module):
        use_mass = False
        use_charge = False
        use_energy = False

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 6)

        def forward(self, spectra, key_padding_mask, mass=None, charge=None):
            del key_padding_mask, mass, charge
            return {"emb": self.linear(spectra), "mask": None}

    class MeanPooler(nn.Module):
        def forward(self, values, mask):
            del mask
            return values.mean(dim=1)

    encoder = ToyEncoder()
    head = MetricProjectionHead(6, output_dim=128)
    embedder = MetricLearningEmbedder(encoder, MeanPooler(), head)
    output = embedder(torch.randn(4, 3, 2), torch.zeros(4, 3, dtype=torch.bool))
    assert output.shape == (4, 128)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-6)
    loss, _ = supervised_contrastive_loss(output, ["A", "A", "B", "B"])
    loss.backward()
    assert encoder.linear.weight.grad is not None
    assert head.net[0].weight.grad is not None


def test_positive_batch_sampler_guarantees_two_examples_per_identity():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "train.lance"
        rows = []
        for peptide in ("A", "B", "C", "D"):
            for scan in range(3):
                rows.append(
                    {
                        "seq": peptide,
                        "peak_file": "run.mgf",
                        "scan_id": len(rows),
                        "precursor_mz": 500.0,
                        "precursor_charge": 2,
                        "mz_array": [100.0],
                        "intensity_array": [1.0],
                    }
                )
        lance.write_dataset(pa.Table.from_pylist(rows), path)
        sampler = PeptideIdentityBatchSampler(
            path,
            label_column="seq",
            peptides_per_batch=3,
            spectra_per_peptide=2,
            seed=11,
            batches_per_epoch=2,
        )
        batches = list(sampler)
        assert len(batches) == 2
        for batch in batches:
            labels = [rows[index]["seq"] for index in batch]
            assert len(batch) == 6
            assert sorted(labels.count(label) for label in set(labels)) == [2, 2, 2]


def test_hard_negative_sampler_prioritizes_strict_same_charge_neighbors():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "train.lance"
        rows = []
        for peptide, mz in (("A", 500.0000), ("B", 500.0040), ("C", 650.0000)):
            for scan in range(2):
                rows.append(
                    {
                        "seq": peptide,
                        "peak_file": "run.mgf",
                        "scan_id": len(rows),
                        "precursor_mz": mz,
                        "precursor_charge": 2,
                        "mz_array": [100.0],
                        "intensity_array": [1.0],
                    }
                )
        lance.write_dataset(pa.Table.from_pylist(rows), path)
        sampler = PeptideIdentityBatchSampler(
            path,
            label_column="seq",
            peptides_per_batch=2,
            spectra_per_peptide=2,
            strict_ppm=10.0,
            seed=7,
            batches_per_epoch=1,
        )
        batch = next(iter(sampler))
        labels = {rows[index]["seq"] for index in batch}
        diagnostics = sampler.diagnostics()
        assert labels == {"A", "B"}
        assert diagnostics["cross_peptide_same_charge_10ppm_fraction"] == 1.0
        assert diagnostics["fallback_batch_fraction"] == 0.0
        assert diagnostics["cross_peptide_same_charge_wider_than_10ppm_fraction"] == 0.0
        assert diagnostics["cross_peptide_cross_charge_global_fraction"] == 0.0


def test_hard_negative_sampler_never_overfills_odd_remaining_capacity():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "train.lance"
        rows = []
        for peptide_index in range(10):
            for scan in range(2):
                rows.append(
                    {
                        "seq": f"P{peptide_index}",
                        "peak_file": "run.mgf",
                        "scan_id": len(rows),
                        "precursor_mz": 500.0 + peptide_index * 0.003,
                        "precursor_charge": 2,
                        "mz_array": [100.0],
                        "intensity_array": [1.0],
                    }
                )
        lance.write_dataset(pa.Table.from_pylist(rows), path)
        sampler = PeptideIdentityBatchSampler(
            path,
            label_column="seq",
            peptides_per_batch=5,
            spectra_per_peptide=2,
            strict_ppm=10.0,
            seed=13,
            batches_per_epoch=20,
        )
        assert all(len(batch) == 10 for batch in sampler)


def test_downstream_wrapper_exposes_metric_projection_to_callbacks():
    from argparse import Namespace
    from src.wrappers.downstream_wrappers import SupervisedMetricLearning

    class ToyEncoder(nn.Module):
        use_mass = False
        use_charge = False
        use_energy = False
        running_units = 4

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 4)

        def forward(self, spectra, key_padding_mask, mass=None, charge=None):
            del key_padding_mask, mass, charge
            return {"emb": self.linear(spectra), "mask": None}

    class MeanPooler(nn.Module):
        def forward(self, values, mask):
            del mask
            return values.mean(dim=1)

    args = Namespace(
        batch_size=-1, pin_mem=False, accum_iter=1, num_devices=1, num_nodes=1,
        scale_lr_by_batchsize=False, mask_zero_tokens=False, log_wandb=False,
        max_charge=10, clip_grad=1.0, precursor_conditioning="conditioned",
        freeze_encoder=False,
    )
    task = {
        "batch_size": 4,
        "weight_decay": 0.0,
        "head_lr": 1e-3,
        "encoder_lr": 1e-4,
    }
    wrapper = SupervisedMetricLearning(
        ToyEncoder(), MetricProjectionHead(4), args, task_dict=task, pooler=MeanPooler()
    )
    embedder = wrapper.get_embedder(trainable=False)
    output = embedder(torch.randn(2, 3, 2), torch.zeros(2, 3, dtype=torch.bool))
    assert embedder.running_units == 128
    assert torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_canonical_callbacks_receive_metric_head_embedder():
    from argparse import Namespace
    import src.callbacks.embedding_evaluation_callback as embedding_callback_module
    import src.callbacks.pair_evaluation_callback as pair_callback_module
    from src.callbacks.embedding_evaluation_callback import SpectrumEmbeddingEvaluationCallback
    from src.callbacks.pair_evaluation_callback import PeptideIonPairEvaluationCallback

    class ToyEncoder(nn.Module):
        use_mass = False
        use_charge = False
        use_energy = False

        def forward(self, spectra, key_padding_mask, mass=None, charge=None):
            del key_padding_mask, mass, charge
            return {"emb": spectra, "mask": None}

    class MeanPooler(nn.Module):
        def forward(self, values, mask):
            del mask
            return values.mean(dim=1)

    embedder = MetricLearningEmbedder(
        ToyEncoder(), MeanPooler(), MetricProjectionHead(2, output_dim=8)
    )
    received = []

    def fake_embedding_eval(actual_embedder, *args, **kwargs):
        del args, kwargs
        received.append(("embed_eval", actual_embedder))
        return {"macro": {"map": 0.5}}

    def fake_pair_eval(actual_embedder, *args, **kwargs):
        del args, kwargs
        received.append(("pair_eval", actual_embedder))
        return {"pair_sets": {"same_charge_10ppm": {"macro": {"roc_auc": 0.5}}}}

    originals = (
        embedding_callback_module.evaluate_embedder,
        embedding_callback_module.flatten_report_for_logging,
        pair_callback_module.evaluate_pair_embedder,
        pair_callback_module.flatten_pair_report_for_logging,
    )
    embedding_callback_module.evaluate_embedder = fake_embedding_eval
    embedding_callback_module.flatten_report_for_logging = lambda *args, **kwargs: {}
    pair_callback_module.evaluate_pair_embedder = fake_pair_eval
    pair_callback_module.flatten_pair_report_for_logging = lambda *args, **kwargs: {}

    args = Namespace(
        embedding_baseline="model", probe_on_fit_start=True, probe_every_n_steps=1
    )
    config = {"online": {"enabled": True}, "evaluations": [{"name": "test"}]}

    class Module:
        device = torch.device("cpu")

        def get_embedder(self, trainable=False):
            assert not trainable
            return embedder

    class Logger:
        def log_metrics(self, metrics, step):
            del metrics, step

    class Trainer:
        logger = Logger()
        global_step = 0

    module = Module()
    trainer = Trainer()
    try:
        SpectrumEmbeddingEvaluationCallback(config, args)._run(trainer, module)
        PeptideIonPairEvaluationCallback(config, args)._run(trainer, module)
    finally:
        (
            embedding_callback_module.evaluate_embedder,
            embedding_callback_module.flatten_report_for_logging,
            pair_callback_module.evaluate_pair_embedder,
            pair_callback_module.flatten_pair_report_for_logging,
        ) = originals

    assert received == [("embed_eval", embedder), ("pair_eval", embedder)]
    output = received[0][1](
        torch.randn(3, 4, 2), torch.zeros(3, 4, dtype=torch.bool)
    )
    assert output.shape == (3, 8)
    assert torch.allclose(output.norm(dim=-1), torch.ones(3), atol=1e-6)


# Keep direct execution compatible with this repository's lightweight test style.
if __name__ == "__main__":
    test_supcon_excludes_self_and_uses_other_labels_as_negatives()
    test_supcon_handles_multiple_positives_and_rejects_empty_positive_batch()
    test_metric_head_is_normalized_and_propagates_to_encoder()
    test_positive_batch_sampler_guarantees_two_examples_per_identity()
    test_hard_negative_sampler_prioritizes_strict_same_charge_neighbors()
    test_hard_negative_sampler_never_overfills_odd_remaining_capacity()
    test_downstream_wrapper_exposes_metric_projection_to_callbacks()
    test_canonical_callbacks_receive_metric_head_embedder()
    print("passed: tests/test_supervised_metric_learning.py")
