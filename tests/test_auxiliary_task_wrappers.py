"""Focused contracts for SQA-derived auxiliary downstream tasks."""

from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest

import lance
import pyarrow as pa
import torch

from src.data.lance_datasets import SafeLanceDataset
from src.models.custom.heads import (
    linear_ordinal_head,
    linear_regression_head,
    sqa_mlp_head,
)
from src.wrappers.downstream_wrappers import (
    ChimericityAssessment,
    OxidizedMethionineAssessment,
    RetentionTimeProbe,
)


class TinyEmbedder(torch.nn.Module):
    def __init__(self, dimension=8):
        super().__init__()
        self.projection = torch.nn.Linear(2, dimension)
        self.running_units = dimension

    def forward(self, spectra, **kwargs):
        del kwargs
        return self.projection(spectra.mean(dim=1))


def args(*, freeze_encoder: bool) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=-1,
        pin_mem=False,
        accum_iter=1,
        num_devices=1,
        num_nodes=1,
        scale_lr_by_batchsize=False,
        mask_zero_tokens=False,
        log_wandb=False,
        max_charge=10,
        clip_grad=0.0,
        freeze_encoder=freeze_encoder,
        embedding_dir=None,
        encoder_model="tiny",
        encoder_weights=None,
        max_peaks=8,
        downstream_root_dir="/tmp/auxiliary-test",
        precursor_conditioning="conditioned",
    )


def binary_config(**extra):
    config = {
        "batch_size": 2,
        "weight_decay": 0.0,
        "head_lr": 1e-3,
        "encoder_lr": 1e-3,
        "target_column": "label",
    }
    config.update(extra)
    return config


def batch(*, include_mask=False):
    result = {
        "mz_array": torch.tensor([[100.0, 200.0], [110.0, 210.0]]),
        "intensity_array": torch.tensor([[0.2, 1.0], [1.0, 0.3]]),
        "precursor_mass": torch.tensor([500.0, 600.0]),
        "precursor_charge": torch.tensor([2.0, 3.0]),
        "peak_lengths": torch.tensor([[2], [2]], dtype=torch.int32),
        "label": torch.tensor([0, 1]),
        "nrt_aligned": torch.tensor([0.2, 0.8]),
        "index": torch.tensor([0, 1]),
    }
    if include_mask:
        result["matched_backbone"] = torch.tensor([1, 0])
    return result


class AuxiliaryTaskWrapperTests(unittest.TestCase):
    def test_binary_tasks_reuse_sqa_head_and_alias_configured_target(self):
        encoder = TinyEmbedder()
        head = sqa_mlp_head(d_model=encoder.running_units)
        task = ChimericityAssessment(encoder, head, global_args=args(freeze_encoder=False), task_dict=binary_config())
        parsed, _ = task._parse_batch(batch())
        self.assertTrue(torch.equal(parsed["quality"], torch.tensor([0, 1])))
        logits = task.forward(parsed, batch_idx=0)
        loss, _ = task._get_train_stats(logits, parsed)
        loss.backward()
        self.assertIsNotNone(encoder.projection.weight.grad)
        self.assertIsNotNone(head.final.weight.grad)

    def test_oxidation_accepts_matched_backbone_as_evaluation_only_field(self):
        task = OxidizedMethionineAssessment(
            TinyEmbedder(), sqa_mlp_head(d_model=8), global_args=args(freeze_encoder=False),
            task_dict=binary_config(evaluation_mask_column="matched_backbone"),
        )
        parsed, _ = task._parse_batch(batch(include_mask=True))
        self.assertTrue(torch.equal(parsed["evaluation_mask"], torch.tensor([True, False])))
        self.assertNotIn("matched_backbone", parsed["mz_ab"] if isinstance(parsed["mz_ab"], dict) else {})

    def test_rt_is_frozen_linear_soft_ordinal_probe(self):
        ordinal = {"low": 0.0, "high": 1.0, "n_bins": 64, "sigma": 0.02}
        encoder = TinyEmbedder()
        head = linear_ordinal_head(d_model=encoder.running_units, num_classes=64)
        task = RetentionTimeProbe(
            encoder, head, global_args=args(freeze_encoder=True),
            task_dict=binary_config(target_column="nrt_aligned", soft_ordinal=ordinal),
        )
        self.assertIsInstance(task.decoder, torch.nn.Linear)
        logits = torch.zeros(2, 64, requires_grad=True)
        loss, prediction = task._loss_and_prediction(logits, torch.tensor([0.2, 0.8]))
        loss.backward()
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(torch.isfinite(logits.grad).all())
        regression = RetentionTimeProbe(
            TinyEmbedder(), linear_regression_head(d_model=8),
            global_args=args(freeze_encoder=True),
            task_dict=binary_config(target_column="nrt_aligned", prediction_mode="regression"),
        )
        regression_loss, regression_prediction = regression._loss_and_prediction(
            torch.tensor([[0.1], [0.9]], requires_grad=True),
            torch.tensor([0.2, 0.8]),
        )
        self.assertEqual(regression_prediction.shape, (2,))
        self.assertGreater(regression_loss.item(), 0.0)
        with self.assertRaises(ValueError):
            RetentionTimeProbe(
                TinyEmbedder(), linear_ordinal_head(d_model=8, num_classes=64),
                global_args=args(freeze_encoder=False),
                task_dict=binary_config(target_column="nrt_aligned", soft_ordinal=ordinal),
            )

    def test_optional_lance_indices_are_emitted_without_changing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny.lance"
            lance.write_dataset(pa.Table.from_pylist([{
                "mz_array": [100.0], "intensity_array": [1.0], "label": 1,
            }]), path)
            item = SafeLanceDataset(path, columns=["label"], include_index=True)[0]
        self.assertEqual(item["index"], 0)
        self.assertEqual(set(item), {"mz_array", "intensity_array", "label", "index"})

    def test_legacy_dataset_without_optional_index_flag_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/tiny.lance"
            lance.write_dataset(pa.Table.from_pylist([{
                "mz_array": [100.0], "intensity_array": [1.0], "label": 1,
            }]), path)
            dataset = SafeLanceDataset(path, columns=["label"])
            del dataset.include_index
            item = dataset[0]
            items = dataset.__getitems__([0])
        self.assertNotIn("index", item)
        self.assertNotIn("index", items[0])


if __name__ == "__main__":
    unittest.main()
