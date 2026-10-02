import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from src.collate_functions import default_filter_peaks, top_intensity_minmax
from src.data.lance_data_module import ReplaySampledDataset
from src.data.unified_tokenizer import PeptideTokenizer
from src.schedulers import CosineWarmupScheduler
from src import utils
from src.wrappers.downstream_wrappers import DeNovoTeacherForcing


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"value": self.values[index]}

    def __getitems__(self, indices):
        return [self[index] for index in indices]


class KitchenSinkDeNovoTests(unittest.TestCase):
    def test_pa11_manifest_special_token_contract_and_aliases(self):
        tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(
            "configs/tokenizers/pa11.json"
        )
        self.assertEqual(tokenizer.vocab_size, 45)
        self.assertEqual(tokenizer.pad_token_id, 43)
        self.assertEqual(tokenizer.bos_token_id, 44)
        self.assertEqual(tokenizer.eos_token_id, 44)
        self.assertIsNone(tokenizer.unk_token_id)
        for sequence in (
            "C", "C+57.021", "M", "M+15.995", "N+0.984",
            "S+79.966", "+42.011PEPTIDE",
        ):
            tokens = tokenizer.tokenize(sequence)
            self.assertGreater(tokens.numel(), 0)
        with self.assertRaisesRegex(ValueError, "Unknown peptide token"):
            tokenizer.tokenize("J")

    def test_top_intensity_minmax_keeps_unwindowed_peaks_in_mz_order(self):
        mz = torch.tensor([3001.0, 100.0, 700.0, 50.0])
        intensity = torch.tensor([2.0, 8.0, 9.0, 1.0])
        selected_mz, selected_intensity = top_intensity_minmax(mz, intensity, 2)
        self.assertTrue(torch.equal(selected_mz, torch.tensor([100.0, 700.0])))
        self.assertTrue(torch.equal(selected_intensity, torch.tensor([0.0, 1.0])))

    def test_pairwise_and_hybrid_transfer_preprocessing_contracts_are_distinct(self):
        with open("configs/master_denovo_kitchensink_v3.yaml", encoding="utf-8") as handle:
            pairwise_master = yaml.safe_load(handle)
        with open(
            "configs/master_denovo_dion_hybrid_kitchensink_v3.yaml",
            encoding="utf-8",
        ) as handle:
            hybrid_master = yaml.safe_load(handle)
        with open("configs/downstream/denovo_kitchensink_v3.yaml", encoding="utf-8") as handle:
            pairwise_task = yaml.safe_load(handle)
        with open(
            "configs/downstream/denovo_kitchensink_v3_dion_hybrid.yaml",
            encoding="utf-8",
        ) as handle:
            hybrid_task = yaml.safe_load(handle)

        self.assertEqual(pairwise_master["peak_filter_method"], "top_intensity_minmax")
        self.assertEqual(pairwise_master["max_peaks"], 150)
        self.assertEqual(pairwise_task["top_peaks"], 150)
        self.assertEqual(
            pairwise_task["denovo_tf"]["encoder_precursor_conditioning"], "none"
        )

        self.assertEqual(hybrid_master["peak_filter_method"], "default")
        self.assertEqual(hybrid_master["intensity_scaling"], "basepeak")
        self.assertEqual(hybrid_master["min_mz"], 0)
        self.assertEqual(hybrid_master["max_mz"], 2500)
        self.assertEqual(hybrid_master["max_peaks"], 200)
        self.assertEqual(hybrid_task["top_peaks"], 200)
        self.assertEqual(
            hybrid_task["denovo_tf"]["encoder_precursor_conditioning"], "conditioned"
        )

    def test_default_dino_filter_remains_windowed_basepeak(self):
        mz = torch.tensor([10.0, 100.0, 2600.0, 700.0])
        intensity = torch.tensor([2.0, 4.0, 8.0, 1.0])
        filtered_mz, filtered_intensity = default_filter_peaks(
            mz, intensity, 200, "basepeak", min_mz=0.0, max_mz=2500.0
        )
        self.assertTrue(torch.equal(filtered_mz, torch.tensor([10.0, 100.0, 700.0])))
        self.assertTrue(torch.equal(filtered_intensity, torch.tensor([0.5, 1.0, 0.25])))

    def test_replay_sample_is_deterministic_and_preserves_requested_order(self):
        replay = ReplaySampledDataset(_Dataset(["p0", "p1", "p2"]), _Dataset(["r0", "r1"]), replay_ratio=1.5, seed=7)
        initial = replay._replay_indices.copy()
        self.assertEqual(len(replay), 7)
        values = [item["value"] for item in replay.__getitems__([5, 0, 6, 2, 3])]
        self.assertEqual(values[1:4], ["p0", values[2], "p2"])
        self.assertEqual(values[0], f"r{initial[2]}")
        self.assertEqual(values[4], f"r{initial[0]}")
        replay.set_epoch(0)
        self.assertTrue((replay._replay_indices == initial).all())
        replay.set_epoch(1)
        self.assertFalse((replay._replay_indices == initial).all())

    def test_cli_max_peaks_overrides_downstream_recipe_for_lance_denovo(self):
        with open(
            "config_cluster_b/downstream/denovo_mskb_final_dion_hybrid.yaml",
            encoding="utf-8",
        ) as handle:
            config = yaml.safe_load(handle)
        args = SimpleNamespace(
            downstream_task="denovo_tf",
            max_peaks=1000,
            peak_filter_method="default",
            intensity_scaling="basepeak",
            min_mz=0.0,
            max_mz=2500.0,
            min_intensity=0.0001,
            remove_precursor_tol=0.0,
            batch_size=-1,
            num_workers=0,
            pin_mem=False,
            num_devices=1,
            num_nodes=1,
            downstream_root_dir="unused",
            downstream_train_path="",
            downstream_val_path="",
            downstream_test_path="",
        )
        with patch("src.utils.LanceDataModule") as data_module:
            utils.get_lance_peptide_data_module(
                config, args, "configurable_lance", seed=0
            )

        collate_fn = data_module.call_args.kwargs["collate_fn"]
        self.assertEqual(collate_fn.keywords["max_peaks"], 1000)

    def test_frozen_denovo_trains_decoder_only(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.use_mass = True
                self.use_charge = True

        class TinyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.reverse = False

        global_args = SimpleNamespace(
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
            freeze_encoder=True,
            max_length=100,
        )
        task = DeNovoTeacherForcing(
            TinyEncoder(),
            TinyDecoder(),
            global_args,
            tokenizer=PeptideTokenizer.from_numeric_mass_delta_manifest(
                "configs/tokenizers/pa11.json"
            ),
            task_dict={
                "batch_size": 2,
                "weight_decay": 0.0,
                "learning_rate": 1e-4,
                "warmup_steps": 0,
                "cosine_period_steps": 10,
            },
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in task.encoder.parameters()))
        optimizer = task.configure_optimizers()[0][0]
        self.assertEqual(
            {id(parameter) for group in optimizer.param_groups for parameter in group["params"]},
            {id(parameter) for parameter in task.decoder.parameters()},
        )

    def test_cosine_warmup_matches_reference_factor(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=2e-4)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, cosine_period_steps=100)
        expected = {
            0: 0.0,
            1: 2e-4 * 0.5 * (1 + math.cos(math.pi / 100)) / 10,
            10: 2e-4 * 0.5 * (1 + math.cos(math.pi / 10)),
            100: 0.0,
            150: 0.0,
        }
        for step, value in expected.items():
            self.assertAlmostEqual(scheduler.get_lr_factor(step) * 2e-4, float(value), places=12)


if __name__ == "__main__":
    unittest.main()
