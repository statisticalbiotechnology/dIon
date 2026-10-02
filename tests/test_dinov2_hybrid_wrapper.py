from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml

from src.models.custom.encoder import encoder_tiny
from src.wrappers.pretrain_wrappers import dIonPretrainWrapper


def _global_args():
    return SimpleNamespace(
        batch_size=-1,
        pin_mem=False,
        accum_iter=1,
        num_devices=1,
        num_nodes=1,
        scale_lr_by_batchsize=0,
        mask_zero_tokens=False,
        log_wandb=0,
        max_charge=10,
        clip_grad=0.0,
        max_peaks=10,
    )


class HybridDinoV2WrapperTests(unittest.TestCase):
    def _build_wrapper(self, config_name):
        task = yaml.safe_load(Path(config_name).read_text())["dion"]
        task.update(
            mlp_out_dim=8,
            mlp_hidden_dim=16,
            mlp_bottleneck_dim=8,
            pooling="average",
            batch_size=2,
            epochs=1,
            warmup_teacher_temp_epochs=0,
        )
        return dIonPretrainWrapper(
            encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
            _global_args(),
            task_dict=task,
        )

    @staticmethod
    def _batch():
        peaks = torch.tensor(
            [
                [[100.0 + i, 1.0] for i in range(10)],
                [[500.0 + i, 1.0] for i in range(10)],
            ]
        )
        return {
            "mz_array": peaks[..., 0],
            "intensity_array": peaks[..., 1],
            "peak_lengths": torch.tensor([[10], [10]]),
            "precursor_mz": torch.tensor([400.0, 700.0]),
            "precursor_mass": torch.tensor([800.0, 2100.0]),
            "precursor_charge": torch.tensor([2, 3]),
        }

    @staticmethod
    def _assert_finite_loss(wrapper, parsed):
        returns = wrapper.forward(parsed)
        dino_loss = wrapper.dino_loss(
            returns["student_out"],
            returns["teacher_out"],
            epoch=0,
            teacher_temp=0.07,
            student_groups=wrapper.dino_student_groups,
            student_group_weights=wrapper.dino_student_group_weights,
        )
        assert torch.isfinite(dino_loss)

    def test_hybrid_mixes_only_globals_and_nulls_only_locals(self):
        wrapper = self._build_wrapper(
            "configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml"
        )
        wrapper._current_mix_strength = lambda: 1.0
        parsed, _ = wrapper._parse_batch(self._batch(), Eval=False)

        self.assertEqual(wrapper.dino_student_groups, [
            "conditional_global", "conditional_global", "null_local", "null_local",
        ])
        self.assertTrue(torch.equal(parsed["charge"][:2], torch.tensor([[2, 3], [2, 3]])))
        self.assertTrue(torch.equal(parsed["mass"][:2], torch.tensor([[800.0, 2100.0], [800.0, 2100.0]])))
        self.assertTrue(torch.equal(parsed["charge"][2:], torch.zeros((2, 2), dtype=torch.long)))
        self.assertTrue(torch.equal(parsed["mass"][2:], torch.zeros((2, 2))))
        self.assertTrue((parsed["mix_provenance"][0] == wrapper.mix_aug.DISTRACTOR).any())
        self.assertTrue((parsed["mix_provenance"][1] == wrapper.mix_aug.DISTRACTOR).any())
        self.assertTrue(all((provenance == wrapper.mix_aug.ANCHOR).logical_or(
            provenance == wrapper.mix_aug.PADDING
        ).all() for provenance in parsed["mix_provenance"][2:]))
        self._assert_finite_loss(wrapper, parsed)

    def test_hybrid_ibot_uses_aligned_clean_views_and_rank_mask_tokens(self):
        task = yaml.safe_load(Path(
            "configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml"
        ).read_text())["dion"]
        task.update(
            mlp_out_dim=8,
            mlp_hidden_dim=16,
            mlp_bottleneck_dim=8,
            ibot_mlp_out_dim=8,
            ibot_mlp_hidden_dim=16,
            ibot_mlp_bottleneck_dim=8,
            ibot_loss_weight=0.1,
            ibot_mask_sample_probability=1.0,
            ibot_mask_ratio_min_max=[0.5, 0.5],
            pooling="average",
            batch_size=2,
            epochs=1,
            warmup_teacher_temp_epochs=0,
            ibot_warmup_teacher_temp_epochs=0,
        )
        wrapper = dIonPretrainWrapper(
            encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
            _global_args(),
            task_dict=task,
        )
        wrapper._current_mix_strength = lambda: 1.0
        parsed, _ = wrapper._parse_batch(self._batch(), Eval=False)

        self.assertEqual(parsed["dino_student_crop_indices"], (0, 1, 2, 3))
        self.assertEqual(parsed["ibot_student_crop_indices"], (4, 5))
        self.assertEqual(len(parsed["student_crops"]), 6)
        self.assertEqual(parsed["student_token_masks"][:4], [None] * 4)
        self.assertTrue(
            all(mask is not None for mask in parsed["student_token_masks"][4:])
        )
        for index, teacher_crop in zip(
            parsed["ibot_student_crop_indices"], parsed["teacher_crops"]
        ):
            self.assertTrue(
                torch.equal(parsed["student_crops"][index][0], teacher_crop[0])
            )
            self.assertTrue(
                torch.equal(parsed["student_crops"][index][1], teacher_crop[1])
            )
        self.assertTrue(
            torch.equal(
                parsed["charge"][2:4], torch.zeros((2, 2), dtype=torch.long)
            )
        )
        self.assertTrue(
            torch.equal(
                parsed["charge"][4:], torch.tensor([[2, 3], [2, 3]])
            )
        )

        backbone = wrapper.student.backbone
        original_run_blocks = backbone.run_blocks
        backbone.run_blocks = lambda out, **_: {"out": out}
        try:
            clean = backbone.update_embed(
                parsed["teacher_crops"][0][0],
                key_padding_mask=parsed["teacher_crops"][0][1],
                mass=parsed["real_precursor_mass"],
                charge=parsed["real_precursor_charge"],
            )
            masked = backbone.update_embed(
                parsed["teacher_crops"][0][0],
                key_padding_mask=parsed["teacher_crops"][0][1],
                mass=parsed["real_precursor_mass"],
                charge=parsed["real_precursor_charge"],
                token_mask=parsed["dense_masks"][0],
                mask_token=wrapper.student.mask_token,
                mask_rank_embeddings=wrapper.student.mask_rank_embeddings,
            )
        finally:
            backbone.run_blocks = original_run_blocks

        num_prefix = clean["num_cem_tokens"]
        mask = parsed["dense_masks"][0]
        expected = (
            wrapper.student.mask_token.view(1, 1, -1)
            + wrapper.student.mask_rank_embeddings[: mask.shape[1]].unsqueeze(0)
        )
        clean_peaks = clean["emb"][:, num_prefix:]
        masked_peaks = masked["emb"][:, num_prefix:]
        self.assertTrue(
            torch.equal(masked_peaks[mask], expected.expand_as(masked_peaks)[mask])
        )
        self.assertTrue(torch.equal(masked_peaks[~mask], clean_peaks[~mask]))

        returns = wrapper.forward(parsed)
        self.assertEqual(returns["student_out"].shape[0], 8)
        self.assertEqual(len(returns["student_patch_out"]), 2)
        self.assertTrue(all(value.ndim == 2 for value in returns["student_patch_out"]))
        dino_loss = wrapper.dino_loss(
            returns["student_out"],
            returns["teacher_out"],
            epoch=0,
            teacher_temp=0.07,
            student_groups=wrapper.dino_student_groups,
            student_group_weights=wrapper.dino_student_group_weights,
        )
        ibot_loss = wrapper.ibot_loss(
            returns["student_patch_out"],
            returns["teacher_patch_out"],
            parsed["dense_masks"],
            epoch=0,
            teacher_temp=0.07,
        )
        loss = dino_loss + 0.1 * ibot_loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(wrapper.student.mask_token.grad.abs().sum().item(), 0)
        self.assertGreater(
            wrapper.student.mask_rank_embeddings.grad.abs().sum().item(), 0
        )

    def test_hybrid_validation_compares_conditioned_and_null_global_views(self):
        wrapper = self._build_wrapper(
            "configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml"
        )
        parsed, _ = wrapper._parse_batch(self._batch(), Eval=True)
        returns = wrapper.forward(parsed)
        diagnostics = wrapper._dual_condition_global_diagnostics(
            returns, parsed, teacher_temp=0.07
        )

        expected = {
            "diag/sinkhorn_ce/conditioned_global_clean",
            "diag/sinkhorn_ce/null_global_clean",
            "diag/raw_softmax_kl/conditioned_global_clean_aligned",
            "diag/raw_softmax_kl/null_global_clean_aligned",
            "diag/sinkhorn_teacher_entropy_bits",
            "diag/sinkhorn_teacher_marginal_entropy_bits",
        }
        self.assertEqual(set(diagnostics), expected)
        for value in diagnostics.values():
            self.assertTrue(torch.isfinite(value))

    def test_h26_validation_compares_conditioned_and_null_global_views(self):
        wrapper = self._build_wrapper(
            "configs/pretrain/dion_h26_null_global_conditioned_local.yaml"
        )
        parsed, _ = wrapper._parse_batch(self._batch(), Eval=True)
        returns = wrapper.forward(parsed)
        diagnostics = wrapper._dual_condition_global_diagnostics(
            returns, parsed, teacher_temp=0.07
        )

        self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))

    def test_h26_nulls_globals_and_keeps_clean_conditioned_locals(self):
        wrapper = self._build_wrapper(
            "configs/pretrain/dion_h26_null_global_conditioned_local.yaml"
        )
        parsed, _ = wrapper._parse_batch(self._batch(), Eval=False)

        self.assertEqual(wrapper.dino_student_groups, [
            "null_global", "null_global", "conditional_local", "conditional_local",
        ])
        self.assertTrue(torch.equal(parsed["charge"][:2], torch.zeros((2, 2), dtype=torch.long)))
        self.assertTrue(torch.equal(parsed["mass"][:2], torch.zeros((2, 2))))
        self.assertTrue(torch.equal(parsed["charge"][2:], torch.tensor([[2, 3], [2, 3]])))
        self.assertTrue(torch.equal(parsed["mass"][2:], torch.tensor([[800.0, 2100.0], [800.0, 2100.0]])))
        self.assertIsNone(parsed["mix_provenance"])
        self._assert_finite_loss(wrapper, parsed)

    def test_pure_distractor_has_only_two_mixed_global_students(self):
        wrapper = self._build_wrapper(
            "configs/pretrain/dion_pure_distractor_global_100pct.yaml"
        )
        wrapper._current_mix_strength = lambda: 1.0
        parsed, _ = wrapper._parse_batch(self._batch(), Eval=False)

        self.assertEqual(len(parsed["student_crops"]), 2)
        self.assertIsNone(wrapper.dino_student_groups)
        self.assertIsNone(wrapper.dino_student_group_weights)
        self.assertTrue(torch.equal(parsed["charge"], torch.tensor([[2, 3], [2, 3]])))
        self.assertTrue(torch.equal(parsed["mass"], torch.tensor([[800.0, 2100.0], [800.0, 2100.0]])))
        self.assertTrue(all((provenance == wrapper.mix_aug.DISTRACTOR).any()
                            for provenance in parsed["mix_provenance"]))
        self._assert_finite_loss(wrapper, parsed)


if __name__ == "__main__":
    unittest.main()
