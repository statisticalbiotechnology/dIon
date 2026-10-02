from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import yaml

from src.models.custom.encoder import encoder_tiny
from src.wrappers.pretrain_wrappers import dIonPretrainWrapper


class GramRefinementWrapperTests(unittest.TestCase):
    @staticmethod
    def _args(encoder_weights="/tmp/late-refinement.ckpt"):
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
            encoder_weights=encoder_weights,
            resume=False,
        )

    @staticmethod
    def _task():
        task = yaml.safe_load(
            Path("configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml").read_text()
        )["dion"]
        task.update(
            mlp_out_dim=8,
            mlp_hidden_dim=16,
            mlp_bottleneck_dim=8,
            pooling="average",
            batch_size=2,
            epochs=1,
            warmup_teacher_temp_epochs=0,
            gram_enabled=True,
            gram_refinement_checkpoint_mode="weights_only",
            gram_loss_weight=0.25,
            gram_warmup_steps=0,
            gram_clean_crop_index=0,
            gram_teacher_refresh=False,
        )
        return task

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

    def test_gram_teacher_is_frozen_backbone_and_loss_is_additive(self):
        task = self._task()
        with TemporaryDirectory() as directory:
            teacher_path = Path(directory) / "dense-good.ckpt"
            source = dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(),
                task_dict={**task, "gram_enabled": False},
            )
            torch.save(
                {
                    "state_dict": {
                        f"teacher.backbone.{key}": value
                        for key, value in source.teacher.backbone.state_dict().items()
                    }
                },
                teacher_path,
            )
            task["gram_teacher_checkpoint"] = str(teacher_path)
            wrapper = dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(),
                task_dict=task,
            )

            self.assertFalse(wrapper.gram_teacher.training)
            self.assertFalse(
                any(parameter.requires_grad for parameter in wrapper.gram_teacher.parameters())
            )
            # A late base checkpoint has no Gram-teacher keys. The wrapper must
            # preserve the separately loaded dense-good teacher during this load.
            missing, unexpected = wrapper.load_state_dict(source.state_dict(), strict=False)
            self.assertEqual(missing, [])
            self.assertEqual(unexpected, [])
            for key, expected in source.teacher.backbone.state_dict().items():
                self.assertTrue(torch.equal(wrapper.gram_teacher.state_dict()[key], expected))

            wrapper._trainer = SimpleNamespace(global_step=0, current_epoch=0)
            wrapper.gram_refinement_start_step.fill_(0)
            parsed, _ = wrapper._parse_batch(self._batch(), Eval=False)
            returned = wrapper.forward(parsed)
            loss, _, _, _, gram_loss, gram_weight, _ = wrapper._get_losses(returned, parsed)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(gram_loss))
            self.assertTrue(gram_loss.requires_grad)
            self.assertFalse(returned["gram_teacher_tokens"].requires_grad)
            self.assertEqual(gram_weight, 0.25)
            self.assertTrue(
                torch.equal(
                    returned["gram_padding_mask"], parsed["teacher_crops"][0][1]
                )
            )
            loss.backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in wrapper.student.backbone.parameters())
            )

    def test_refinement_mode_must_match_resume_flag(self):
        task = self._task()
        task["gram_teacher_checkpoint"] = "/tmp/dense-good.ckpt"
        task["gram_refinement_checkpoint_mode"] = "resume"
        with self.assertRaisesRegex(ValueError, "conflicts with --resume"):
            dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(),
                task_dict=task,
            )

    def test_same_checkpoint_roles_are_rejected(self):
        task = self._task()
        task["gram_teacher_checkpoint"] = "/tmp/same.ckpt"
        with self.assertRaisesRegex(ValueError, "must differ"):
            dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(encoder_weights="/tmp/same.ckpt"),
                task_dict=task,
            )


    def test_cosine_gram_warmup_and_refresh_schedule(self):
        task = self._task()
        task.update(
            gram_loss_weight=2.0,
            gram_warmup_steps=4,
            gram_warmup_schedule="cosine",
            gram_teacher_refresh=True,
            gram_teacher_refresh_first_step=2,
            gram_teacher_refresh_every_n_steps=2,
            gram_teacher_refresh_max_updates=2,
        )
        with TemporaryDirectory() as directory:
            teacher_path = Path(directory) / "dense-good.ckpt"
            source = dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(),
                task_dict={**task, "gram_enabled": False},
            )
            torch.save(
                {
                    "state_dict": {
                        f"teacher.backbone.{key}": value
                        for key, value in source.teacher.backbone.state_dict().items()
                    }
                },
                teacher_path,
            )
            task["gram_teacher_checkpoint"] = str(teacher_path)
            wrapper = dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(),
                task_dict=task,
            )
            wrapper._trainer = SimpleNamespace(global_step=0, current_epoch=0)
            wrapper.gram_refinement_start_step.fill_(0)
            wrapper.gram_teacher_refresh_start_step.fill_(0)

            self.assertAlmostEqual(wrapper._current_gram_weight(), 1.0 - 2**-0.5)
            wrapper._trainer.global_step = 1
            self.assertAlmostEqual(wrapper._current_gram_weight(), 1.0)
            wrapper._trainer.global_step = 3
            self.assertAlmostEqual(wrapper._current_gram_weight(), 2.0)

            with torch.no_grad():
                for parameter in wrapper.teacher.backbone.parameters():
                    parameter.fill_(0.125)
            wrapper._trainer.global_step = 0
            self.assertFalse(wrapper._maybe_refresh_gram_teacher())
            wrapper._trainer.global_step = 1
            self.assertTrue(wrapper._maybe_refresh_gram_teacher())
            self.assertEqual(wrapper.gram_teacher_refresh_count.item(), 1)
            for expected, actual in zip(
                wrapper.teacher.backbone.parameters(), wrapper.gram_teacher.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))

            with torch.no_grad():
                for parameter in wrapper.teacher.backbone.parameters():
                    parameter.fill_(0.25)
            wrapper._trainer.global_step = 3
            self.assertTrue(wrapper._maybe_refresh_gram_teacher())
            self.assertEqual(wrapper.gram_teacher_refresh_count.item(), 2)
            wrapper._trainer.global_step = 5
            self.assertFalse(wrapper._maybe_refresh_gram_teacher())


    def test_legacy_checkpoint_gram_clock_starts_after_restore(self):
        task = self._task()
        task.update(
            gram_loss_weight=2.0,
            gram_warmup_steps=1000,
            gram_warmup_schedule="cosine",
        )
        with TemporaryDirectory() as directory:
            teacher_path = Path(directory) / "dense-good.ckpt"
            source = dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(),
                task_dict={**task, "gram_enabled": False},
            )
            torch.save(
                {"state_dict": {
                    f"teacher.backbone.{key}": value
                    for key, value in source.teacher.backbone.state_dict().items()
                }},
                teacher_path,
            )
            task["gram_teacher_checkpoint"] = str(teacher_path)
            wrapper = dIonPretrainWrapper(
                encoder_tiny(use_mass=True, use_charge=True, max_charge=10),
                self._args(),
                task_dict=task,
            )

            wrapper._trainer = SimpleNamespace(global_step=315560, current_epoch=280)
            self.assertEqual(wrapper.gram_refinement_start_step.item(), -1)
            wrapper.on_train_start()
            self.assertEqual(wrapper.gram_refinement_start_step.item(), 315560)
            self.assertAlmostEqual(wrapper._current_gram_weight(), 0.00000493479)


if __name__ == "__main__":
    unittest.main()
