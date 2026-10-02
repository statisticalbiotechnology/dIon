from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import yaml

from src.pl_callbacks import WarmupConstantLRCallback
from src.utils import configure_callbacks
from src.wrappers.pretrain_wrappers import (
    dIonPretrainWrapper,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_CONFIG = REPO_ROOT / (
    "config_cluster_b/pretrain/"
    "dion_hybrid_distractor_null_local_100pct_indefinite.yaml"
)
MASTER_CONFIG = REPO_ROOT / (
    "config_cluster_b/master_dion_hybrid_distractor_null_local_indefinite.yaml"
)


class IndefiniteDinoScheduleTests(unittest.TestCase):
    @staticmethod
    def _global_args():
        return SimpleNamespace(
            barebones=True,
            batch_size=-1,
            scale_lr_by_batchsize=False,
            accum_iter=1,
            num_devices=1,
            num_nodes=1,
            profile_flops=False,
            early_stop=0,
        )

    def test_lr_warmup_then_constant_is_horizon_independent(self):
        task = yaml.safe_load(PRETRAIN_CONFIG.read_text())["dion"]
        common_steps = [0, 1, 20_000, 39_999, 40_000, 112_578, 500_000]
        traces = []

        for nominal_epochs in (100, 300):
            horizon_task = {**task, "epochs": nominal_epochs}
            callback = next(
                callback
                for callback in configure_callbacks(self._global_args(), horizon_task)
                if isinstance(callback, WarmupConstantLRCallback)
            )
            traces.append([callback._calculate_lr(step) for step in common_steps])

        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0][0], 0.0)
        self.assertEqual(traces[0][2], 4.0e-5)
        self.assertEqual(traces[0][4], 8.0e-5)
        self.assertEqual(traces[0][-1], 8.0e-5)

    def test_callback_updates_the_actual_optimizer_lr(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=123.0)
        trainer = SimpleNamespace(optimizers=[optimizer], global_step=50)
        module = SimpleNamespace(lr=None)
        callback = WarmupConstantLRCallback(
            lr_start=0.0,
            blr=1.0e-3,
            warmup_duration=100,
            anneal_per_step=True,
        )

        callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)

        self.assertEqual(optimizer.param_groups[0]["lr"], 5.0e-4)
        self.assertEqual(module.lr, 5.0e-4)

    def test_cosine_ema_cap_preserves_the_schedule_until_the_cap(self):
        schedule = SimpleNamespace(
            schedule_mode="cosine",
            trainer=SimpleNamespace(global_step=0),
            momentum_schedule=[0.9980, 0.9985, 0.9987, 0.9990],
            teacher_momentum_cap=0.9986,
        )

        self.assertEqual(
            dIonPretrainWrapper._current_teacher_momentum(schedule), 0.9980
        )
        schedule.trainer.global_step = 1
        self.assertEqual(
            dIonPretrainWrapper._current_teacher_momentum(schedule), 0.9985
        )
        schedule.trainer.global_step = 2
        self.assertEqual(
            dIonPretrainWrapper._current_teacher_momentum(schedule), 0.9986
        )
        schedule.trainer.global_step = 100
        self.assertEqual(
            dIonPretrainWrapper._current_teacher_momentum(schedule), 0.9986
        )

    def test_constant_ema_momentum_is_independent_of_nominal_horizon(self):
        schedule = SimpleNamespace(
            schedule_mode="indefinite",
            teacher_momentum_start=0.998,
        )

        short_run = dIonPretrainWrapper._current_teacher_momentum(schedule)
        long_run = dIonPretrainWrapper._current_teacher_momentum(schedule)

        self.assertEqual(short_run, 0.998)
        self.assertEqual(short_run, long_run)

    def test_indefinite_config_selects_constant_lr_and_preserves_ms2_values(self):
        task = yaml.safe_load(PRETRAIN_CONFIG.read_text())["dion"]
        master = yaml.safe_load(MASTER_CONFIG.read_text())
        callbacks = configure_callbacks(self._global_args(), task)

        self.assertEqual(task["schedule_mode"], "indefinite")
        self.assertEqual(task["teacher_momentum_start"], 0.998)
        self.assertNotIn("teacher_momentum_end", task)
        self.assertEqual(task["weight_decay"], 0.04)
        self.assertEqual(task["warmup_teacher_temp"], 0.03)
        self.assertEqual(task["teacher_temp"], 0.07)
        self.assertEqual(task["warmup_teacher_temp_epochs"], 10)
        self.assertEqual(task["blr"], 8.0e-5)
        self.assertNotIn("lr_end", task)
        self.assertNotIn("decay_duration", task)
        self.assertTrue(
            any(isinstance(callback, WarmupConstantLRCallback) for callback in callbacks)
        )
        self.assertEqual(master["pretrain_config"], str(PRETRAIN_CONFIG.relative_to(REPO_ROOT)))

    def test_teacher_temperature_warmup_is_horizon_independent(self):
        schedule = dIonPretrainWrapper._build_teacher_temp_step_schedule(
            warmup_teacher_temp=0.03,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=10,
            niter_per_ep=1_126,
        )

        self.assertEqual(len(schedule), 11_260)
        self.assertAlmostEqual(float(schedule[0]), 0.03)
        self.assertAlmostEqual(float(schedule[-1]), 0.07)
        def temperature_at(step):
            return float(schedule[min(step, len(schedule) - 1)])

        self.assertAlmostEqual(temperature_at(11_259), 0.07)
        self.assertAlmostEqual(temperature_at(11_260), 0.07)
        self.assertAlmostEqual(temperature_at(500_000), 0.07)


if __name__ == "__main__":
    unittest.main()
