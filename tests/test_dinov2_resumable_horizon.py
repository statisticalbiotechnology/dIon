from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import yaml

from src.pl_callbacks import StopAfterEpochCallback


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTINUATION_CONFIG = REPO_ROOT / (
    "config_cluster_b/pretrain/"
    "dion_hybrid_distractor_null_local_100pct_300epochs.yaml"
)
PART1_CONFIG = REPO_ROOT / (
    "config_cluster_b/pretrain/"
    "dion_hybrid_distractor_null_local_100pct_300epochs_part1.yaml"
)


class ResumableHorizonTests(unittest.TestCase):
    def test_stop_after_epoch_uses_completed_epoch_count(self):
        callback = StopAfterEpochCallback(stop_after_epoch=250)
        trainer = SimpleNamespace(current_epoch=248, should_stop=False)

        callback.on_train_epoch_end(trainer, pl_module=None)
        self.assertFalse(trainer.should_stop)

        trainer.current_epoch = 249
        callback.on_train_epoch_end(trainer, pl_module=None)
        self.assertTrue(trainer.should_stop)

    def test_completed_segment_is_not_run_twice_after_resume(self):
        callback = StopAfterEpochCallback(stop_after_epoch=3)
        trainer = SimpleNamespace(current_epoch=3, should_stop=False)

        callback.on_fit_start(trainer, pl_module=None)

        self.assertTrue(trainer.should_stop)
        self.assertFalse(callback._stop_requested)

    def test_stop_callback_saves_a_named_boundary_checkpoint(self):
        callback = StopAfterEpochCallback(stop_after_epoch=250)
        callback._stop_requested = True
        trainer = SimpleNamespace(
            checkpoint_callback=SimpleNamespace(dirpath="/tmp/checkpoints"),
            save_checkpoint=Mock(),
            is_global_zero=False,
        )

        callback.on_train_end(trainer, pl_module=None)
        trainer.save_checkpoint.assert_called_once_with("/tmp/checkpoints/segment_end.ckpt")

    def test_part1_preserves_the_300_epoch_schedule_horizon(self):
        continuation = yaml.safe_load(CONTINUATION_CONFIG.read_text())["dion"]
        part1 = yaml.safe_load(PART1_CONFIG.read_text())["dion"]

        self.assertEqual(continuation["epochs"], 300)
        self.assertEqual(part1["epochs"], 300)
        self.assertEqual(continuation["decay_duration"], 450_000)
        self.assertEqual(part1["decay_duration"], 450_000)
        self.assertNotIn("stop_after_epoch", continuation)
        self.assertEqual(part1["stop_after_epoch"], 250)


if __name__ == "__main__":
    unittest.main()
