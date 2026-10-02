import unittest
from unittest.mock import MagicMock, patch

from pytorch_lightning.callbacks import ModelCheckpoint

from src.main import safe_load_pretrain_wrapper
from src.pl_callbacks import AlwaysSaveLastCheckpoint


class AlwaysSaveLastCheckpointTests(unittest.TestCase):
    def test_saves_last_when_monitored_top_k_did_not_save(self):
        callback = AlwaysSaveLastCheckpoint(
            dirpath="unused", monitor="metric", mode="min", save_top_k=1, save_last=True
        )
        trainer = MagicMock()
        trainer.current_epoch = 4
        trainer.global_step = 50
        callback._last_global_step_saved = 40

        with (
            patch.object(ModelCheckpoint, "on_validation_end"),
            patch.object(callback, "_should_skip_saving_checkpoint", return_value=False),
            patch.object(callback, "_should_save_on_train_epoch_end", return_value=False),
            patch.object(callback, "_monitor_candidates", return_value={"metric": 1.0}),
            patch.object(callback, "_save_last_checkpoint") as save_last,
        ):
            callback.on_validation_end(trainer, MagicMock())

        save_last.assert_called_once_with(trainer, {"metric": 1.0})

    def test_does_not_duplicate_last_when_top_k_saved_this_step(self):
        callback = AlwaysSaveLastCheckpoint(
            dirpath="unused", monitor="metric", mode="min", save_top_k=1, save_last=True
        )
        trainer = MagicMock()
        trainer.current_epoch = 4
        trainer.global_step = 50
        callback._last_global_step_saved = 50

        with (
            patch.object(ModelCheckpoint, "on_validation_end"),
            patch.object(callback, "_should_skip_saving_checkpoint", return_value=False),
            patch.object(callback, "_should_save_on_train_epoch_end", return_value=False),
            patch.object(callback, "_save_last_checkpoint") as save_last,
        ):
            callback.on_validation_end(trainer, MagicMock())

        save_last.assert_not_called()


class PretrainCheckpointLoadingTests(unittest.TestCase):
    def test_ignores_only_gram_refinement_teacher_and_metadata(self):
        checkpoint = {
            "state_dict": {
                "expected": 1,
                "gram_teacher.backbone.weight": 2,
                "gram_refinement_start_step": 3,
            }
        }
        module = MagicMock()
        module.load_state_dict.return_value = (
            [],
            ["gram_teacher.backbone.weight", "gram_refinement_start_step"],
        )

        with patch("src.main.load_checkpoint_safely", return_value=checkpoint):
            self.assertIs(safe_load_pretrain_wrapper(lambda **_: module, "unused"), module)

        module.load_state_dict.assert_called_once_with(checkpoint["state_dict"], strict=False)

    def test_rejects_non_gram_unexpected_checkpoint_keys(self):
        module = MagicMock()
        module.load_state_dict.return_value = ([], ["not_a_refinement_key"])

        with (
            patch("src.main.load_checkpoint_safely", return_value={"state_dict": {}}),
            self.assertRaisesRegex(RuntimeError, "not_a_refinement_key"),
        ):
            safe_load_pretrain_wrapper(lambda **_: module, "unused")
