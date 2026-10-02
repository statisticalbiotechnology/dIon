import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.parse_args import create_output_dirs


class DistributedOutputDirectoryTests(unittest.TestCase):
    def test_slurm_ranks_share_the_timestamped_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rank_zero = SimpleNamespace(
                output_dir=str(root / "checkpoints" / "checkpoint"),
                log_dir=str(root / "logs" / "log"),
            )
            worker = SimpleNamespace(
                output_dir=rank_zero.output_dir,
                log_dir=rank_zero.log_dir,
            )
            env = {"SLURM_JOB_ID": "unit-test", "SLURM_NTASKS": "2"}

            with patch.dict(os.environ, env, clear=False):
                create_output_dirs(rank_zero, is_main_process=True)
                create_output_dirs(worker, is_main_process=False)

            self.assertEqual(worker.output_dir, rank_zero.output_dir)
            self.assertEqual(worker.log_dir, rank_zero.log_dir)
            self.assertTrue(Path(rank_zero.output_dir).is_dir())
            self.assertTrue(Path(rank_zero.log_dir).is_dir())

    def test_single_rank_keeps_the_existing_local_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=str(Path(tmpdir) / "checkpoint"),
                log_dir=str(Path(tmpdir) / "log"),
            )
            with patch.dict(os.environ, {"SLURM_JOB_ID": "unit-test"}, clear=False):
                create_output_dirs(args, is_main_process=True)

            self.assertTrue(Path(args.output_dir).is_dir())
            self.assertTrue(Path(args.log_dir).is_dir())
