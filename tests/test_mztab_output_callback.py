from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.pl_callbacks import MztabOutputCallback


class MztabOutputCallbackTest(TestCase):
    def test_gathers_python_prediction_objects_across_ranks(self):
        callback = MztabOutputCallback("/tmp", SimpleNamespace())
        local_outputs = {
            "predictions": [{"peptide": ["A"], "peptide_score": 0.9}],
            "peptides_true": [["A"]],
        }
        remote_outputs = {
            "predictions": [{"peptide": ["B"], "peptide_score": 0.8}],
            "peptides_true": [["B"]],
        }
        module = SimpleNamespace(trainer=SimpleNamespace(world_size=2))

        def gather_object(destination, source):
            self.assertIs(source, local_outputs)
            destination[:] = [local_outputs, remote_outputs]

        with patch("src.pl_callbacks.dist.all_gather_object", side_effect=gather_object):
            gathered = callback._gather_outputs(local_outputs, module)

        self.assertEqual(gathered["predictions"], local_outputs["predictions"] + remote_outputs["predictions"])
        self.assertEqual(gathered["peptides_true"], local_outputs["peptides_true"] + remote_outputs["peptides_true"])

    def test_skips_distributed_sampler_padding_duplicates(self):
        callback = MztabOutputCallback("/tmp", SimpleNamespace())
        callback.writer = MagicMock()
        outputs = {
            "predictions": [{
                "peptide": ["A"], "peak_file": "test.mgf", "scan_id": 7,
                "precursor_charge": 2.0, "precursor_mz": 100.0,
                "peptide_score": 0.9, "aa_scores": [0.9],
            }],
            "peptides_true": [["A"]],
        }
        trainer = SimpleNamespace(is_global_zero=True)
        module = SimpleNamespace(
            trainer=SimpleNamespace(world_size=1),
            peptide_mass_calculator=SimpleNamespace(mass=lambda peptide, charge: 100.0),
        )

        callback.on_test_batch_end(trainer, module, outputs, batch={}, batch_idx=0)
        callback.on_test_batch_end(trainer, module, outputs, batch={}, batch_idx=1)

        self.assertEqual(callback.writer.write_psm.call_count, 1)
        callback.writer.flush.assert_called()
