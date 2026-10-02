import argparse
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_denovo_mztab.py"
SPEC = importlib.util.spec_from_file_location("validate_denovo_mztab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ValidateDenovoMztabTest(unittest.TestCase):
    def test_scores_all_rows_and_writes_confidence_curve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mztab = root / "predictions.mzTab"
            curve = root / "curve.csv"
            with mztab.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["MTD", "mzTab-version", "1.0.0"])
                writer.writerow(["PSH", *MODULE.PSM_COLUMNS])
                base = ["null"] * len(MODULE.PSM_COLUMNS)
                rows = [
                    ("A,G", "0.9", "A,G"),
                    ("A", "0.5", "A,G"),
                    ("null", "null", "A,G"),
                ]
                for index, (sequence, score, truth) in enumerate(rows, start=1):
                    values = base.copy()
                    values[0] = sequence
                    values[1] = str(index)
                    values[7] = score
                    values[13] = f"test.mgf:{index}"
                    values[19] = truth
                    writer.writerow(["PSM", *values])

            summary = MODULE.validate_and_score(
                argparse.Namespace(
                    mztab=mztab,
                    tokenizer_manifest=Path("configs/tokenizers/pa11.json"),
                    expected_rows=3,
                    curve_csv=curve,
                    summary_json=None,
                )
            )

            self.assertEqual(summary["psm_rows"], 3)
            self.assertEqual(summary["unique_spectra"], 3)
            self.assertEqual(summary["null_confidence_rows"], 1)
            self.assertAlmostEqual(summary["peptide_precision_100pct_coverage"], 1 / 3)
            self.assertTrue(0 < summary["precision_coverage_auc"] < 1)
            self.assertEqual(len(curve.read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
