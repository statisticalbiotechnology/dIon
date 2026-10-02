import unittest

import torch

from src.wrappers.pretrain_wrappers import dIonPretrainWrapper


class DinoV2PrecursorChargeTests(unittest.TestCase):
    def test_accepts_real_supported_charges(self):
        dIonPretrainWrapper._validate_real_precursor_charges(
            torch.tensor([1, 2, 10], dtype=torch.long), max_charge=10
        )

    def test_rejects_reserved_or_out_of_range_real_charges(self):
        for charges in (torch.tensor([0]), torch.tensor([11])):
            with self.subTest(charges=charges.tolist()):
                with self.assertRaisesRegex(ValueError, "Charge 0 is reserved"):
                    dIonPretrainWrapper._validate_real_precursor_charges(
                        charges, max_charge=10
                    )

    def test_rejects_non_integral_real_charge(self):
        with self.assertRaisesRegex(ValueError, "integral"):
            dIonPretrainWrapper._validate_real_precursor_charges(
                torch.tensor([2.5]), max_charge=10
            )


if __name__ == "__main__":
    unittest.main()
