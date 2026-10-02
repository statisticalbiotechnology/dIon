from unittest import TestCase

import numpy as np

from src.wrappers.downstream_wrappers import DeNovoTeacherForcing


class DeNovoSpeciesMetricsTest(TestCase):
    def test_accumulates_casanovo_counts_per_species(self):
        wrapper = object.__new__(DeNovoTeacherForcing)
        wrapper._test_species_counts = {}
        wrapper._accumulate_test_species_metrics(
            ["human", "mouse", "human"],
            [(np.array([True, True]), True), (np.array([True, False]), False), (np.array([False]), False)],
            [["A", "B"], ["A", "B"], ["A"]],
            [["A", "B"], ["A", "C"], ["C"]],
        )
        self.assertEqual(wrapper._test_species_counts["human"], [2, 1, 2, 3, 3])
        self.assertEqual(wrapper._test_species_counts["mouse"], [1, 0, 1, 2, 2])

    def test_converts_counts_to_species_metrics(self):
        values = DeNovoTeacherForcing._species_metric_values([4, 2, 12, 16, 15])
        self.assertEqual(values["n_spectra"], 4.0)
        self.assertAlmostEqual(values["pep_prec"], 0.5)
        self.assertAlmostEqual(values["aa_prec"], 0.8)
        self.assertAlmostEqual(values["aa_recall"], 0.75)
