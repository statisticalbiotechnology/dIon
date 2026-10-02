"""Tests for the soft-label ordinal target utilities."""

import math
import unittest

import torch

from src.soft_ordinal import (
    BinSpec,
    decode_expectation,
    decode_mode_refined,
    decode_std,
    soft_cross_entropy,
    soft_labels,
)

SPEC = BinSpec(low=0.0, high=1.0, n_bins=64, sigma=0.024)


class BinSpecTests(unittest.TestCase):
    def test_geometry(self):
        self.assertAlmostEqual(SPEC.width, 1 / 64)
        centers = SPEC.centers()
        self.assertEqual(centers.shape, (64,))
        self.assertAlmostEqual(float(centers[0]), 1 / 128, places=6)
        self.assertAlmostEqual(float(centers[-1]), 1 - 1 / 128, places=6)

    def test_validation(self):
        for kwargs in (
            dict(low=0.0, high=1.0, n_bins=1, sigma=0.1),
            dict(low=1.0, high=1.0, n_bins=8, sigma=0.1),
            dict(low=0.0, high=1.0, n_bins=8, sigma=0.0),
        ):
            with self.assertRaises(ValueError):
                BinSpec(**kwargs)


class SoftLabelTests(unittest.TestCase):
    def test_rows_normalized_including_at_edges(self):
        values = torch.tensor([0.0, 0.5, 1.0, -3.0, 7.0])
        target = soft_labels(values, SPEC)
        self.assertEqual(target.shape, (5, 64))
        self.assertTrue(torch.allclose(target.sum(-1), torch.ones(5), atol=1e-6))
        self.assertTrue((target >= 0).all())

    def test_peak_sits_on_the_true_value(self):
        for value in (0.1, 0.37, 0.82):
            target = soft_labels(torch.tensor([value]), SPEC)
            peak = int(target.argmax(-1))
            self.assertAlmostEqual(float(SPEC.centers()[peak]), value,
                                   delta=SPEC.width)

    def test_sigma_controls_width(self):
        narrow = soft_labels(torch.tensor([0.5]), BinSpec(0.0, 1.0, 64, 0.01))
        wide = soft_labels(torch.tensor([0.5]), BinSpec(0.0, 1.0, 64, 0.10))
        self.assertGreater(float(narrow.max()), float(wide.max()))


class DecoderTests(unittest.TestCase):
    def test_expectation_round_trips_the_soft_label(self):
        values = torch.tensor([0.15, 0.4, 0.63, 0.88])
        # Treat the soft label's log as logits: decoding must return the value.
        logits = torch.log(soft_labels(values, SPEC) + 1e-12)
        decoded = decode_expectation(logits, SPEC)
        self.assertTrue(torch.allclose(decoded, values, atol=SPEC.width))

    def test_std_tracks_sigma_and_flags_uncertainty(self):
        logits = torch.log(soft_labels(torch.tensor([0.5]), SPEC) + 1e-12)
        self.assertAlmostEqual(float(decode_std(logits, SPEC)), SPEC.sigma,
                               delta=0.5 * SPEC.width)
        flat = torch.zeros(1, SPEC.n_bins)          # maximally uncertain
        self.assertGreater(float(decode_std(flat, SPEC)),
                           float(decode_std(logits, SPEC)))

    def test_mode_decoder_beats_expectation_when_bimodal(self):
        # Dominant mode at 0.2, secondary at 0.9: the expectation is dragged
        # between them, the mode decoder stays on the dominant peak.
        logits = torch.log(
            0.8 * soft_labels(torch.tensor([0.2]), SPEC)
            + 0.2 * soft_labels(torch.tensor([0.9]), SPEC) + 1e-12
        )
        self.assertLess(abs(float(decode_mode_refined(logits, SPEC)) - 0.2),
                        abs(float(decode_expectation(logits, SPEC)) - 0.2))

    def test_mode_decoder_subbin_resolution(self):
        # A value deliberately off-centre within its bin: interpolation should
        # land closer than the bin centre does.
        value = float(SPEC.centers()[20]) + 0.3 * SPEC.width
        logits = torch.log(soft_labels(torch.tensor([value]), SPEC) + 1e-12)
        refined = float(decode_mode_refined(logits, SPEC))
        self.assertLess(abs(refined - value), 0.5 * SPEC.width)


class LossTests(unittest.TestCase):
    def test_minimized_at_the_target(self):
        target = soft_labels(torch.tensor([0.45]), SPEC)
        good = soft_cross_entropy(torch.log(target + 1e-12), target)
        bad = soft_cross_entropy(
            torch.log(soft_labels(torch.tensor([0.9]), SPEC) + 1e-12), target)
        self.assertLess(float(good), float(bad))
        self.assertGreater(float(good), 0.0)

    def test_entropy_lower_bound(self):
        # Cross-entropy against a soft target cannot go below its entropy.
        target = soft_labels(torch.tensor([0.45]), SPEC)
        entropy = -(target * torch.log(target + 1e-12)).sum()
        loss = soft_cross_entropy(torch.log(target + 1e-12), target)
        self.assertAlmostEqual(float(loss), float(entropy), places=4)

    def test_gradients_flow_and_shape_is_checked(self):
        logits = torch.zeros(4, SPEC.n_bins, requires_grad=True)
        target = soft_labels(torch.tensor([0.1, 0.3, 0.5, 0.7]), SPEC)
        soft_cross_entropy(logits, target).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(-1) > 0).all())
        with self.assertRaises(ValueError):
            soft_cross_entropy(torch.zeros(4, 8), target)

    def test_batch_independence(self):
        values = torch.tensor([0.2, 0.8])
        batched = soft_labels(values, SPEC)
        for i, value in enumerate(values):
            single = soft_labels(value.reshape(1), SPEC)
            self.assertTrue(torch.allclose(batched[i], single[0], atol=1e-7))


if __name__ == "__main__":
    unittest.main()
