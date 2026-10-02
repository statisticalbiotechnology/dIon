import unittest

import torch

from src.models.dino.gram import per_spectrum_gram_mse


class GramAnchorLossTests(unittest.TestCase):
    def test_identical_geometry_has_zero_loss(self):
        tokens = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [4.0, 4.0]]], requires_grad=True
        )
        loss = per_spectrum_gram_mse(
            tokens, tokens.detach(), torch.tensor([[False, False, True]])
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))

    def test_padding_is_ignored(self):
        student = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, -7.0]]])
        teacher = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [-2.0, 5.0]]])
        loss = per_spectrum_gram_mse(
            student, teacher, torch.tensor([[False, False, True]])
        )
        self.assertTrue(torch.allclose(loss, torch.tensor(0.0)))

    def test_each_spectrum_is_equally_weighted(self):
        student = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            ]
        )
        teacher = student.clone()
        teacher[0, 1] = torch.tensor([1.0, 0.0])
        padding = torch.tensor([[False, False, True], [False, False, False]])
        combined = per_spectrum_gram_mse(student, teacher, padding)
        first = per_spectrum_gram_mse(student[:1], teacher[:1], padding[:1])
        second = per_spectrum_gram_mse(student[1:], teacher[1:], padding[1:])
        self.assertTrue(torch.allclose(combined, 0.5 * (first + second)))


if __name__ == "__main__":
    unittest.main()
