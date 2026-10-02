import unittest
from types import MethodType

import torch
import torch.nn.functional as F

from src.models.dino import DINOLoss


class GroupedDinoLossTests(unittest.TestCase):
    def test_hybrid_groups_are_averaged_then_equally_weighted(self):
        loss_fn = DINOLoss(
            out_dim=2,
            num_crops_tot=4,
            num_global_crops=2,
            warmup_teacher_temp=0.1,
            teacher_temp=0.1,
            warmup_teacher_temp_epochs=1,
            nepochs=1,
            student_temp=1.0,
        )
        teacher_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss_fn._teacher_targets = MethodType(
            lambda self, output, temperature: teacher_targets, loss_fn
        )
        student = torch.tensor(
            [[0.2, 0.8], [0.7, 0.3], [0.9, 0.1], [0.4, 0.6]]
        )
        grouped = loss_fn(
            student,
            torch.empty_like(teacher_targets),
            epoch=0,
            student_groups=[
                "conditional_global",
                "conditional_global",
                "null_local",
                "null_local",
            ],
            student_group_weights={
                "conditional_global": 0.5,
                "null_local": 0.5,
            },
        )
        ce = lambda target, logits: -(target * F.log_softmax(logits, dim=-1)).sum()
        conditional = 0.5 * (
            ce(teacher_targets[0], student[1])
            + ce(teacher_targets[1], student[0])
        )
        null_local = 0.25 * (
            ce(teacher_targets[0], student[2])
            + ce(teacher_targets[0], student[3])
            + ce(teacher_targets[1], student[2])
            + ce(teacher_targets[1], student[3])
        )
        self.assertTrue(torch.allclose(grouped, 0.5 * (conditional + null_local)))

        grouped_with_diagnostics, group_losses = loss_fn(
            student,
            torch.empty_like(teacher_targets),
            epoch=0,
            student_groups=[
                "conditional_global",
                "conditional_global",
                "null_local",
                "null_local",
            ],
            student_group_weights={
                "conditional_global": 0.5,
                "null_local": 0.5,
            },
            return_group_losses=True,
        )
        self.assertTrue(torch.allclose(grouped_with_diagnostics, grouped))
        self.assertTrue(
            torch.allclose(group_losses["conditional_global"], conditional)
        )
        self.assertTrue(torch.allclose(group_losses["null_local"], null_local))


if __name__ == "__main__":
    unittest.main()
