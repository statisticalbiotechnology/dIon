"""Small reusable training schedules."""

from __future__ import annotations

import math
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up multiplied by a cosine half-period, stepped per optimizer update."""

    def __init__(self, optimizer, warmup_steps: int, cosine_period_steps: int):
        if cosine_period_steps <= 0:
            raise ValueError("cosine_period_steps must be positive.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        self.warmup_steps = int(warmup_steps)
        self.cosine_period_steps = int(cosine_period_steps)
        super().__init__(optimizer)

    def get_lr_factor(self, step: int) -> float:
        cosine_step = min(int(step), self.cosine_period_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * cosine_step / self.cosine_period_steps))
        if self.warmup_steps and step <= self.warmup_steps:
            factor *= step / self.warmup_steps
        return factor

    def get_lr(self):
        return [base_lr * self.get_lr_factor(self.last_epoch) for base_lr in self.base_lrs]
