"""Soft-label ordinal targets: regression posed as binned classification.

Transformers regress badly -- an MSE head on a scalar gives poorly conditioned
gradients and is dominated by outliers. The standard fix is to discretize the
target axis, train a classifier over bins against a *soft* target distribution
centred on the true value, and decode a continuous prediction as the
expectation over bin centres. AlphaFold's distogram is the canonical example;
DORN (depth) and label-distribution learning (age estimation) are the same
family.

Two properties matter beyond conditioning:

* the head emits a distribution, so its width is a free uncertainty estimate,
  and genuinely bimodal cases can be represented rather than averaged away;
* the kernel width need not be a hyperparameter. For a measured quantity it
  should be the measurement's own replicate spread, which makes the soft label
  an honest statement of what the target actually pins down. For the retention
  time benchmark, ``sigma`` is the replicate spread of the same peptidoform
  across runs, recorded in that benchmark's manifest.

Typical use, with the benchmark's recorded bin spec:

    spec = BinSpec(low=0.0, high=1.0, n_bins=64, sigma=0.024)
    target = soft_labels(nrt, spec)            # (B, 64), rows sum to 1
    loss = soft_cross_entropy(logits, target)
    prediction = decode_expectation(logits, spec)     # back to nrt units
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BinSpec:
    """Discretization of a bounded scalar target.

    Args:
        low, high: target range; values outside are clamped, not dropped.
        n_bins: number of bins. Bins much narrower than ``sigma`` only model
            noise; much wider throws away resolution. ``n_bins`` such that the
            bin width is roughly ``sigma`` is a sensible default.
        sigma: kernel width of the soft label, in target units.
    """

    low: float
    high: float
    n_bins: int
    sigma: float

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if self.high <= self.low:
            raise ValueError("high must exceed low.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")

    @property
    def width(self) -> float:
        return (self.high - self.low) / self.n_bins

    def centers(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """Bin centres, shape (n_bins,)."""
        edges = torch.linspace(self.low, self.high, self.n_bins + 1,
                               device=device, dtype=dtype)
        return 0.5 * (edges[:-1] + edges[1:])


def soft_labels(values: torch.Tensor, spec: BinSpec) -> torch.Tensor:
    """Gaussian-smoothed target distribution over bins, shape (..., n_bins).

    Each row is a normalized Gaussian of width ``spec.sigma`` evaluated at the
    bin centres. Rows sum to 1 even when the value sits at the range edge,
    because the truncated mass is renormalized rather than discarded.
    """
    values = torch.as_tensor(values)
    centers = spec.centers(device=values.device, dtype=values.dtype)
    clamped = values.clamp(spec.low, spec.high).unsqueeze(-1)
    logits = -0.5 * ((centers - clamped) / spec.sigma) ** 2
    return torch.softmax(logits, dim=-1)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy against a soft target (equals KL up to a constant)."""
    if logits.shape != target.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs "
                         f"target {tuple(target.shape)}")
    return -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def decode_expectation(logits: torch.Tensor, spec: BinSpec) -> torch.Tensor:
    """Continuous prediction: the distribution's mean over bin centres."""
    probabilities = torch.softmax(logits, dim=-1)
    centers = spec.centers(device=logits.device, dtype=probabilities.dtype)
    return (probabilities * centers).sum(dim=-1)


def decode_std(logits: torch.Tensor, spec: BinSpec) -> torch.Tensor:
    """Predicted standard deviation -- the head's own uncertainty estimate."""
    probabilities = torch.softmax(logits, dim=-1)
    centers = spec.centers(device=logits.device, dtype=probabilities.dtype)
    mean = (probabilities * centers).sum(dim=-1, keepdim=True)
    variance = (probabilities * (centers - mean) ** 2).sum(dim=-1)
    return variance.clamp_min(0).sqrt()


def decode_mode_refined(logits: torch.Tensor, spec: BinSpec) -> torch.Tensor:
    """Parabolic interpolation around the arg-max bin.

    The expectation is the right decoder for a unimodal posterior, but it is
    pulled toward the range centre when the distribution is skewed or
    multimodal. This decoder instead fits a parabola through the peak bin and
    its two neighbours, which localizes a dominant mode without being dragged
    by a secondary one.
    """
    log_probabilities = torch.log_softmax(logits, dim=-1)
    peak = log_probabilities.argmax(dim=-1)
    n_bins = spec.n_bins
    left = (peak - 1).clamp(0, n_bins - 1)
    right = (peak + 1).clamp(0, n_bins - 1)
    gather = lambda index: log_probabilities.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    y_left, y_peak, y_right = gather(left), gather(peak), gather(right)
    denominator = y_left - 2 * y_peak + y_right
    offset = torch.where(
        denominator.abs() > 1e-12,
        0.5 * (y_left - y_right) / denominator,
        torch.zeros_like(denominator),
    ).clamp(-0.5, 0.5)
    # Interior bins only: at the edges the parabola is not constrained.
    offset = torch.where((peak == 0) | (peak == n_bins - 1),
                         torch.zeros_like(offset), offset)
    centers = spec.centers(device=logits.device, dtype=log_probabilities.dtype)
    return centers[peak] + offset * spec.width
