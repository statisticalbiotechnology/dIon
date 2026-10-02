from __future__ import annotations

import math

import torch

from src.embed_eval.utils import numerical_rank_from_singular_values, prepare_embeddings


def stable_rank(
    Z: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> float:
    """Stable rank ||M||_F^2 / ||M||_2^2."""
    with torch.no_grad():
        X = prepare_embeddings(Z, center=center, dtype=torch.float64)
        if min(X.shape) == 0:
            return 0.0
        s = torch.linalg.svdvals(X)
        if s.numel() == 0 or float(s[0].item()) <= eps:
            return 0.0
        return float((torch.sum(s * s) / (s[0] * s[0])).item())


def pseudo_condition_number(
    Z: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-12,
    rank_tol: float | None = None,
    clamp: bool = False,
) -> float:
    """Largest singular value divided by smallest nonzero singular value.

    If the matrix is effectively rank deficient relative to min(n, d), this
    returns inf unless clamp=True.
    """
    with torch.no_grad():
        X = prepare_embeddings(Z, center=center, dtype=torch.float64)
        if min(X.shape) == 0:
            return math.inf
        s = torch.linalg.svdvals(X)
        if s.numel() == 0 or float(s[0].item()) <= eps:
            return math.inf
        rank = numerical_rank_from_singular_values(s, shape=tuple(X.shape), rank_tol=rank_tol, eps=eps)
        full_rank = min(X.shape)
        if rank < full_rank and not clamp:
            return math.inf
        if rank == 0:
            return math.inf
        smallest = s[rank - 1]
        if clamp:
            smallest = smallest.clamp_min(eps)
        elif float(smallest.item()) <= eps:
            return math.inf
        return float((s[0] / smallest).item())

