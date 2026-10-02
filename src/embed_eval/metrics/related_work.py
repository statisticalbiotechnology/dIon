from __future__ import annotations

import math

import torch

from src.embed_eval.utils import prepare_embeddings


def _singular_values(
    Z: torch.Tensor,
    *,
    center: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    with torch.no_grad():
        X = prepare_embeddings(Z, center=center, dtype=dtype)
        if min(X.shape) == 0:
            return torch.empty(0, dtype=dtype, device=X.device)
        return torch.linalg.svdvals(X)


def _covariance_eigenvalues(
    Z: torch.Tensor,
    *,
    center: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    with torch.no_grad():
        X = prepare_embeddings(Z, center=center, dtype=dtype)
        n = X.shape[0]
        if n <= 1 or min(X.shape) == 0:
            return torch.empty(0, dtype=dtype, device=X.device)
        s = torch.linalg.svdvals(X)
        vals = (s * s) / max(n - 1, 1)
        return torch.sort(vals, descending=True).values


def alpha_req(
    Z: torch.Tensor,
    *,
    center: bool = True,
    min_rank: int = 1,
    max_rank: int | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Fit covariance eigenvalues to lambda_i proportional to i^-alpha."""
    with torch.no_grad():
        vals = _covariance_eigenvalues(Z, center=center)
        vals = vals[vals > eps]
        if max_rank is not None:
            vals = vals[:max_rank]
        if min_rank > 1:
            vals = vals[min_rank - 1 :]
            offset = min_rank - 1
        else:
            offset = 0
        if vals.numel() < 2:
            return {
                "alpha": 0.0,
                "intercept": float(torch.log(vals[0]).item()) if vals.numel() else 0.0,
                "r2": 0.0,
                "n_points": float(vals.numel()),
            }

        ranks = torch.arange(offset + 1, offset + 1 + vals.numel(), device=vals.device, dtype=vals.dtype)
        x = torch.log(ranks)
        y = torch.log(vals)
        X = torch.stack([torch.ones_like(x), x], dim=1)
        coef = torch.linalg.lstsq(X, y).solution
        pred = X @ coef
        ss_res = torch.sum((y - pred) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        r2 = 1.0 - float((ss_res / ss_tot).item()) if float(ss_tot.item()) > 0 else 0.0
        return {
            "alpha": float((-coef[1]).item()),
            "intercept": float(coef[0].item()),
            "r2": r2,
            "n_points": float(vals.numel()),
        }


def alpha_req_score(Z: torch.Tensor, **kwargs) -> float:
    return alpha_req(Z, **kwargs)["alpha"]


def rankme_entropy(
    Z: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> float:
    """Entropy of normalized singular values."""
    with torch.no_grad():
        s = _singular_values(Z, center=center)
        total = s.sum()
        if float(total.item()) <= eps:
            return 0.0
        p = s / total.clamp_min(eps)
        p = p[p > eps]
        entropy = -torch.sum(p * torch.log(p))
        return float(entropy.item())


def rankme_effective_rank(Z: torch.Tensor, *, center: bool = True, eps: float = 1e-12) -> float:
    return float(math.exp(rankme_entropy(Z, center=center, eps=eps)))


def rankme_normalized(Z: torch.Tensor, *, center: bool = True, eps: float = 1e-12) -> float:
    n, d = Z.shape
    max_rank = min(n - 1 if center else n, d)
    max_rank = max(1, max_rank)
    return rankme_effective_rank(Z, center=center, eps=eps) / float(max_rank)


def nesum(
    Z: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> float:
    """Normalized eigenvalue sum: sum_i lambda_i / lambda_0, with 0/0 = 0."""
    with torch.no_grad():
        vals = _covariance_eigenvalues(Z, center=center)
        if vals.numel() == 0:
            return 0.0
        top = vals[0]
        if float(top.item()) <= eps:
            return 0.0
        return float((vals.sum() / top).item())
