from __future__ import annotations

import torch

from src.embed_eval.utils import numerical_rank_from_singular_values, prepare_embeddings


def coherence(
    Z: torch.Tensor,
    *,
    center: bool = True,
    rank: int | None = None,
    rank_tol: float | None = None,
) -> dict[str, float]:
    """Practical SVD coherence scores for left and right singular-vector bases."""
    with torch.no_grad():
        X = prepare_embeddings(Z, center=center, dtype=torch.float64)
        n, d = X.shape
        if min(n, d) == 0:
            return {"left": 0.0, "right": 0.0, "max": 0.0, "rank": 0.0}
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        r = rank if rank is not None else numerical_rank_from_singular_values(S, shape=(n, d), rank_tol=rank_tol)
        r = int(max(0, min(r, S.numel())))
        if r == 0:
            return {"left": 0.0, "right": 0.0, "max": 0.0, "rank": 0.0}
        U_r = U[:, :r]
        V_r = Vh[:r, :].mT
        left = (n / r) * torch.sum(U_r * U_r, dim=1).max()
        right = (d / r) * torch.sum(V_r * V_r, dim=1).max()
        left_f = float(left.item())
        right_f = float(right.item())
        return {"left": left_f, "right": right_f, "max": max(left_f, right_f), "rank": float(r)}


def left_coherence(Z: torch.Tensor, **kwargs) -> float:
    return coherence(Z, **kwargs)["left"]


def right_coherence(Z: torch.Tensor, **kwargs) -> float:
    return coherence(Z, **kwargs)["right"]


def mu0_incoherence(Z: torch.Tensor, **kwargs) -> dict[str, float]:
    """Paper-faithful mu0 incoherence/coherence parameter."""
    return coherence(Z, **kwargs)


def incoherence(Z: torch.Tensor, **kwargs) -> dict[str, float]:
    """Alias for the paper's mu0 incoherence parameter."""
    return mu0_incoherence(Z, **kwargs)


def inverse_mu0_spread(Z: torch.Tensor, **kwargs) -> dict[str, float]:
    """Reciprocal mu0 spread score; larger means less aligned/spiky."""
    scores = coherence(Z, **kwargs)
    return {
        "left": 1.0 / scores["left"] if scores["left"] > 0 else 0.0,
        "right": 1.0 / scores["right"] if scores["right"] > 0 else 0.0,
        "min": 1.0 / scores["max"] if scores["max"] > 0 else 0.0,
        "rank": scores["rank"],
    }
