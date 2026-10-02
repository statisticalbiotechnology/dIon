from __future__ import annotations

import math

import torch

from src.embed_eval.metrics.high_dimensional_probability import self_cluster
from src.embed_eval.metrics.linear_classifier_perspective import coherence
from src.embed_eval.metrics.numerical_linear_algebra import (
    pseudo_condition_number,
    stable_rank,
)
from src.embed_eval.metrics.related_work import (
    alpha_req,
    nesum,
    rankme_effective_rank,
    rankme_entropy,
    rankme_normalized,
)
from src.embed_eval.utils import numerical_rank_from_singular_values, prepare_embeddings


def evaluate_embedding_suite(
    Z: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    center: bool = True,
    device: str | torch.device | None = None,
    max_samples_self_cluster: int | None = 10000,
) -> dict[str, float]:
    """Compute a flat suite of unsupervised embedding-quality metrics."""
    del y
    with torch.no_grad():
        X = Z.detach()
        if device is not None:
            X = X.to(device)
        X64 = prepare_embeddings(X, center=center, dtype=torch.float64)
        n, d = X64.shape
        norms = torch.linalg.vector_norm(X.detach().to(dtype=torch.float64), dim=1) if n else torch.empty(0)
        s = torch.linalg.svdvals(X64) if min(n, d) > 0 else torch.empty(0, dtype=torch.float64, device=X64.device)
        rank = numerical_rank_from_singular_values(s, shape=(n, d))
        bottom = float(s[rank - 1].item()) if rank > 0 else 0.0

        alpha = alpha_req(X, center=center)
        coh = coherence(X, center=center)

        results = {
            "n": float(n),
            "d": float(d),
            "mean_norm": float(norms.mean().item()) if norms.numel() else 0.0,
            "std_norm": float(norms.std(unbiased=False).item()) if norms.numel() else 0.0,
            "singular_top": float(s[0].item()) if s.numel() else 0.0,
            "singular_bottom_nonzero": bottom,
            "numerical_rank": float(rank),
            "related_work.alpha_req": alpha["alpha"],
            "related_work.alpha_req_r2": alpha["r2"],
            "related_work.rankme_entropy": rankme_entropy(X, center=center),
            "related_work.rankme_effective_rank": rankme_effective_rank(X, center=center),
            "related_work.rankme_normalized": rankme_normalized(X, center=center),
            "related_work.nesum": nesum(X, center=center),
            "linear_classifier_perspective.coherence_left": coh["left"],
            "linear_classifier_perspective.coherence_right": coh["right"],
            "linear_classifier_perspective.coherence_max": coh["max"],
            "numerical_linear_algebra.stable_rank": stable_rank(X, center=center),
            "numerical_linear_algebra.pseudo_condition_number": pseudo_condition_number(X, center=center),
            "high_dimensional_probability.self_cluster": self_cluster(
                X,
                max_samples=max_samples_self_cluster,
                seed=0,
            ),
        }
        for key, value in list(results.items()):
            if isinstance(value, float) and math.isnan(value):
                results[key] = 0.0
        return results

