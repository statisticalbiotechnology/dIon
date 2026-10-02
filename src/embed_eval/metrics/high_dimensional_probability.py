from __future__ import annotations

import torch

from src.embed_eval.utils import prepare_embeddings


def self_cluster(
    Z: torch.Tensor,
    *,
    normalize: bool = True,
    max_samples: int | None = 10000,
    seed: int | None = None,
    eps: float = 1e-12,
) -> float:
    """SelfCluster score for L2-normalized embeddings.

    Uses the squared Frobenius norm of W W^T, which makes random unit vectors
    center near 0 and fully collapsed unit vectors approach 1.
    """
    with torch.no_grad():
        W = prepare_embeddings(Z, center=False, normalize=normalize, dtype=torch.float64, eps=eps)
        n, d = W.shape
        if n <= 1 or d <= 1:
            return 0.0
        if max_samples is not None and n > max_samples:
            generator = torch.Generator(device=W.device)
            if seed is not None:
                generator.manual_seed(seed)
            idx = torch.randperm(n, generator=generator, device=W.device)[:max_samples]
            W = W[idx]
            n = W.shape[0]
        gram = W @ W.mT
        fro_sq = torch.sum(gram * gram)
        numerator = d * fro_sq - n * (d + n - 1)
        denominator = (d - 1) * (n - 1) * n
        return float((numerator / denominator).item())

