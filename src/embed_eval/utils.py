from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_2d_tensor(Z: torch.Tensor) -> torch.Tensor:
    if not isinstance(Z, torch.Tensor):
        raise TypeError("Z must be a torch.Tensor")
    if Z.ndim != 2:
        raise ValueError(f"Z must have shape [n_samples, embedding_dim], got {tuple(Z.shape)}")
    return Z


def prepare_embeddings(
    Z: torch.Tensor,
    *,
    center: bool = True,
    normalize: bool = False,
    dtype: torch.dtype | None = torch.float64,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Detach and optionally center/L2-normalize an embedding matrix."""
    Z = ensure_2d_tensor(Z).detach()
    if dtype is not None and Z.dtype != dtype:
        Z = Z.to(dtype=dtype)
    if center:
        Z = Z - Z.mean(dim=0, keepdim=True)
    if normalize:
        Z = torch.nn.functional.normalize(Z, p=2, dim=1, eps=eps)
    return Z


def numerical_rank_from_singular_values(
    singular_values: torch.Tensor,
    *,
    shape: tuple[int, int] | None = None,
    rank_tol: float | None = None,
    eps: float | None = None,
) -> int:
    if singular_values.numel() == 0:
        return 0
    s = singular_values.detach()
    top = float(s.max().item())
    if top <= 0:
        return 0
    if rank_tol is None:
        if eps is None:
            eps = torch.finfo(s.dtype).eps if s.dtype.is_floating_point else 1e-12
        scale = max(shape) if shape is not None else s.numel()
        rank_tol = scale * eps * top
    return int((s > rank_tol).sum().item())


def to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be converted to float")
        return float(value.detach().cpu().item())
    return float(value)


def setup_matplotlib_cache(output_dir: str | os.PathLike[str] = "outputs") -> None:
    output = Path(output_dir)
    mpl_dir = output / ".mplconfig"
    cache_dir = output / ".cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
