from __future__ import annotations

import os
from typing import Any, Tuple

import torch


def infer_rank_world_size(*, expected_world_size: int | None = None) -> Tuple[int, int]:
    """Infer (rank, world_size) for single-process or DDP runs."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
            if expected_world_size not in (None, 1) and world_size != int(expected_world_size):
                raise ValueError(
                    f"Distributed world_size={world_size} does not match expected_world_size={expected_world_size}."
                )
            return rank, world_size
    except Exception:
        pass

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if expected_world_size not in (None, 1) and world_size != int(expected_world_size):
        raise ValueError(
            f"Inferred WORLD_SIZE={world_size} does not match expected_world_size={expected_world_size}. "
            "If you intended to run distributed, launch with srun/torchrun so RANK/WORLD_SIZE are set."
        )
    return rank, world_size


def make_lance_safe_loader(*, dataset, num_workers: int, use_safe: bool = True, **kwargs: Any):
    """Create a DataLoader that is safe for Lance's multiprocessing constraints."""
    if num_workers <= 0 or not use_safe:
        kwargs.pop("multiprocessing_context", None)
        kwargs.pop("persistent_workers", None)
        return torch.utils.data.DataLoader(dataset, num_workers=0, **kwargs)

    from lance.torch.data import get_safe_loader

    # Lance defaults batch_size to 32 even when batch_sampler is supplied.
    # PyTorch requires its batch_size sentinel to remain 1 in that case and
    # then delegates batching entirely to batch_sampler.
    if kwargs.get("batch_sampler") is not None and "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    return get_safe_loader(dataset, num_workers=int(num_workers), **kwargs)
