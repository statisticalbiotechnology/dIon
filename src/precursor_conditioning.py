"""Explicit precursor conditioning modes for trained mass/charge encoders."""

from __future__ import annotations

import torch


VALID_PRECURSOR_CONDITIONING = {"conditioned", "null"}


def validate_precursor_conditioning(mode: str) -> str:
    if mode not in VALID_PRECURSOR_CONDITIONING:
        raise ValueError(
            "precursor_conditioning must be one of "
            f"{sorted(VALID_PRECURSOR_CONDITIONING)}, got {mode!r}."
        )
    return mode


def condition_precursor_inputs(
    mass: torch.Tensor | None,
    charge: torch.Tensor | None,
    mode: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return conditioned inputs or the dIon dual objective's explicit null state.

    The null state is deliberately mass=0 plus charge=0. For the dIon dual
    checkpoint, charge index 0 is the learned student-only null embedding.
    """
    validate_precursor_conditioning(mode)
    if mode == "conditioned":
        return mass, charge
    if mass is None or charge is None:
        raise ValueError(
            "Null precursor conditioning requires both mass and charge encoder inputs."
        )
    return torch.zeros_like(mass), torch.zeros_like(charge)
