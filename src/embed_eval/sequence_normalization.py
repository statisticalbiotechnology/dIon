"""Canonical peptide-backbone handling for cross-dataset split exclusions."""

from __future__ import annotations

import json
from pathlib import Path


BACKBONE_NORMALIZATION_VERSION = "uppercase_amino_acids_v1"


def normalize_peptide_backbone(sequence: str) -> str:
    """Remove flanking residues and modification notation, retaining amino acids.

    This deliberately treats differently encoded modifications of the same
    peptide backbone as overlapping for conservative split construction.
    """
    return "".join(char for char in sequence if "A" <= char <= "Z")


def load_backbone_exclusion(path: str | Path | None) -> set[str]:
    """Load a versioned JSON backbone-exclusion manifest, if one is supplied."""
    if path is None:
        return set()
    source = Path(path)
    payload = json.loads(source.read_text())
    if payload.get("normalization_version") != BACKBONE_NORMALIZATION_VERSION:
        raise ValueError(
            f"Unsupported backbone normalization in {source}: "
            f"{payload.get('normalization_version')!r}"
        )
    values = payload.get("excluded_backbones")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{source} does not contain a string excluded_backbones list.")
    return set(values)
