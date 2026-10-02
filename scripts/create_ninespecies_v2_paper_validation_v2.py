#!/usr/bin/env python
"""Materialize compact NineSpecies validation data disjoint from locked test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq

from scripts.create_ninespecies_v2_paper_test import create_benchmarks
from src.embed_eval.sequence_normalization import (
    BACKBONE_NORMALIZATION_VERSION,
    normalize_peptide_backbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/path/to/data/9_species_V2/lance_species")
    parser.add_argument("--paper-test-root", default="/path/to/data/probing_datasets/ninespecies_v2_paper_test")
    parser.add_argument("--output-root", default="/path/to/data/probing_datasets/ninespecies_v2_paper_validation")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-peptide-groups", type=int, default=500)
    parser.add_argument("--max-peptide-ion-groups", type=int, default=500)
    parser.add_argument("--max-pairs-per-species", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    test_root = Path(args.paper_test_root)
    output_root = Path(args.output_root)
    sequences = set(pq.read_table(test_root / "retrieval.parquet", columns=["peptide_id"])["peptide_id"].to_pylist())
    sequences.update(pq.read_table(test_root / "pair_discrimination" / "spectra.parquet", columns=["modified_sequence"])["modified_sequence"].to_pylist())
    excluded_backbones = sorted(normalize_peptide_backbone(sequence) for sequence in sequences if sequence)
    output_root.mkdir(parents=True, exist_ok=True)
    exclusion_path = output_root / "paper_test_excluded_backbones.json"
    exclusion_path.write_text(json.dumps({"source": str(test_root), "normalization_version": BACKBONE_NORMALIZATION_VERSION, "excluded_backbone_count": len(excluded_backbones), "excluded_backbones": excluded_backbones}, indent=2, sort_keys=True) + "\n")
    config = argparse.Namespace(source_root=args.source_root, source_dataset_version="9_species_V2", excluded_backbones_path=str(exclusion_path), output_root=str(output_root), seed=args.seed, max_peptide_groups=args.max_peptide_groups, max_peptide_ion_groups=args.max_peptide_ion_groups, max_spectra_per_group=10, max_positive_pairs_per_group=20, max_pairs_per_species=args.max_pairs_per_species, precursor_tolerance_ppm=10.0, overwrite=args.overwrite)
    print(json.dumps(create_benchmarks(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
