#!/usr/bin/env python
"""Materialize the active bacterial validation and strict PXD010613 benchmarks.

This is the v3 charge-filtered counterpart to
``create_bacterial_paper_benchmarks.py``.  The benchmark keeps the existing
output layout but deliberately does not recreate the weaker run-held-out view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.create_bacterial_paper_benchmarks import _training_sequences, _write_split


def parse_args() -> argparse.Namespace:
    root = "/path/to/data/bacteria_PXD010000__PXD010613/annotated_regenerated_v3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-lance", default=f"{root}/train.lance")
    parser.add_argument("--validation-lance", default=f"{root}/val.lance")
    parser.add_argument("--test-lance", default=f"{root}/test.lance")
    parser.add_argument("--output-root", default="/path/to/data/probing_datasets/bacterial_paper_benchmarks")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--validation-max-retrieval-groups", type=int, default=20000)
    parser.add_argument("--validation-max-pair-groups", type=int, default=20000)
    parser.add_argument("--validation-max-pairs-per-protocol", type=int, default=100000)
    parser.add_argument("--test-max-retrieval-groups", type=int, default=None)
    parser.add_argument("--test-max-pair-groups", type=int, default=None)
    parser.add_argument("--test-max-pairs-per-protocol", type=int, default=None)
    parser.add_argument("--max-spectra-per-group", type=int, default=10)
    parser.add_argument("--max-positive-pairs-per-group", type=int, default=20)
    parser.add_argument("--precursor-tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    train_path = Path(args.train_lance)
    validation_path = Path(args.validation_lance)
    test_path = Path(args.test_lance)
    for path in (train_path, validation_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)
    training_sequences = _training_sequences(train_path, validation_path)
    root = Path(args.output_root)
    results = {
        "validation": _write_split(
            dataset_path=validation_path,
            output_root=root / "validation",
            split_name="pxd010000_validation_v3_charge_filtered",
            seed=args.seed,
            excluded_sequences=set(),
            exclusion_description="none",
            max_retrieval_groups=args.validation_max_retrieval_groups,
            max_pair_groups=args.validation_max_pair_groups,
            max_pairs_per_protocol=args.validation_max_pairs_per_protocol,
            max_spectra_per_group=args.max_spectra_per_group,
            max_positive_pairs_per_group=args.max_positive_pairs_per_group,
            tolerance_ppm=args.precursor_tolerance_ppm,
            overwrite=args.overwrite,
        ),
        "test_peptide_disjoint": _write_split(
            dataset_path=test_path,
            output_root=root / "test_peptide_disjoint",
            split_name="pxd010613_run_and_peptide_disjoint_v3_charge_filtered",
            seed=args.seed,
            excluded_sequences=training_sequences,
            exclusion_description="exact modified sequences present in PXD010000 v3 train.lance or val.lance",
            max_retrieval_groups=args.test_max_retrieval_groups,
            max_pair_groups=args.test_max_pair_groups,
            max_pairs_per_protocol=args.test_max_pairs_per_protocol,
            max_spectra_per_group=args.max_spectra_per_group,
            max_positive_pairs_per_group=args.max_positive_pairs_per_group,
            tolerance_ppm=args.precursor_tolerance_ppm,
            overwrite=args.overwrite,
        ),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
