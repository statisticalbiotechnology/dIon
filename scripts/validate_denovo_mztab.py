#!/usr/bin/env python3
"""Validate dIon de novo mzTab predictions and score Casanovo-style metrics.

The peptide match definition is exactly the one used by
``DeNovoTransformerWrapper.test_step``: ``src.casanovo_eval.aa_match`` with
the configured tokenizer's residue masses.  A null prediction is retained as
an incorrect spectrum at 100% coverage and is ranked after numeric scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src import casanovo_eval as evaluate
from src.data.unified_tokenizer import PeptideTokenizer


PSM_COLUMNS = (
    "sequence",
    "PSM_ID",
    "accession",
    "unique",
    "database",
    "database_version",
    "search_engine",
    "search_engine_score[1]",
    "modifications",
    "retention_time",
    "charge",
    "exp_mass_to_charge",
    "calc_mass_to_charge",
    "spectra_ref",
    "pre",
    "post",
    "start",
    "end",
    "opt_ms_run[1]_aa_scores",
    "opt_ms_run[1]_ground_truth_sequence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mztab", type=Path, help="Prediction mzTab written by MztabOutputCallback.")
    parser.add_argument(
        "--tokenizer-manifest",
        type=Path,
        default=Path("configs/tokenizers/pa11.json"),
        help="Numeric mass-delta tokenizer manifest used during prediction.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Require exactly this many PSM rows (for example 196979 for charge-<5 MSKB-final).",
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=None,
        help="Optional CSV of confidence-ranked precision/coverage points, grouped by equal score.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for the same summary printed to stdout.",
    )
    return parser.parse_args()


def peptide_tokens(value: str) -> list[str]:
    return [] if value in {"", "null"} else value.split(",")


def write_curve(path: Path, ranked_matches: list[tuple[float, bool]]) -> float:
    ranked_matches.sort(key=lambda item: item[0], reverse=True)
    total = len(ranked_matches)
    correct = 0
    previous_coverage = 0.0
    previous_precision = 1.0
    auc = 0.0
    rank = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank", "coverage", "peptide_precision", "correct", "score"))
        writer.writeheader()
        writer.writerow({"rank": 0, "coverage": 0.0, "peptide_precision": 1.0, "correct": 0, "score": ""})
        while rank < total:
            score = ranked_matches[rank][0]
            group_end = rank
            while group_end < total and ranked_matches[group_end][0] == score:
                correct += int(ranked_matches[group_end][1])
                group_end += 1
            coverage = group_end / total
            precision = correct / group_end
            auc += (coverage - previous_coverage) * (precision + previous_precision) / 2
            writer.writerow(
                {
                    "rank": group_end,
                    "coverage": coverage,
                    "peptide_precision": precision,
                    "correct": correct,
                    "score": "" if math.isinf(score) and score < 0 else score,
                }
            )
            rank = group_end
            previous_coverage = coverage
            previous_precision = precision
    return auc


def validate_and_score(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(args.tokenizer_manifest)
    residues = tokenizer.residues
    psm_count = 0
    metadata_count = 0
    header: tuple[str, ...] | None = None
    seen_spectra: set[str] = set()
    null_scores = 0
    aa_correct = 0
    aa_true = 0
    aa_predicted = 0
    peptide_correct = 0
    ranked_matches: list[tuple[float, bool]] | None = [] if args.curve_csv else None

    with args.mztab.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row:
                continue
            if row[0] == "MTD":
                metadata_count += 1
                continue
            if row[0] == "PSH":
                if header is not None:
                    raise ValueError(f"Duplicate PSH header at line {line_number}.")
                header = tuple(row[1:])
                if header != PSM_COLUMNS:
                    raise ValueError(f"Unexpected PSM header at line {line_number}: {header!r}")
                continue
            if row[0] != "PSM":
                raise ValueError(f"Unexpected mzTab record {row[0]!r} at line {line_number}.")
            if header is None:
                raise ValueError(f"PSM record before PSH header at line {line_number}.")
            if len(row) != len(PSM_COLUMNS) + 1:
                raise ValueError(f"Wrong PSM field count at line {line_number}: {len(row)}.")

            values = dict(zip(header, row[1:], strict=True))
            expected_id = str(psm_count + 1)
            if values["PSM_ID"] != expected_id:
                raise ValueError(
                    f"Non-sequential PSM_ID at line {line_number}: "
                    f"expected {expected_id}, got {values['PSM_ID']!r}."
                )
            spectrum_ref = values["spectra_ref"]
            if spectrum_ref in seen_spectra:
                raise ValueError(f"Duplicate spectra_ref at line {line_number}: {spectrum_ref!r}.")
            seen_spectra.add(spectrum_ref)

            predicted = peptide_tokens(values["sequence"])
            truth = peptide_tokens(values["opt_ms_run[1]_ground_truth_sequence"])
            unknown = (set(predicted) | set(truth)) - residues.keys()
            if unknown:
                raise ValueError(f"Unknown residue token(s) at line {line_number}: {sorted(unknown)!r}.")
            aa_matches, peptide_match = evaluate.aa_match(truth, predicted, residues)
            aa_correct += int(aa_matches.sum())
            aa_true += len(truth)
            aa_predicted += len(predicted)
            peptide_correct += int(peptide_match)

            score_value = values["search_engine_score[1]"]
            if score_value == "null":
                null_scores += 1
                score = float("-inf")
            else:
                score = float(score_value)
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite confidence at line {line_number}: {score_value!r}.")
            if ranked_matches is not None:
                ranked_matches.append((score, bool(peptide_match)))
            psm_count += 1

    if header is None:
        raise ValueError("mzTab has no PSH header.")
    if metadata_count == 0:
        raise ValueError("mzTab has no MTD metadata rows.")
    if args.expected_rows is not None and psm_count != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} PSM rows, found {psm_count}.")

    summary: dict[str, Any] = {
        "mztab": str(args.mztab),
        "psm_rows": psm_count,
        "metadata_rows": metadata_count,
        "unique_spectra": len(seen_spectra),
        "null_confidence_rows": null_scores,
        "peptide_precision_100pct_coverage": peptide_correct / psm_count if psm_count else 0.0,
        "aa_precision": aa_correct / aa_predicted if aa_predicted else 0.0,
        "aa_recall": aa_correct / aa_true if aa_true else 0.0,
    }
    if ranked_matches is not None:
        summary["precision_coverage_auc"] = write_curve(args.curve_csv, ranked_matches)
        summary["curve_csv"] = str(args.curve_csv)
    return summary


def main() -> None:
    args = parse_args()
    summary = validate_and_score(args)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
