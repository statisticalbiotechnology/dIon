#!/usr/bin/env python3
"""Validate and package oracle-precursor DIA de novo prediction CSV output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import casanovo_eval as evaluate
from src.data.unified_tokenizer import PeptideTokenizer
from scripts.validate_denovo_mztab import write_curve


REQUIRED_COLUMNS = {
    "canonical_index", "scan_id", "split", "source_row_index",
    "precursor_mz", "precursor_charge", "isolation_low", "isolation_high",
    "true_sequence", "predicted_sequence", "peptide_confidence", "no_prediction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, default=Path("configs/tokenizers/pa11.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    return parser.parse_args()


def peptide_tokens(value: str, tokenizer: PeptideTokenizer) -> list[str]:
    return [] if not value else tokenizer.preprocess_sequence(value)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite package: {args.output_dir}")
    for path in (args.predictions_csv, args.checkpoint, args.config, args.dataset_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(args.tokenizer_manifest)
    residues = tokenizer.residues
    seen_indices, seen_queries, ranked_matches = set(), set(), []
    aa_correct = aa_true = aa_predicted = peptide_correct = no_predictions = 0

    with args.predictions_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Prediction CSV lacks required columns: {sorted(missing)!r}")
        for row_number, row in enumerate(reader, start=2):
            index = int(row["canonical_index"])
            if index in seen_indices:
                raise ValueError(f"Duplicate canonical index {index} at CSV row {row_number}")
            seen_indices.add(index)
            query = (row["scan_id"], row["precursor_mz"], row["precursor_charge"], row["true_sequence"])
            if query in seen_queries:
                raise ValueError(f"Duplicate oracle target query {query!r} at CSV row {row_number}")
            seen_queries.add(query)
            if not all(row[name] for name in ("scan_id", "split", "source_row_index", "precursor_mz", "precursor_charge")):
                raise ValueError(f"Missing DIA query provenance at CSV row {row_number}")

            no_prediction = row["no_prediction"].lower() == "true"
            predicted = [] if no_prediction else peptide_tokens(row["predicted_sequence"], tokenizer)
            truth = peptide_tokens(row["true_sequence"], tokenizer)
            unknown = (set(predicted) | set(truth)) - residues.keys()
            if unknown:
                raise ValueError(f"Unknown residue token(s) at CSV row {row_number}: {sorted(unknown)!r}")
            if no_prediction:
                if row["peptide_confidence"]:
                    raise ValueError(f"No-prediction row has confidence at CSV row {row_number}")
                score = float("-inf")
                no_predictions += 1
            else:
                score = float(row["peptide_confidence"])
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite confidence at CSV row {row_number}")

            aa_matches, peptide_match = evaluate.aa_match(truth, predicted, residues)
            aa_correct += int(aa_matches.sum())
            aa_true += len(truth)
            aa_predicted += len(predicted)
            peptide_correct += int(peptide_match)
            ranked_matches.append((score, bool(peptide_match)))

    expected_indices = set(range(args.expected_rows))
    if seen_indices != expected_indices:
        missing = sorted(expected_indices - seen_indices)
        extras = sorted(seen_indices - expected_indices)
        raise ValueError(f"Canonical coverage mismatch: missing={missing[:10]}, extras={extras[:10]}")

    args.output_dir.mkdir(parents=True)
    prediction_copy = args.output_dir / "predictions.csv"
    shutil.copy2(args.predictions_csv, prediction_copy)
    curve_csv = args.output_dir / "precision_coverage.csv"
    summary = {
        "schema_version": 1,
        "evaluation": "oracle_precursor_dia_mixture_decoding",
        "prediction_rows": len(seen_indices),
        "unique_oracle_target_queries": len(seen_queries),
        "no_prediction_rows": no_predictions,
        "peptide_precision_100pct_coverage": peptide_correct / len(seen_indices),
        "aa_precision": aa_correct / aa_predicted if aa_predicted else 0.0,
        "aa_recall": aa_correct / aa_true if aa_true else 0.0,
        "precision_coverage_auc": write_curve(curve_csv, ranked_matches),
        "matcher": "src.casanovo_eval.aa_match",
        "tokenizer_manifest": str(args.tokenizer_manifest.resolve()),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "package": "dion_oracle_precursor_dia_prediction",
        "predictions": {"path": prediction_copy.name, "sha256": sha256(prediction_copy)},
        "precision_coverage": {"path": curve_csv.name, "sha256": sha256(curve_csv)},
        "metrics": {"path": metrics_path.name, "sha256": sha256(metrics_path)},
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "config": {"path": str(args.config), "sha256": sha256(args.config)},
        "dataset_manifest": {"path": str(args.dataset_manifest), "sha256": sha256(args.dataset_manifest)},
        "source_predictions_csv": {"path": str(args.predictions_csv), "sha256": sha256(args.predictions_csv)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metrics": summary, "manifest": manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
