#!/usr/bin/env python3
"""Import the test-only dIon oracle-precursor DIA decoding pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path("/path/to/results/denovo_eval/denovo_dia_oracle_precursor/dia_fragment_v1_pilot")
LEDGER = Path("results/paper/metrics_long.csv")
FIELDS = ("experiment_id", "task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name", "value", "status", "priority", "selection_role", "higher_is_better", "report_path", "notes")
KEY_FIELDS = ("task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name")
EXPECTED_HASHES = {
    "predictions.csv": "48816982e2789290315e849302ce171989e002f80ec06bb19272e3916e21b8e6",
    "metrics.json": "67abe7e854cf7d74181d4aebf951b53839673ce6430058ad3dc9678878fec182",
    "precision_coverage.csv": "af83ce4be6a3e812f73a83ad6367215ed84a35ec86b30ae0d2f8f6922e9580b1",
}
METRICS = (
    ("peptide_precision_100pct_coverage", "denovo/peptide_precision", "primary"),
    ("aa_precision", "denovo/aa_precision", "secondary"),
    ("aa_recall", "denovo/aa_recall", "secondary"),
    ("precision_coverage_auc", "denovo/precision_coverage_auc", "diagnostic"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()

    manifest = json.loads((args.root / "manifest.json").read_text())
    if manifest.get("package") != "dion_oracle_precursor_dia_prediction":
        raise ValueError("Unexpected DIA pilot package type.")
    for name, expected in EXPECTED_HASHES.items():
        if sha256(args.root / name) != expected:
            raise ValueError(f"Hash mismatch for {name}")
        manifest_entry = manifest["precision_coverage" if name == "precision_coverage.csv" else name.removesuffix(".json").removesuffix(".csv")]
        if manifest_entry["sha256"] != expected:
            raise ValueError(f"Manifest hash mismatch for {name}")

    metrics_path = args.root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    expected_metrics = {
        "evaluation": "oracle_precursor_dia_mixture_decoding",
        "matcher": "src.casanovo_eval.aa_match",
        "prediction_rows": 17_781,
        "unique_oracle_target_queries": 17_781,
        "no_prediction_rows": 0,
        "peptide_precision_100pct_coverage": 0.10567459647938811,
        "precision_coverage_auc": 0.3277542728859211,
        "aa_precision": 0.2247482592883642,
        "aa_recall": 0.22723157369597755,
    }
    for field, expected in expected_metrics.items():
        if metrics.get(field) != expected:
            raise ValueError(f"Unexpected {field}: {metrics.get(field)!r} != {expected!r}")

    with (args.root / "predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    indices = [int(row["canonical_index"]) for row in predictions]
    queries = {
        (row["scan_id"], row["true_sequence"], row["precursor_mz"], row["precursor_charge"])
        for row in predictions
    }
    widths = [
        float(row["isolation_high"]) - float(row["isolation_low"])
        for row in predictions
    ]
    if (len(predictions) != 17_781 or sorted(indices) != list(range(17_781))
            or len({row["scan_id"] for row in predictions}) != 9_973
            or len(queries) != 17_781 or min(widths) != 26.0
            or any(row["no_prediction"].lower() != "false" for row in predictions)
            or {row["split"] for row in predictions} != {"train", "val", "test"}
            or any(not math.isfinite(float(row["peptide_confidence"])) for row in predictions)):
        raise ValueError("Unexpected DIA prediction roster or accounting.")

    with (args.root / "precision_coverage.csv").open(newline="") as handle:
        curve = list(csv.DictReader(handle))
    if (curve[0]["rank"] != "0" or curve[-1]["rank"] != "17781"
            or float(curve[-1]["coverage"]) != 1.0
            or float(curve[-1]["peptide_precision"]) != expected_metrics["peptide_precision_100pct_coverage"]):
        raise ValueError("Unexpected precision-coverage curve endpoints.")

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    cohort = "dia_oracle_precursor_dia_fragment_v1_pilot"
    rows = [row for row in rows if not (row["task"] == "denovo" and row["cohort"] == cohort)]
    indexed = {tuple(row[field] for field in KEY_FIELDS) for row in rows}
    notes = (
        "Test-only selected-model pilot over all available dia_fragment_v1 target queries: "
        "17,781 unique (scan, target peptide, precursor) rows across 9,973 real wide-window DIA "
        "MS2 scans (minimum isolation width 26 m/z). Each row supplies an oracle DIA-NN precursor "
        "m/z and charge to decode that component peptide; repeated scans under different queries are "
        "intentional. This is not MS1 feature detection, component selection, or complete physical "
        "DIA deconvolution. Frozen Gram-refined dIon encoder with target-swapped decoder, selected "
        "stable epoch 8, 200 peaks, precursor conditioning, and one autoregressive beam. Metrics were "
        "recomputed from the complete prediction table with the PA1.1 tokenizer and dIon's "
        "canonical Casanovo matcher. Source rows retain train/val/test provenance labels, but this "
        "union was deliberately evaluated once as a test-only pilot; no development rows exist."
    )
    for source_name, metric_name, priority in METRICS:
        row = {
            "experiment_id": "denovo_dia_oracle_precursor_dia_fragment_v1_pilot_20260925",
            "task": "denovo", "cohort": cohort, "split": "test",
            "corpus": "dia_fragment_v1_all_target_queries",
            "model_id": "dion_gram_refined_target_swapped_epoch8",
            "conditioning": "oracle_precursor_conditioned",
            "representation": "frozen_gram_refined_encoder",
            "metric_name": metric_name, "value": repr(float(metrics[source_name])),
            "status": "complete", "priority": priority, "selection_role": "selected",
            "higher_is_better": "true", "report_path": str(metrics_path), "notes": notes,
        }
        key = tuple(row[field] for field in KEY_FIELDS)
        if key in indexed:
            raise ValueError(f"Duplicate DIA pilot ledger key: {metric_name}")
        rows.append(row)
        indexed.add(key)

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Imported 4 test-only dIon oracle-precursor DIA pilot metrics.")


if __name__ == "__main__":
    main()
