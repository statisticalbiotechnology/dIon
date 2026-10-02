#!/usr/bin/env python3
"""Import exact prediction-mode Kingdoms metrics for selected V5 200-peak models."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


FIELDS = (
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
)
LEDGER = Path("results/paper/metrics_long.csv")
ROOT = Path("/path/to/results/denovo_eval/denovo_kingdoms_species_cap100k")
PREDICTION_ROOT = ROOT / "200_peaks_prediction_mode"
CURVE_ROOT = ROOT / "precision_coverage"
DENOMINATOR = 4_926_232
RUNS = (
    ("v5_hybrid300_finetuned", "dion_dnlv1_metrics.json", 5),
    ("v5_scratch_encoderld", "scratch_dnlv1_metrics.json", 36),
)


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def main() -> None:
    with LEDGER.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    indexed = {key(row): position for position, row in enumerate(rows)}

    for model_id, report_name, expected_no_prediction in RUNS:
        report_path = CURVE_ROOT / report_name
        report = json.loads(report_path.read_text())
        manifest = json.loads((PREDICTION_ROOT / model_id / "run_manifest.json").read_text())
        if report.get("canonical_denominator") != DENOMINATOR:
            raise ValueError(f"Unexpected denominator in {report_path}")
        if report.get("scoring_mode") != "dion_pa11_canonical_mass_matcher":
            raise ValueError(f"Unexpected scoring mode in {report_path}")
        if report.get("no_prediction_count") != expected_no_prediction:
            raise ValueError(f"Unexpected no-prediction count in {report_path}")
        if report.get("emitted_predictions") + report.get("no_prediction_count") != DENOMINATOR:
            raise ValueError(f"Incomplete prediction accounting in {report_path}")
        matches = report.get("peptide_matches")
        precision = report.get("peptide_precision_at_full_coverage")
        curve_auc = report.get("peptide_precision_coverage_auc")
        if not isinstance(matches, int) or not isinstance(precision, (int, float)):
            raise ValueError(f"Invalid exact peptide metrics in {report_path}")
        if not math.isclose(float(precision), matches / DENOMINATOR, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Peptide precision is not an exact match count in {report_path}")
        if not isinstance(curve_auc, (int, float)) or not math.isfinite(float(curve_auc)):
            raise ValueError(f"Invalid precision-coverage AUC in {report_path}")
        if manifest.get("canonical_denominator") != DENOMINATOR or manifest.get("model") != model_id:
            raise ValueError(f"Prediction manifest mismatch for {model_id}")

        notes = (
            "Exact prediction-mode re-score of the complete canonical 4,926,232-spectrum Kingdoms "
            "test table using dIon's PA1.1 tokenizer and canonical mass matcher. Every canonical "
            f"index occurs once; {expected_no_prediction} decoder abstentions remain explicit incorrect "
            "rows at full coverage. This value supersedes the padded online distributed aggregate for "
            "peptide precision; the historical DDP-padding qualification does not apply to this metric."
        )
        for metric_name, value, priority in (
            ("denovo/peptide_precision", float(precision), "primary"),
            ("denovo/precision_coverage_auc", float(curve_auc), "diagnostic"),
        ):
            row = {
                "experiment_id": "denovo_kingdoms_species_cap100k_200peak_prediction_20260923",
                "task": "denovo", "cohort": "kingdoms_run_disjoint_v1_species_cap100k",
                "split": "test", "corpus": "kingdoms_run_disjoint_cap100k",
                "model_id": model_id, "conditioning": "conditioned",
                "representation": "full_finetune", "metric_name": metric_name,
                "value": repr(value), "status": "complete", "priority": priority,
                "selection_role": "selected", "higher_is_better": "true",
                "report_path": str(report_path), "notes": notes,
            }
            row_key = key(row)
            if row_key in indexed:
                rows[indexed[row_key]].update(row)
            else:
                indexed[row_key] = len(rows)
                rows.append(row)

    with LEDGER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Imported exact 200-peak Kingdoms prediction metrics for dIon and scratch.")


if __name__ == "__main__":
    main()
