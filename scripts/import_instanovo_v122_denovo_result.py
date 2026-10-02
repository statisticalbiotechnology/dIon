#!/usr/bin/env python3
"""Import the validated raw-MGF InstaNovo v1.2.2 MSKB result into the paper ledger."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_ROWS = 196_979
FIELDS = (
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
)
METRICS = (
    ("peptide_precision_at_full_coverage", "denovo/peptide_precision", "primary"),
    ("aa_precision_at_full_coverage", "denovo/aa_precision", "secondary"),
    ("aa_recall_at_full_coverage", "denovo/aa_recall", "secondary"),
    ("peptide_precision_coverage_auc", "denovo/precision_coverage_auc", "diagnostic"),
)
COHORT = "mskb_final_charge_lt5_for_casanovo_v_gt_5_0"
CORPUS = "mskb_final_charge_lt5"


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def upsert(rows: list[dict[str, str]], row: dict[str, str]) -> None:
    row_key = key(row)
    for existing in rows:
        if key(existing) == row_key:
            existing.update(row)
            return
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=Path(
            "/path/to/results/denovo_eval/instanovo_v1_2_2/"
            "mskb_final_charge_lt5_raw_mgf/metrics.json"
        ),
    )
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    expected = {
        "full_denominator_count": EXPECTED_ROWS,
        "mgf_spectrum_count": EXPECTED_ROWS,
        "predicted_spectrum_count": EXPECTED_ROWS,
        "total_no_prediction_count": 0,
        "prediction_source": "instanovo_v1_2_2_csv",
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"Unexpected {field}: {report.get(field)!r}, expected {value!r}")
    if report.get("missing_mgf_indices") != []:
        raise ValueError("InstaNovo report has unmatched MGF rows.")

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")


    instanovo_note = (
        "Stock InstaNovo v1.2.2 code / v1.2.0 supervised checkpoint prediction-mode result on the "
        "fixed 196,979-spectrum MSKB-final charge-<5 held-out subset, decoded with genuine greedy "
        "search (num_beams=1, use_knapsack=false) from the raw byte-preserved MGF. All inputs emitted "
        "a prediction and full-coverage metrics therefore have zero missing-prediction errors. "
        "Leakage concern: overlap with the published InstaNovo training data has not yet been ruled out."
    )
    for report_key, metric_name, priority in METRICS:
        value = report.get(report_key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Report does not contain numeric {report_key}.")
        upsert(rows, {
            "experiment_id": "denovo_mskb_final_charge_lt5_instanovo_v122_20260917",
            "task": "denovo", "cohort": COHORT, "split": "test", "corpus": CORPUS,
            "model_id": "instanovo_v1_2_2", "conditioning": "not_applicable",
            "representation": "released_denovo", "metric_name": metric_name,
            "value": repr(float(value)), "status": "complete", "priority": priority,
            "selection_role": "leakage_concern", "higher_is_better": "true",
            "report_path": str(args.report), "notes": instanovo_note,
        })

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported validated InstaNovo v1.2.2 MSKB metrics from {args.report}")


if __name__ == "__main__":
    main()
