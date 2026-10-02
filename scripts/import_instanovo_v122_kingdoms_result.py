#!/usr/bin/env python3
"""Import validated full-charge Kingdoms InstaNovo v1.2.2 metrics into the ledger."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_ROWS = 4_926_232
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


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=Path(
            "/path/to/results/denovo_eval/instanovo_v1_2_2/"
            "kingdoms_species_cap100k_full_charge_sharded/metrics.json"
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
        "scoring_mode": "dion_pa11_canonical_mass_matcher",
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
    indexed = {key(row): position for position, row in enumerate(rows)}
    notes = (
        "Stock InstaNovo v1.2.2 code / v1.2.0 supervised checkpoint prediction-mode result on the "
        "full-charge deterministic 100k-per-species Kingdoms test, decoded with genuine greedy search "
        "(num_beams=1, use_knapsack=false) from the raw byte-preserved MGF. All 4,926,232 inputs emitted "
        "a prediction. Peptide matching uses the dIon PA1.1 canonical tokenizer and in-repository mass matcher."
    )
    for report_key, metric_name, priority in METRICS:
        value = report.get(report_key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Report does not contain numeric {report_key}.")
        row = {
            "experiment_id": "denovo_kingdoms_species_cap100k_instanovo_v122_20260918",
            "task": "denovo",
            "cohort": "kingdoms_run_disjoint_v1_species_cap100k",
            "split": "test",
            "corpus": "kingdoms_run_disjoint_cap100k",
            "model_id": "instanovo_v1_2_2",
            "conditioning": "not_applicable",
            "representation": "released_denovo",
            "metric_name": metric_name,
            "value": repr(float(value)),
            "status": "complete",
            "priority": priority,
            "selection_role": "selected",
            "higher_is_better": "true",
            "report_path": str(args.report),
            "notes": notes,
        }
        existing = indexed.get(key(row))
        if existing is None:
            indexed[key(row)] = len(rows)
            rows.append(row)
        else:
            rows[existing].update(row)

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported validated InstaNovo v1.2.2 Kingdoms metrics from {args.report}")


if __name__ == "__main__":
    main()
