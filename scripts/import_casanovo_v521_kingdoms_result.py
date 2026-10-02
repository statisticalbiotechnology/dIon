#!/usr/bin/env python3
"""Import validated Casanovo v5.2.1 Kingdoms peptide metrics into the ledger."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_FULL_DENOMINATOR = 4_926_232
EXPECTED_SUPPORTED_INPUTS = 4_912_728
FIELDS = (
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
)
METRICS = (
    ("peptide_precision_at_full_coverage", "denovo/peptide_precision", "primary"),
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
            "/path/to/results/denovo_eval/casanovo_v5_2_1/"
            "kingdoms_species_cap100k_full_denominator_sharded/metrics.json"
        ),
    )
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    required = {
        "full_denominator_count": EXPECTED_FULL_DENOMINATOR,
        "mgf_spectrum_count": EXPECTED_SUPPORTED_INPUTS,
        "scoring_mode": "dion_pa11_canonical_mass_matcher",
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"Unexpected {field}: {report.get(field)!r}, expected {expected!r}")
    if int(report.get("predicted_spectrum_count", 0)) < 1:
        raise ValueError("Casanovo report has no emitted predictions.")
    if int(report.get("total_no_prediction_count", -1)) != (
        EXPECTED_FULL_DENOMINATOR - int(report["predicted_spectrum_count"])
    ):
        raise ValueError("Casanovo no-prediction accounting does not match the full denominator.")
    if report.get("aa_precision_at_full_coverage") is not None or report.get("aa_recall_at_full_coverage") is not None:
        raise ValueError("This importer expects the full-denominator report to defer AA metrics.")

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    indexed = {key(row): position for position, row in enumerate(rows)}
    notes = (
        "Stock Casanovo v5.2.1 code / v5.2.0 Orbitrap checkpoint prediction-mode result on the "
        "raw, byte-preserved, charge-1--4 Kingdoms input MGF. Peptide matching uses the dIon "
        "PA1.1 canonical tokenizer and in-repository mass matcher. The full 4,926,232-spectrum "
        f"denominator includes {report['unsupported_no_prediction_count']:,} unsupported charge >4 rows "
        f"and {report['missing_supported_prediction_count']:,} supported rows without an emitted PSM, "
        "all as zero-confidence peptide errors. AA metrics remain pending because their full-denominator "
        "calculation requires labels for the unsupported rows outside Casanovo's charge-filtered input."
    )
    for report_key, metric_name, priority in METRICS:
        value = report.get(report_key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Report does not contain numeric {report_key}.")
        row = {
            "experiment_id": "denovo_kingdoms_species_cap100k_casanovo_v521_20260918",
            "task": "denovo",
            "cohort": "kingdoms_run_disjoint_v1_species_cap100k",
            "split": "test",
            "corpus": "kingdoms_run_disjoint_cap100k",
            "model_id": "casanovo_v5_2_1",
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
    print(f"Imported Casanovo v5.2.1 Kingdoms peptide metrics from {args.report}")


if __name__ == "__main__":
    main()
