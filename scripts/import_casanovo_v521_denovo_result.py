#!/usr/bin/env python3
"""Import one validated stock Casanovo v5.2.1 de novo report into the ledger."""
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


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path,
        default=Path("/path/to/results/denovo_eval/casanovo_v5_2_1/mskb_final_charge_lt5_raw_mgf/metrics.json"),
    )
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    required = {
        "full_denominator_count": EXPECTED_ROWS,
        "mgf_spectrum_count": EXPECTED_ROWS,
        "scoring_mode": "casanovo_v5_checkpoint_tokenizer",
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"Unexpected {field}: {report.get(field)!r}, expected {expected!r}")
    if int(report.get("predicted_spectrum_count", 0)) < 1:
        raise ValueError("Casanovo report has no emitted predictions.")

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    index = {key(row): position for position, row in enumerate(rows)}
    notes = (
        "Stock Casanovo v5.2.1 code / v5.2.0 Orbitrap checkpoint prediction-mode result from the raw, byte-preserved MGF on the fixed "
        "196,979-spectrum MSKB-final charge-<5 held-out subset. The decoder omitted "
        f"{report['total_no_prediction_count']} inputs, retained as zero-confidence errors in full-coverage metrics."
    )
    for report_key, metric_name, priority in METRICS:
        value = report.get(report_key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Report does not contain numeric {report_key}.")
        row = {
            "experiment_id": "denovo_mskb_final_charge_lt5_casanovo_v521_raw_mgf_20260917",
            "task": "denovo",
            "cohort": "mskb_final_charge_lt5_for_casanovo_v_gt_5_0",
            "split": "test",
            "corpus": "mskb_final_charge_lt5",
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
        existing = index.get(key(row))
        if existing is None:
            index[key(row)] = len(rows)
            rows.append(row)
        else:
            rows[existing].update(row)

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported Casanovo v5.2.1 MSKB metrics from {args.report}")


if __name__ == "__main__":
    main()
