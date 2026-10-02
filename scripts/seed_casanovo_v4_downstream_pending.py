"""Idempotently add planned Casanovo v4 frozen-head validation cells to the paper ledger."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROWS = (
    ("sqa", "casanovo19pxd_balanced_charge1_10", "casanovo19pxd", "sqa_frozen", "classification/roc_auc", "primary", "development", "true"),
    ("chimericity", "PXD024584_HYE_V1", "PXD024584_HYE", "binary_frozen", "classification/roc_auc", "primary", "development", "true"),
    ("oxidized_met", "PXD010613_oxidized_met_V1", "PXD010613", "binary_frozen", "classification/roc_auc", "primary", "development", "true"),
    ("oxidized_met", "PXD010613_oxidized_met_V1", "PXD010613", "binary_frozen", "classification/matched_backbone_roc_auc", "secondary", "development", "true"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_ordinal_frozen", "regression/mae", "primary", "development", "false"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_ordinal_frozen", "regression/r2", "secondary", "development", "true"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_ordinal_frozen", "regression/spearman", "secondary", "development", "true"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_ordinal_frozen", "regression/pearson", "diagnostic", "development", "true"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_ordinal_frozen", "regression/delta_t95", "sensitivity", "development", "false"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_regression_frozen", "regression/mae", "primary", "development", "false"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_regression_frozen", "regression/r2", "secondary", "development", "true"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_regression_frozen", "regression/spearman", "secondary", "development", "true"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_regression_frozen", "regression/pearson", "diagnostic", "development", "true"),
    ("retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies", "linear_regression_frozen", "regression/delta_t95", "sensitivity", "development", "false"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()
    with args.ledger.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"Empty ledger: {args.ledger}")
        existing = list(reader)
    experiment_id = "casanovo_v4_frozen_downstream_validation_planned_20260911"
    keys = {
        (row["experiment_id"], row["task"], row["cohort"], row["representation"], row["metric_name"])
        for row in existing
    }
    added = 0
    for task, cohort, corpus, representation, metric_name, priority, selection_role, higher_is_better in ROWS:
        key = (experiment_id, task, cohort, representation, metric_name)
        if key in keys:
            continue
        existing.append({
            "experiment_id": experiment_id,
            "task": task,
            "cohort": cohort,
            "split": "validation",
            "corpus": corpus,
            "model_id": "casanovo_v4_peak_mean",
            "conditioning": "not_applicable",
            "representation": representation,
            "metric_name": metric_name,
            "value": "",
            "status": "pending",
            "priority": priority,
            "selection_role": selection_role,
            "higher_is_better": higher_is_better,
            "report_path": "",
            "notes": "Planned released Casanovo v4.0.0 peak-mean frozen representation baseline; vectors use the external model preprocessing and complete split-indexed cache.",
        })
        added += 1
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
    print(f"Added {added} pending Casanovo v4 downstream rows to {args.ledger}")


if __name__ == "__main__":
    main()
