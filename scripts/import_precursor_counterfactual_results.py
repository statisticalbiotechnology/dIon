"""Import fixed-mixture precursor-counterfactual reports into the paper ledger."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
]
REPORTS = {
    "hybrid_300_last_9spec_val_100pctB.json": ("backbone_cosine", "cosine"),
    "hybrid_300_last_9spec_val_100pctB_euclidean.json": ("backbone_euclidean", "euclidean"),
    "hybrid_300_last_9spec_val_100pctB_dino_head_js.json": ("dino_head_jensen_shannon", "jensen_shannon"),
}
METRICS = {
    "source_accuracy_anchor_condition": ("counterfactual/a_condition_source_accuracy", "primary", "development"),
    "source_accuracy_distractor_condition": ("counterfactual/b_condition_source_accuracy", "primary", "development"),
    "paired_source_accuracy": ("counterfactual/both_conditions_source_accuracy", "primary", "development"),
    "mean_source_accuracy": ("counterfactual/mean_source_accuracy", "secondary", "diagnostic"),
    "anchor_condition_margin": ("counterfactual/a_condition_positive_negative_margin", "secondary", "diagnostic"),
    "distractor_condition_margin": ("counterfactual/b_condition_positive_negative_margin", "secondary", "diagnostic"),
    "mean_source_margin": ("counterfactual/mean_positive_negative_margin", "secondary", "diagnostic"),
}
NULL_LOCAL_REPORT = "hybrid_300_last_9spec_val_null_local_consistency.json"
NULL_LOCAL_METRICS = {
    "local_view_1_source_accuracy": ("null_local/view_1_source_accuracy", "primary", "development"),
    "local_view_2_source_accuracy": ("null_local/view_2_source_accuracy", "primary", "development"),
    "paired_source_accuracy": ("null_local/both_views_source_accuracy", "primary", "development"),
    "mean_source_accuracy": ("null_local/mean_source_accuracy", "secondary", "diagnostic"),
    "local_view_1_margin": ("null_local/view_1_positive_negative_margin", "secondary", "diagnostic"),
    "local_view_2_margin": ("null_local/view_2_positive_negative_margin", "secondary", "diagnostic"),
    "mean_source_margin": ("null_local/mean_positive_negative_margin", "secondary", "diagnostic"),
}


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("results/representation/precursor_counterfactual"),
    )
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {row_key(row): index for index, row in enumerate(rows)}
    imported = 0
    for filename, (representation, distance_metric) in REPORTS.items():
        report_path = args.reports_root / filename
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        if report.get("mixing", {}).get("distractor_peak_fraction") != 1.0:
            raise ValueError(f"{report_path} is not the locked 100%-B intervention.")
        aggregate = report["metrics"]["macro"]
        common = {
            "experiment_id": "representation_precursor_counterfactual_100pctB_20260911",
            "task": "representation",
            "cohort": "precursor_counterfactual_100pctB",
            "split": "validation",
            "corpus": "ninespecies_v2",
            "model_id": "hybrid_300_last",
            "conditioning": "conditioned",
            "representation": representation,
            "status": "complete",
            "higher_is_better": "true",
            "report_path": str(report_path),
            "notes": (
                "Fixed 100%-B A+B precursor-swap intervention; 4,488 deterministic "
                "different-peptide pairs, species-macro aggregate. "
                f"Distance={distance_metric}."
            ),
        }
        for source_key, (metric_name, priority, selection_role) in METRICS.items():
            row = dict(
                common,
                metric_name=metric_name,
                value=repr(float(aggregate[source_key])),
                priority=priority,
                selection_role=selection_role,
            )
            key = row_key(row)
            if key in by_key:
                rows[by_key[key]].update(row)
            else:
                by_key[key] = len(rows)
                rows.append(row)
            imported += 1

    report_path = args.reports_root / NULL_LOCAL_REPORT
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("protocol") != "null_local_view_consistency":
            raise ValueError(f"Unexpected null-local protocol in {report_path}.")
        aggregate = report["metrics"]["macro"]
        common = {
            "experiment_id": "representation_null_local_consistency_20260911",
            "task": "representation",
            "cohort": "null_local_consistency",
            "split": "validation",
            "corpus": "ninespecies_v2",
            "model_id": "hybrid_300_last",
            "conditioning": "null_local",
            "representation": "backbone_cosine",
            "status": "complete",
            "higher_is_better": "true",
            "report_path": str(report_path),
            "notes": (
                "Two configured independent 60% random-batched peak subsets with "
                "the learned null precursor, evaluated against full conditioned A "
                "and deterministic matched different-peptide B controls; species-macro aggregate."
            ),
        }
        missing = set(NULL_LOCAL_METRICS) - set(aggregate)
        if missing:
            raise ValueError(
                f"{report_path} has the obsolete null-local metric schema; rerun the evaluator. "
                f"Missing={sorted(missing)}"
            )
        for source_key, (metric_name, priority, selection_role) in NULL_LOCAL_METRICS.items():
            row = dict(
                common,
                metric_name=metric_name,
                value=repr(float(aggregate[source_key])),
                priority=priority,
                selection_role=selection_role,
            )
            key = row_key(row)
            if key in by_key:
                rows[by_key[key]].update(row)
            else:
                by_key[key] = len(rows)
                rows.append(row)
            imported += 1

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {imported} counterfactual metrics: {args.ledger}")


if __name__ == "__main__":
    main()
