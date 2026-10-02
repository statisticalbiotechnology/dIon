"""Seed expected pending locked-test rows so dashboard completion is truthful."""
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
REP_METRICS = (
    ("retrieval/broad_map", "primary"),
    ("retrieval/same_charge_10ppm_map", "secondary"),
    ("pair/same_charge_10ppm/roc_auc", "primary"),
    ("pair/same_charge_10ppm/average_precision", "primary"),
)
DOWNSTREAM_METRICS = {
    "sqa": (("classification/roc_auc", "primary", True),),
    "chimericity": (("classification/roc_auc", "primary", True),),
    "oxidized_met": (
        ("classification/roc_auc", "primary", True),
        ("classification/matched_backbone_roc_auc", "secondary", True),
    ),
    "retention_time": (
        ("regression/mae", "primary", False),
        ("regression/r2", "secondary", True),
        ("regression/spearman", "secondary", True),
        ("regression/pearson", "diagnostic", True),
        ("regression/delta_t95", "sensitivity", False),
    ),
}
TASK_META = {
    "sqa": ("casanovo19pxd_balanced_charge1_10", "casanovo19pxd"),
    "chimericity": ("PXD024584_HYE_V1", "PXD024584_HYE"),
    "oxidized_met": ("PXD010613_oxidized_met_V1", "PXD010613"),
    "retention_time": ("run_aligned_retention_time_V1", "run_aligned_multispecies"),
}


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def pending(**kwargs: str) -> dict[str, str]:
    return {
        "value": "", "status": "pending", "selection_role": "locked_test",
        "report_path": "", "higher_is_better": "true", **kwargs,
    }


def representation_rows(selection: Path, cohort: str) -> list[dict[str, str]]:
    data = json.loads(selection.read_text())
    ledger_cohort = "full_charge" if cohort == "primary" else cohort
    rows: list[dict[str, str]] = []
    for item in data["representation"]["models"]:
        model_id = item["model_id"]
        representation = "metric_head" if model_id.startswith("metric_") else "backbone"
        for conditioning in item["conditionings"]:
            for corpus in item["corpora"]:
                for metric_name, priority in REP_METRICS:
                    rows.append(pending(
                        experiment_id=f"locked_test_{ledger_cohort}_{model_id}", task="representation",
                        cohort=ledger_cohort, split="test", corpus=corpus, model_id=model_id,
                        conditioning=conditioning, representation=representation,
                        metric_name=metric_name, priority=priority,
                        notes="Frozen validation-selected held-out evaluation; queued or awaiting import.",
                    ))
    return rows


def downstream_identity(label: str, task: str) -> tuple[str, str, str]:
    model_id = "hybrid_300_last"
    conditioning = "null" if "_null_" in label else "conditioned"
    if task == "retention_time":
        return model_id, conditioning, "linear_ordinal_frozen" if "ordinal" in label else "linear_regression_frozen"
    if task == "sqa":
        return model_id, conditioning, "sqa_frozen" if label.endswith("_frozen") else "sqa_finetuned"
    return model_id, conditioning, "binary_frozen" if label.endswith("_frozen") else "binary_finetuned"


def downstream_rows(selection: Path) -> list[dict[str, str]]:
    data = json.loads(selection.read_text())
    rows: list[dict[str, str]] = []
    for item in data["downstream"]:
        task = item["task"]
        model_id, conditioning, representation = downstream_identity(item["model_label"], task)
        cohort, corpus = TASK_META[task]
        for metric_name, priority, higher in DOWNSTREAM_METRICS[task]:
            rows.append(pending(
                experiment_id="locked_test_hybrid300last_downstream", task=task,
                cohort=cohort, split="test", corpus=corpus, model_id=model_id,
                conditioning=conditioning, representation=representation,
                metric_name=metric_name, priority=priority,
                higher_is_better=str(higher).lower(),
                notes="Hybrid-300-last validation-selected held-out evaluation; queued or awaiting import.",
            ))
    return rows



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()
    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != FIELDS:
        raise ValueError(f"Unexpected ledger schema: {args.ledger}")
    indexed = {key(row): i for i, row in enumerate(rows)}
    expected = [
        *representation_rows(Path("configs/evaluation/locked_test_hybrid300last_representation.json"), "primary"),
        *representation_rows(Path("configs/evaluation/locked_test_hybrid300last_representation.json"), "charge2to4"),
        *representation_rows(Path("configs/evaluation/locked_test_hybrid300last_metric_learning.json"), "charge2to4"),
        *downstream_rows(Path("configs/evaluation/locked_test_hybrid300last_downstream.json")),
    ]
    added = 0
    for row in expected:
        if key(row) not in indexed:
            indexed[key(row)] = len(rows)
            rows.append(row)
            added += 1
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"Seeded {added} expected locked-test metric rows.")


if __name__ == "__main__":
    main()
