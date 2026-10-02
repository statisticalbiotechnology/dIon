"""Import completed frozen-random auxiliary runs from their local W&B summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = [
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
]
TASK_META = {
    "chimericity": ("PXD024584_HYE_V1", "PXD024584_HYE", [("test_auc", "classification/roc_auc", True, "primary")]),
    "oxidized_met": ("PXD010613_oxidized_met_V1", "PXD010613", [
        ("test_auc", "classification/roc_auc", True, "primary"),
        ("test_matched_backbone_auc", "classification/matched_backbone_roc_auc", True, "secondary"),
    ]),
}

def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in ("task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/path/to/results/auxiliary_random_frozen"))
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()
    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != FIELDS:
        raise ValueError(f"Unexpected ledger schema: {args.ledger}")
    indexed = {row_key(row): i for i, row in enumerate(rows)}
    imported = 0
    for task, (cohort, corpus, metrics) in TASK_META.items():
        summaries = sorted(args.root.glob(f"{task}_random_conditioned_frozen_*/logs*/**/wandb-summary.json"))
        if len(summaries) != 1:
            raise ValueError(f"Expected exactly one completed {task} summary under {args.root}; found {len(summaries)}")
        summary_path = summaries[0]
        summary = json.loads(summary_path.read_text())
        for source, metric, higher, priority in metrics:
            value = summary.get(source)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Missing finite {source} in {summary_path}")
            row = {
                "experiment_id": "auxiliary_random_frozen_validation_selected",
                "task": task, "cohort": cohort, "split": "test", "corpus": corpus,
                "model_id": "random_conditioned_frozen", "conditioning": "conditioned",
                "representation": "binary_frozen", "metric_name": metric,
                "value": repr(float(value)), "status": "complete", "priority": priority,
                "selection_role": "selected", "higher_is_better": str(higher).lower(),
                "report_path": str(summary_path),
                "notes": "Frozen random-transformer control; checkpoint selected by the 100-batch validation monitor and tested on the complete held-out split.",
            }
            key = row_key(row)
            if key in indexed:
                rows[indexed[key]].update(row)
            else:
                indexed[key] = len(rows)
                rows.append(row)
            imported += 1
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {imported} frozen-random auxiliary metrics without printing values.")

if __name__ == "__main__":
    main()
