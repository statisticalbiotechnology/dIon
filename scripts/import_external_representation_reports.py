"""Import one completed external retrieval/pair report pair into the paper ledger.

The import key is the canonical representation cell identity.  Re-running this
script replaces its four rows, including the previously seeded pending rows.
"""
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
METRICS = {
    "retrieval/broad_map": ("primary", "development", "map"),
    "retrieval/same_charge_10ppm_map": ("secondary", "diagnostic", "mass_controlled_map"),
    "pair/same_charge_10ppm/roc_auc": ("primary", "development", "roc_auc"),
    "pair/same_charge_10ppm/average_precision": ("primary", "development", "average_precision"),
}


def _metric_source(report: dict, kind: str) -> dict:
    if "metrics_by_distance" in report:
        cosine = report["metrics_by_distance"]["cosine"]
        return cosine["macro"] if kind == "retrieval" else cosine["pair_sets"]["same_charge_10ppm"]["macro"]
    return report["macro"] if kind == "retrieval" else report["pair_sets"]["same_charge_10ppm"]["macro"]


def _key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--pairs-report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--conditioning", default="not_applicable")
    parser.add_argument("--representation", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()

    reports = (("retrieval", args.retrieval_report), ("pairs", args.pairs_report))
    for _, path in reports:
        if not path.is_file():
            raise FileNotFoundError(path)

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {_key(row): index for index, row in enumerate(rows)}
    imported = 0
    for kind, path in reports:
        source = _metric_source(json.loads(path.read_text()), kind)
        prefix = "retrieval/" if kind == "retrieval" else "pair/same_charge_10ppm/"
        for metric_name, (priority, selection_role, source_key) in METRICS.items():
            if not metric_name.startswith(prefix):
                continue
            value = source.get(source_key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                raise ValueError(f"{path} does not contain cosine {source_key!r}.")
            row = {
                "experiment_id": args.experiment_id,
                "task": "representation",
                "cohort": args.cohort,
                "split": args.split,
                "corpus": args.corpus,
                "model_id": args.model_id,
                "conditioning": args.conditioning,
                "representation": args.representation,
                "metric_name": metric_name,
                "value": repr(float(value)),
                "status": "complete",
                "priority": priority,
                "selection_role": selection_role,
                "higher_is_better": "true",
                "report_path": str(path),
                "notes": args.notes,
            }
            row_key = _key(row)
            if row_key in by_key:
                rows[by_key[row_key]].update(row)
            else:
                by_key[row_key] = len(rows)
                rows.append(row)
            imported += 1

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {imported} external representation metrics into {args.ledger}")


if __name__ == "__main__":
    main()
