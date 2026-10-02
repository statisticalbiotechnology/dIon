"""Import selected SQA validation AUROC values from Weights & Biases.

The importer intentionally reads only ``val_auc`` history. It never requests
or writes test metrics. Rows are upserted into the paper-results long ledger,
so rerunning it is idempotent and does not affect training or evaluation jobs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import wandb


LEDGER_FIELDS = [
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
]
RUNS = (
    ("tuufl3px", "conditioned", "sqa_frozen"),
    ("fkill1ac", "null", "sqa_frozen"),
    ("isj6oasf", "conditioned", "sqa_finetuned"),
    ("0qfj2gs9", "null", "sqa_finetuned"),
)


@dataclass(frozen=True)
class BestValidation:
    value: float
    epoch: int | None
    step: int | None


def _best_validation_auc(run) -> BestValidation:
    candidates = []
    for row in run.scan_history(keys=["_step", "epoch", "val_auc"]):
        value = row.get("val_auc")
        if isinstance(value, (int, float)):
            candidates.append((float(value), row.get("epoch"), row.get("_step")))
    if not candidates:
        raise ValueError(f"{run.path} has no numeric val_auc history.")
    value, epoch, step = max(candidates, key=lambda candidate: candidate[0])
    return BestValidation(value, None if epoch is None else int(epoch), None if step is None else int(step))


def _read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LEDGER_FIELDS:
            raise ValueError(f"Unexpected ledger schema in {path}: {reader.fieldnames}")
        return list(reader)


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        row[field]
        for field in (
            "task", "cohort", "split", "corpus", "model_id", "conditioning",
            "representation", "metric_name",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument("--entity-project", default="user/ms2-sqa")
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    rows = _read_ledger(args.ledger)
    # Repair the first importer version's cohort spelling before upserting.
    for row in rows:
        if (
            row.get("experiment_id") == "sqa_hybrid300e219_validation_20260908"
            and row.get("cohort") == "casanovo_19pxd_balanced_charge1_10"
        ):
            row["cohort"] = "casanovo19pxd_balanced_charge1_10"
    deduplicated = {}
    for row in rows:
        deduplicated[_row_key(row)] = row
    rows = list(deduplicated.values())
    indexed = {_row_key(row): index for index, row in enumerate(rows)}
    imported = []
    for run_id, conditioning, representation in RUNS:
        run = api.run(f"{args.entity_project}/{run_id}")
        best = _best_validation_auc(run)
        row = {
            "experiment_id": "sqa_hybrid300e219_validation_20260908",
            "task": "sqa",
            "cohort": "casanovo19pxd_balanced_charge1_10",
            "split": "validation",
            "corpus": "casanovo19pxd",
            "model_id": "hybrid_300_epoch219",
            "conditioning": conditioning,
            "representation": representation,
            "metric_name": "classification/roc_auc",
            "value": repr(best.value),
            "status": "complete",
            "priority": "primary",
            "selection_role": "development",
            "higher_is_better": "true",
            "report_path": f"https://wandb.ai/{args.entity_project}/runs/{run.id}",
            "notes": (
                f"W&B run {run.id}; maximum logged validation AUROC "
                f"at epoch {best.epoch}, step {best.step}."
            ),
        }
        key = _row_key(row)
        if key in indexed:
            rows[indexed[key]] = row
        else:
            indexed[key] = len(rows)
            rows.append(row)
        imported.append((run.id, conditioning, representation, best))

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for run_id, conditioning, representation, best in imported:
        print(
            f"{run_id} {conditioning} {representation}: "
            f"val_auc={best.value:.6f} epoch={best.epoch} step={best.step}"
        )
    print(f"Upserted {len(imported)} SQA validation rows into {args.ledger}")


if __name__ == "__main__":
    main()
