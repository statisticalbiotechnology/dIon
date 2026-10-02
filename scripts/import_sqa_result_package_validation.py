"""Import validation-AUROC selections from a canonical SQA result package.

Only rows with ``split_id=validation`` and
``metric_name=val_auc_selected_checkpoint`` are imported. Test metrics remain
untouched. This makes a historical completed package visible beside active W&B
validation runs in the paper ledger.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LEDGER_FIELDS = [
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
]


def _key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        row[field]
        for field in (
            "task", "cohort", "split", "corpus", "model_id", "conditioning",
            "representation", "metric_name",
        )
    )


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LEDGER_FIELDS:
            raise ValueError(f"Unexpected ledger schema in {path}: {reader.fieldnames}")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/sqa/casanovo_19pxd_auc_selected__20260908T160857Z/metrics.csv"),
    )
    args = parser.parse_args()

    with args.source.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    selected = [
        row for row in source_rows
        if row["split_id"] == "validation"
        and row["metric_name"] == "val_auc_selected_checkpoint"
    ]
    if not selected:
        raise ValueError(f"No selected validation AUROC rows found in {args.source}")

    ledger = _read(args.ledger)
    indexed = {_key(row): index for index, row in enumerate(ledger)}
    source_root = args.source.parent
    imported = 0
    for row in selected:
        representation = "sqa_frozen" if row["freeze_encoder"].lower() == "true" else "sqa_finetuned"
        report_path = source_root / row["report_path"]
        target = {
            "experiment_id": row["experiment_id"],
            "task": "sqa",
            "cohort": row["dataset_id"],
            "split": "validation",
            "corpus": "casanovo19pxd",
            "model_id": row["model_id"],
            "conditioning": row["precursor_conditioning"],
            "representation": representation,
            "metric_name": "classification/roc_auc",
            "value": row["metric_value"],
            "status": "complete",
            "priority": "primary",
            "selection_role": "development",
            "higher_is_better": "true",
            "report_path": str(report_path),
            "notes": (
                "Canonical completed SQA package; selected by validation AUROC. "
                f"Source checkpoint selection: {row['checkpoint_selection']}."
            ),
        }
        key = _key(target)
        if key in indexed:
            ledger[indexed[key]] = target
        else:
            indexed[key] = len(ledger)
            ledger.append(target)
        imported += 1

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(ledger)
    print(f"Upserted {imported} SQA validation rows from {args.source} into {args.ledger}")


if __name__ == "__main__":
    main()
