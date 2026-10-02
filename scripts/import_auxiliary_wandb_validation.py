"""Import V1 auxiliary-task validation metrics from Weights & Biases.

Only validation-history keys are requested. The importer selects each run's
checkpoint-selection validation event (max AUROC for binary tasks, minimum MAE
for retention time), then records metrics logged at that same event. It never
reads or writes test metrics and is idempotent by canonical ledger key.
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


@dataclass(frozen=True)
class TaskSpec:
    project: str
    group: str
    task: str
    cohort: str
    corpus: str
    monitor: str
    monitor_higher_is_better: bool
    metrics: tuple[tuple[str, str, bool, str], ...]


SPECS = (
    TaskSpec(
        "user/ms2-chimericity", "chimericity_auxiliary_ablation_v1",
        "chimericity", "PXD024584_HYE_V1", "PXD024584_HYE",
        "val_auc", True,
        (("val_auc", "classification/roc_auc", True, "primary"),),
    ),
    TaskSpec(
        "user/ms2-oxidized-met", "oxidized_met_auxiliary_ablation_v1",
        "oxidized_met", "PXD010613_oxidized_met_V1", "PXD010613",
        "val_auc", True,
        (
            ("val_auc", "classification/roc_auc", True, "primary"),
            ("val_matched_backbone_auc", "classification/matched_backbone_roc_auc", True, "secondary"),
        ),
    ),
    TaskSpec(
        "user/ms2-retention-time", "retention_time_auxiliary_ablation_v1",
        "retention_time", "run_aligned_retention_time_V1", "run_aligned_multispecies",
        "retention_time_val_mae", False,
        (
            ("retention_time_val_mae", "regression/mae", False, "primary"),
            ("retention_time_val_r2", "regression/r2", True, "secondary"),
            ("retention_time_val_spearman", "regression/spearman", True, "secondary"),
            ("retention_time_val_pearson", "regression/pearson", True, "diagnostic"),
            ("retention_time_val_delta_t95", "regression/delta_t95", False, "sensitivity"),
        ),
    ),
)


def _key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def _read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LEDGER_FIELDS:
            raise ValueError(f"Unexpected ledger schema in {path}: {reader.fieldnames}")
        return list(reader)


def _metadata(task: str, run_name: str) -> tuple[str, str, str]:
    name = run_name.lower()
    if "instanovo" in name:
        model_id = "instanovo_fm"
    elif "scratch" in name or "_random_" in name:
        model_id = "scratch"
    elif "binned1024_spectrum_only" in name:
        model_id = "binned1024_spectrum_only"
    elif "binned1024_precursor_metadata" in name:
        model_id = "binned1024_precursor_metadata"
    elif "hybrid300_last" in name or "hybrid300last" in name:
        model_id = "hybrid_300_last"
    elif "hybrid300e219" in name:
        model_id = "hybrid_300_epoch219"
    else:
        raise ValueError(f"Cannot infer model identity from run {run_name!r}")

    conditioning = "not_applicable" if "instanovo" in name else ("not_used" if "spectrum_only" in name else ("null" if "_null_" in name else "conditioned"))
    if task == "retention_time":
        prediction = "ordinal" if "_ordinal_" in name else "regression"
        representation = f"linear_{prediction}_frozen"
    else:
        representation = "binary_frozen" if "_frozen_" in name else "binary_finetuned"
    return model_id, conditioning, representation


def _selected_event(run, spec: TaskSpec) -> tuple[dict, int | None, int | None]:
    keys = ["_step", "epoch", spec.monitor, *(source for source, *_ in spec.metrics)]
    candidates = []
    for row in run.scan_history(keys=keys, page_size=10_000):
        value = row.get(spec.monitor)
        if isinstance(value, (int, float)):
            candidates.append(row)
    if not candidates:
        raise ValueError(f"{run.path} has no numeric {spec.monitor} history")
    selector = max if spec.monitor_higher_is_better else min
    selected = selector(candidates, key=lambda row: float(row[spec.monitor]))
    epoch = selected.get("epoch")
    step = selected.get("_step")
    return selected, None if epoch is None else int(epoch), None if step is None else int(step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument("--task", choices=[spec.task for spec in SPECS], help="Import one task only.")
    parser.add_argument("--run-id", action="append", help="Import only this W&B run ID; repeatable.")
    parser.add_argument("--group", help="Override the configured W&B group for one task import.")
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    rows = _read_ledger(args.ledger)
    indexed = {_key(row): index for index, row in enumerate(rows)}
    imported = []
    specs = [spec for spec in SPECS if args.task in (None, spec.task)]
    for spec in specs:
        group = args.group or spec.group
        runs = [
            run for run in api.runs(spec.project, per_page=100)
            if run.state == "finished" and run.group == group
        ]
        if args.run_id:
            runs = [run for run in runs if run.id in set(args.run_id)]
        if not runs:
            raise ValueError(f"No finished selected runs in {spec.project} group {group!r}")
        print(f"Importing {len(runs)} validation runs from {spec.project} ({group}).", flush=True)
        for run in runs:
            print(f"  reading {run.id} {run.name}", flush=True)
            model_id, conditioning, representation = _metadata(spec.task, run.name)
            selected, epoch, step = _selected_event(run, spec)
            for source_key, metric_name, higher_is_better, priority in spec.metrics:
                value = selected.get(source_key)
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{run.path} did not log {source_key} at its selected event")
                row = {
                    "experiment_id": group,
                    "task": spec.task,
                    "cohort": spec.cohort,
                    "split": "validation",
                    "corpus": spec.corpus,
                    "model_id": model_id,
                    "conditioning": conditioning,
                    "representation": representation,
                    "metric_name": metric_name,
                    "value": repr(float(value)),
                    "status": "complete",
                    "priority": priority,
                    "selection_role": "development",
                    "higher_is_better": str(higher_is_better).lower(),
                    "report_path": f"https://wandb.ai/{spec.project}/runs/{run.id}",
                    "notes": (
                        f"Exploratory V1 task ({group}). W&B run {run.id}; selected by {spec.monitor} "
                        f"at epoch {epoch}, step {step}."
                    ),
                }
                key = _key(row)
                if key in indexed:
                    rows[indexed[key]] = row
                else:
                    indexed[key] = len(rows)
                    rows.append(row)
                imported.append((spec.task, run.id, metric_name, float(value), epoch, step))

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for task, run_id, metric, value, epoch, step in imported:
        print(f"{task} {run_id} {metric}={value:.6f} epoch={epoch} step={step}")
    print(f"Upserted {len(imported)} V1 auxiliary validation metric rows into {args.ledger}")


if __name__ == "__main__":
    main()
