"""Import full-validation frozen external-embedding downstream baselines from W&B.

Only finished eval_only runs are eligible. The training runs, including their
limited validation monitor passes, are deliberately excluded.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import wandb

FIELDS = [
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
]
TASK_META = {
    "sqa": ("casanovo19pxd_balanced_charge1_10", "casanovo19pxd"),
    "chimericity": ("PXD024584_HYE_V1", "PXD024584_HYE"),
    "oxidized_met": ("PXD010613_oxidized_met_V1", "PXD010613"),
    "retention_time": ("run_aligned_retention_time_V1", "run_aligned_multispecies"),
}
METRICS = {
    "sqa": (("val_auc", "classification/roc_auc", True, "primary"),),
    "chimericity": (("val_auc", "classification/roc_auc", True, "primary"),),
    "oxidized_met": (
        ("val_auc", "classification/roc_auc", True, "primary"),
        ("val_matched_backbone_auc", "classification/matched_backbone_roc_auc", True, "secondary"),
    ),
    "retention_time": (
        ("retention_time_val_mae", "regression/mae", False, "primary"),
        ("retention_time_val_r2", "regression/r2", True, "secondary"),
        ("retention_time_val_spearman", "regression/spearman", True, "secondary"),
        ("retention_time_val_pearson", "regression/pearson", True, "diagnostic"),
        ("retention_time_val_delta_t95", "regression/delta_t95", False, "sensitivity"),
    ),
}


@dataclass(frozen=True)
class Spec:
    project: str
    group: str
    task: str
    model_id: str
    conditioning_from_name: bool = False


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def representation(task: str, run_name: str) -> str:
    if task == "retention_time":
        return "linear_ordinal_frozen" if "ordinal" in run_name else "linear_regression_frozen"
    finetuned = "finetune" in run_name
    if task == "sqa":
        return "sqa_finetuned" if finetuned else "sqa_frozen"
    return "binary_finetuned" if finetuned else "binary_frozen"


def specs() -> tuple[Spec, ...]:
    result = []
    for prefix, model_id in (
        ("casanovo_v4", "casanovo_v4_peak_mean"),
        ("instanovo_fm", "instanovo_fm"),
        ("hybrid300last", "hybrid_300_last"),
    ):
        for task, project in (
            ("sqa", "user/ms2-sqa"),
            ("chimericity", "user/ms2-chimericity"),
            ("oxidized_met", "user/ms2-oxidized-met"),
            ("retention_time", "user/ms2-retention-time"),
        ):
            result.append(Spec(
                project, f"{task}_{prefix}_validation", task, model_id,
                conditioning_from_name=(prefix == "hybrid300last"),
            ))
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()
    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != FIELDS:
        raise ValueError(f"Unexpected ledger schema: {args.ledger}")
    indexed = {key(row): i for i, row in enumerate(rows)}
    api = wandb.Api(timeout=60)
    imported = 0
    selected_runs = 0
    for spec in specs():
        runs = [
            run for run in api.runs(spec.project, per_page=100)
            if run.state == "finished" and run.group == spec.group and bool(run.config.get("eval_only"))
        ]
        if not runs:
            raise ValueError(f"No finished eval_only run in {spec.project} group {spec.group!r}")
        # A scheduler/cleanup retry can evaluate the same selected checkpoint twice.
        # Retain the newest completed record for each checkpoint deterministically.
        deduplicated = {}
        for run in runs:
            checkpoint = str(run.config.get("downstream_weights", ""))
            if not checkpoint:
                raise ValueError(f"{run.path} has no selected downstream_weights")
            prior = deduplicated.get(checkpoint)
            if prior is None or str(run.created_at) > str(prior.created_at):
                deduplicated[checkpoint] = run
        for run in deduplicated.values():
            selected_runs += 1
            cohort, corpus = TASK_META[spec.task]
            run_name = str(run.name).lower()
            rep = representation(spec.task, run_name)
            conditioning = "null" if spec.conditioning_from_name and "_null_" in run_name else "conditioned"
            if not spec.conditioning_from_name:
                conditioning = "not_applicable"
            for source_key, metric_name, higher_is_better, priority in METRICS[spec.task]:
                value = run.summary.get(source_key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{run.path} lacks finite {source_key}")
                row = {
                    "experiment_id": spec.group, "task": spec.task, "cohort": cohort,
                    "split": "validation", "corpus": corpus, "model_id": spec.model_id,
                    "conditioning": conditioning, "representation": rep,
                    "metric_name": metric_name, "value": repr(float(value)), "status": "complete",
                    "priority": priority, "selection_role": "development",
                    "higher_is_better": str(higher_is_better).lower(),
                    "report_path": f"https://wandb.ai/{spec.project}/runs/{run.id}",
                    "notes": "Full validation from finished eval_only external-embedding run; limited monitor pass excluded.",
                }
                row_key = key(row)
                if row_key in indexed:
                    rows[indexed[row_key]].update(row)
                else:
                    indexed[row_key] = len(rows)
                    rows.append(row)
                imported += 1
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {imported} full-validation metrics from {selected_runs} external eval_only runs.")


if __name__ == "__main__":
    main()
