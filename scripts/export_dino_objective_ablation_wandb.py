#!/usr/bin/env python3
"""Export validated W&B curves for the approximately 100-epoch DINO ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb


RUNS = (
    {
        "variant": "plain_dino",
        "wandb_path": "user/dion-pretraining/7waklwb8",
        "priority": "lower_priority",
        "expected_max_peaks": 300,
        "require_pair_metrics": False,
    },
    {
        "variant": "pure_distractor_global",
        "wandb_path": "user/dion-pretraining/7cee11xe",
        "priority": "primary",
        "expected_max_peaks": 200,
        "require_pair_metrics": True,
    },
    {
        "variant": "h26_inverse_control",
        "wandb_path": "user/dion-pretraining/8fdixqb0",
        "priority": "primary",
        "expected_max_peaks": 200,
        "require_pair_metrics": True,
    },
    {
        "variant": "full_dual_objective",
        "wandb_path": "user/dion-pretraining/hjdp7mn2",
        "priority": "primary",
        "expected_max_peaks": 200,
        "require_pair_metrics": True,
    },
)

RETRIEVAL_METRICS = tuple(
    f"embed_eval/{cohort}/broad_map"
    for cohort in ("all_validation", "bact_val_cb", "kingdoms_val_cb", "9spec_v2_val_cb")
)
PAIR_METRICS = tuple(
    f"pair_eval/{cohort}/same_charge_10ppm/{metric}"
    for cohort in ("all_validation", "bact_val_cb", "kingdoms_val_cb", "9spec_v2_val_cb")
    for metric in ("roc_auc", "average_precision")
)
PROBE_METRICS = (
    "probe/knn_val_acc",
    "probe/lin_test_acc",
    "probe/lin_train_loss",
    "probe/lin_val_acc",
    "probe/lin_val_loss",
    "probe/train/alpha",
    "probe/train/coh_max",
    "probe/train/cond",
    "probe/train/nesum",
    "probe/train/rankme_norm",
    "probe/train/self_cluster",
    "probe/train/stable_rank",
)
FIELDS = (
    "variant", "priority", "wandb_path", "run_id", "run_name", "metric_family",
    "metric_name", "x_axis", "x", "wandb_step", "epoch", "trainer_global_step", "value",
)
ENDPOINT_FIELDS = (
    "variant", "priority", "wandb_path", "run_id", "run_name", "metric_family",
    "metric_name", "x_axis", "final_x", "final_value", "peak_x", "peak_value",
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def git_revision() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
    ).stdout.strip())
    return {"commit": revision, "dirty": dirty}


def validate_points(points: list[dict[str, Any]], metric: str, axis: str, summary: dict[str, Any]) -> None:
    if not points:
        raise ValueError(f"No history points found for required metric {metric!r}.")
    x = [float(point[axis]) for point in points]
    if any(right <= left for left, right in zip(x, x[1:])):
        raise ValueError(f"Metric {metric!r} has a duplicate or non-increasing {axis} axis: {x!r}")
    if metric in summary and finite_number(summary[metric]):
        if not math.isclose(float(summary[metric]), float(points[-1][metric]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"Last history value for {metric!r} does not match W&B summary: "
                f"{points[-1][metric]!r} != {summary[metric]!r}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/representation/dino_objective_ablation_100epoch_wandb"),
    )
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    api = wandb.Api(timeout=args.timeout)
    curve_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    archived_runs: list[dict[str, Any]] = []
    requested_metrics = (*RETRIEVAL_METRICS, *PAIR_METRICS, *PROBE_METRICS)

    for spec in RUNS:
        run = api.run(spec["wandb_path"])
        if run.id != spec["wandb_path"].rsplit("/", 1)[-1] or run.state != "finished":
            raise ValueError(f"Unexpected run identity/state for {spec['wandb_path']}: {run.id}, {run.state}")
        config = dict(run.config)
        if config.get("max_peaks") != spec["expected_max_peaks"]:
            raise ValueError(
                f"{run.path} max_peaks={config.get('max_peaks')!r}, expected {spec['expected_max_peaks']}."
            )
        summary = dict(run.summary)
        available = [metric for metric in requested_metrics if metric in summary]
        required = list(PROBE_METRICS)
        if spec["require_pair_metrics"]:
            required.extend(RETRIEVAL_METRICS)
            required.extend(PAIR_METRICS)
        missing = sorted(set(required) - set(available))
        if missing:
            raise ValueError(f"{run.path} is missing required metrics: {missing}")

        history = list(run.scan_history(
            keys=["_step", "trainer/global_step", "epoch", *available], page_size=10_000
        ))
        event_steps = [row.get("_step") for row in history]
        if not event_steps or any(not finite_number(step) for step in event_steps):
            raise ValueError(f"{run.path} has invalid W&B event steps.")
        if any(right <= left for left, right in zip(event_steps, event_steps[1:])):
            raise ValueError(f"{run.path} has duplicate or non-increasing W&B event steps.")
        epoch_step_pairs = [
            (int(row["trainer/global_step"]), int(row["epoch"]))
            for row in history
            if finite_number(row.get("trainer/global_step")) and finite_number(row.get("epoch"))
        ]
        if not epoch_step_pairs:
            raise ValueError(f"{run.path} has no co-logged epoch/global-step records.")
        steps_per_epoch = (max(step for step, _ in epoch_step_pairs) + 1) / (
            max(epoch for _, epoch in epoch_step_pairs) + 1
        )
        if any(math.floor(step / steps_per_epoch) != epoch for step, epoch in epoch_step_pairs):
            raise ValueError(f"{run.path} has an inconsistent global-step-to-epoch mapping.")

        counts: dict[str, int] = {}
        endpoints: dict[str, dict[str, float]] = {}
        for metric in available:
            if metric.startswith("pair_eval/"):
                family = "strict_pair_validation"
            elif metric.startswith("embed_eval/"):
                family = "broad_retrieval_validation"
            else:
                family = "probe"
            axis = "epoch" if family == "probe" else "trainer/global_step"
            points = [
                row for row in history
                if finite_number(row.get(metric)) and finite_number(row.get(axis))
            ]
            validate_points(points, metric, axis, summary)
            counts[metric] = len(points)
            endpoints[metric] = {
                "x": float(points[-1][axis]), "value": float(points[-1][metric])
            }
            peak = max(points, key=lambda point: float(point[metric]))
            endpoint_rows.append({
                "variant": spec["variant"], "priority": spec["priority"],
                "wandb_path": spec["wandb_path"], "run_id": run.id, "run_name": run.name,
                "metric_family": family, "metric_name": metric, "x_axis": axis,
                "final_x": points[-1][axis], "final_value": points[-1][metric],
                "peak_x": peak[axis], "peak_value": peak[metric],
            })
            for point in points:
                curve_rows.append({
                    "variant": spec["variant"], "priority": spec["priority"],
                    "wandb_path": spec["wandb_path"], "run_id": run.id, "run_name": run.name,
                    "metric_family": family, "metric_name": metric,
                    "x_axis": axis, "x": point[axis], "wandb_step": point["_step"],
                    "epoch": point.get("epoch", ""),
                    "trainer_global_step": point.get("trainer/global_step", ""),
                    "value": point[metric],
                })

        pair_points = [count for metric, count in counts.items() if metric.startswith("pair_eval/")]
        probe_points = [count for metric, count in counts.items() if metric.startswith("probe/")]
        archived_runs.append({
            "variant": spec["variant"], "priority": spec["priority"],
            "wandb_path": spec["wandb_path"], "run_id": run.id, "run_name": run.name,
            "state": run.state, "url": run.url, "created_at": str(run.created_at),
            "updated_at": str(getattr(run, "updated_at", "")),
            "expected_max_peaks": spec["expected_max_peaks"],
            "pair_curves_available": bool(pair_points),
            "history_event_count": len(history),
            "training_axis_conversion": {
                "steps_per_epoch": steps_per_epoch,
                "formula": "continuous_epoch = trainer/global_step / steps_per_epoch",
                "validated_epoch_step_pair_count": len(epoch_step_pairs),
            },
            "metric_point_counts": counts,
            "metric_endpoints": endpoints,
            "axis_validation": {
                "strict_pair_validation": "trainer/global_step; strictly increasing per metric",
                "broad_retrieval_validation": "trainer/global_step; strictly increasing per metric",
                "probe": "epoch; strictly increasing per metric",
                "wandb_event_step": "strictly increasing over unsampled scan_history records",
                "summary_check": "last history value equals W&B summary for every exported metric",
            },
            "config": {str(key): json_safe(value) for key, value in sorted(config.items())},
        })

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output, delete=False, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve_rows)
        csv_tmp = Path(handle.name)
    csv_path = output / "curves_long.csv"
    os.replace(csv_tmp, csv_path)
    csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    with tempfile.NamedTemporaryFile("w", dir=output, delete=False, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENDPOINT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(endpoint_rows)
        endpoint_tmp = Path(handle.name)
    endpoint_path = output / "metric_endpoints.csv"
    os.replace(endpoint_tmp, endpoint_path)
    endpoint_sha256 = hashlib.sha256(endpoint_path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Weights & Biases public API using unsampled scan_history",
        "git": git_revision(),
        "curves_file": csv_path.name,
        "curves_sha256": csv_sha256,
        "curve_row_count": len(curve_rows),
        "endpoints_file": endpoint_path.name,
        "endpoints_sha256": endpoint_sha256,
        "endpoint_row_count": len(endpoint_rows),
        "runs": archived_runs,
        "scope_note": (
            "The three primary controls are matched max_peaks=200 approximately 100-epoch runs. "
            "Plain DINO is an older max_peaks=300 lower-priority control and has no strict pair callback curves."
        ),
        "selection_note": (
            "All reported callback cohorts are validation/development subsets (val_cb). "
            "They support objective ablation and model selection, not held-out test claims."
        ),
    }
    atomic_text(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(curve_rows):,} validated curve points from {len(archived_runs)} runs: {output}")


if __name__ == "__main__":
    main()
