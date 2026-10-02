#!/usr/bin/env python3
"""Plot the main-paper DINO objective ablation from persisted validated curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

PANELS = (
    ("pair_eval/all_validation/same_charge_10ppm/roc_auc", "Strict pair AUROC", "Global step"),
    ("embed_eval/all_validation/broad_map", "Broad retrieval mAP", "Global step"),
    ("probe/knn_val_acc", "End-AA kNN validation", "Epoch"),
    ("probe/lin_test_acc", "End-AA linear test", "Epoch"),
    ("probe/train/rankme_norm", "Normalized RankMe", "Epoch"),
    ("probe/train/self_cluster", "Self-cluster", "Epoch"),
)
ORDER = ("pure_distractor_global", "h26_inverse_control", "full_dual_objective")
LABELS = {
    "pure_distractor_global": "Pure mixture",
    "h26_inverse_control": "H26 inverse",
    "full_dual_objective": "Dual objective",
}
COLORS = {
    "pure_distractor_global": "#c44e52",
    "h26_inverse_control": "#dd8452",
    "full_dual_objective": "#2a6f97",
}
MARKERS = {
    "pure_distractor_global": "s",
    "h26_inverse_control": "^",
    "full_dual_objective": "o",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("results/representation/dino_objective_ablation_100epoch_wandb"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input_dir / "main_paper_ablation.png"

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    run_records = {run["variant"]: run for run in manifest["runs"]}
    run_ids = {variant: run["run_id"] for variant, run in run_records.items()}
    missing = set(ORDER) - set(run_ids)
    if missing:
        raise ValueError(f"Manifest is missing main-ablation runs: {sorted(missing)}")
    with (args.input_dir / manifest["curves_file"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), constrained_layout=True)
    for axis, (metric, title, xlabel) in zip(axes.flat, PANELS):
        panel_values = []
        for variant in ORDER:
            points = [row for row in rows if row["variant"] == variant and row["metric_name"] == metric]
            if not points:
                raise ValueError(f"No {metric!r} points for {variant!r}")
            x = [float(point["x"]) for point in points]
            if points[0]["x_axis"] == "trainer/global_step":
                steps_per_epoch = float(
                    run_records[variant]["training_axis_conversion"]["steps_per_epoch"]
                )
                x = [value / steps_per_epoch for value in x]
            y = [float(point["value"]) for point in points]
            panel_values.extend(y)
            axis.plot(
                x,
                y,
                color=COLORS[variant], marker=MARKERS[variant], markersize=4.2,
                linewidth=1.8, label=LABELS[variant], alpha=0.96,
            )
        axis.set_title(title, fontsize=10.5, fontweight="semibold")
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", color="#d9dee3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if metric.startswith("pair_eval/"):
            span = max(panel_values) - min(panel_values)
            padding = max(0.01, 0.12 * span)
            axis.set_ylim(min(panel_values) - padding, max(panel_values) + padding)
        elif metric.endswith("_acc"):
            axis.set_ylim(bottom=0.45)
        elif metric.startswith("embed_eval/") or metric.endswith(("rankme_norm", "self_cluster")):
            axis.set_ylim(bottom=0)

    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Dual-objective ablation on validation probes", fontsize=13, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Wrote main-paper ablation plot: {output}")


if __name__ == "__main__":
    main()
