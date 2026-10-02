#!/usr/bin/env python3
"""Plot the validated approximately 100-epoch DINO objective-ablation curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

PANELS = (
    ("pair_eval/all_validation/same_charge_10ppm/roc_auc", "Strict pair AUROC", "Global step"),
    ("pair_eval/all_validation/same_charge_10ppm/average_precision", "Strict pair AP", "Global step"),
    ("embed_eval/all_validation/broad_map", "Broad retrieval mAP", "Global step"),
    ("probe/knn_val_acc", "End-AA kNN validation", "Epoch"),
    ("probe/lin_test_acc", "End-AA linear test", "Epoch"),
    ("probe/lin_val_acc", "End-AA linear validation", "Epoch"),
    ("probe/train/rankme_norm", "Normalized RankMe", "Epoch"),
    ("probe/train/stable_rank", "Stable rank", "Epoch"),
    ("probe/train/self_cluster", "Self-cluster", "Epoch"),
)
ORDER = ("plain_dino", "pure_distractor_global", "h26_inverse_control", "full_dual_objective")
LABELS = {
    "plain_dino": "Plain DINO (300 peaks)",
    "pure_distractor_global": "Pure mixture",
    "h26_inverse_control": "H26 inverse",
    "full_dual_objective": "Dual objective",
}
COLORS = {
    "plain_dino": "#7f8c8d", "pure_distractor_global": "#c44e52",
    "h26_inverse_control": "#dd8452", "full_dual_objective": "#2a6f97",
}
MARKERS = {
    "plain_dino": "D", "pure_distractor_global": "s",
    "h26_inverse_control": "^", "full_dual_objective": "o",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("results/representation/dino_objective_ablation_100epoch_wandb"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input_dir / "ablation_curves_preview.png"

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    known_ids = {run["variant"]: run["run_id"] for run in manifest["runs"]}
    if tuple(known_ids) != ORDER:
        raise ValueError(f"Unexpected variant order in manifest: {tuple(known_ids)!r}")
    with (args.input_dir / manifest["curves_file"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2), constrained_layout=True)
    for axis, (metric, title, xlabel) in zip(axes.flat, PANELS):
        for variant in ORDER:
            points = [row for row in rows if row["variant"] == variant and row["metric_name"] == metric]
            if not points:
                continue
            x = [float(point["x"]) for point in points]
            y = [float(point["value"]) for point in points]
            axis.plot(
                x, y, color=COLORS[variant], marker=MARKERS[variant], markersize=3.8,
                linewidth=1.7, label=LABELS[variant], alpha=0.95,
            )
        axis.set_title(title, fontsize=10, fontweight="semibold")
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", color="#d9dee3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if metric.startswith("pair_eval/") or metric.endswith("_acc"):
            axis.set_ylim(bottom=0.45)
        elif metric.startswith("embed_eval/"):
            axis.set_ylim(bottom=0)
        elif metric.endswith(("rankme_norm", "self_cluster")):
            axis.set_ylim(bottom=0)

    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("DINO objective ablation: validation trajectories", fontsize=13, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Wrote preview plot: {output}")


if __name__ == "__main__":
    main()
