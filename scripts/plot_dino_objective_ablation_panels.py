#!/usr/bin/env python3
"""Render fixed-geometry vector panels for the main DINO objective ablation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

PANELS = (
    {
        "metric": "pair_eval/all_validation/same_charge_10ppm/roc_auc",
        "title": "Strict pair AUROC",
        "filename": "strict_pair_auroc.pdf",
        "zero_based": False,
    },
    {
        "metric": "embed_eval/all_validation/broad_map",
        "title": "Broad retrieval mAP",
        "filename": "broad_retrieval_map.pdf",
        "zero_based": True,
    },
    {
        "metric": "probe/knn_val_acc",
        "title": "End-AA kNN validation",
        "filename": "end_aa_knn_validation.pdf",
        "zero_based": False,
    },
    {
        "metric": "probe/lin_test_acc",
        "title": "End-AA linear test",
        "filename": "end_aa_linear_test.pdf",
        "zero_based": False,
    },
    {
        "metric": "probe/train/rankme_norm",
        "title": "Normalized RankMe",
        "filename": "rankme_norm.pdf",
        "zero_based": True,
    },
    {
        "metric": "probe/train/self_cluster",
        "title": "Self-cluster",
        "filename": "self_cluster.pdf",
        "zero_based": True,
    },
)
ORDER = ("pure_distractor_global", "h26_inverse_control", "full_dual_objective")
LABELS = {
    "pure_distractor_global": "Pure mixture",
    "h26_inverse_control": "Mixture-free",
    "full_dual_objective": "Dual objective",
}
COLORS = {
    "pure_distractor_global": "#CC78BC",
    "h26_inverse_control": "#DE8F05",
    "full_dual_objective": "#0173B2",
}
MARKERS = {
    "pure_distractor_global": "s",
    "h26_inverse_control": "^",
    "full_dual_objective": "o",
}
FIGURE_SIZE = (3.45, 2.55)
AXES_RECT = (0.1575, 0.205, 0.685, 0.64)


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font_scale=1.4,
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#D7DCE1",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
        },
    )


def metric_points(rows, run_records, variant, metric):
    points = [row for row in rows if row["variant"] == variant and row["metric_name"] == metric]
    if not points:
        raise ValueError(f"No {metric!r} points for {variant!r}")
    x = [float(point["x"]) for point in points]
    if points[0]["x_axis"] == "trainer/global_step":
        steps_per_epoch = float(run_records[variant]["training_axis_conversion"]["steps_per_epoch"])
        x = [value / steps_per_epoch for value in x]
    y = [float(point["value"]) for point in points]
    return x, y


def render_panel(spec, rows, run_records, output_dir):
    figure = plt.figure(figsize=FIGURE_SIZE)
    axis = figure.add_axes(AXES_RECT)
    all_values = []
    handles = []
    for variant in ORDER:
        x, y = metric_points(rows, run_records, variant, spec["metric"])
        all_values.extend(y)
        marker_indices = sorted({round(i * (len(x) - 1) / 4) for i in range(5)})
        (line,) = axis.plot(
            x, y, color=COLORS[variant], marker=MARKERS[variant], markersize=4.3,
            markevery=marker_indices, linewidth=2.1, label=LABELS[variant], alpha=0.96,
        )
        handles.append(line)

    span = max(all_values) - min(all_values)
    padding = max(0.008, 0.10 * span)
    lower = 0.0 if spec["zero_based"] else min(all_values) - padding
    upper = max(all_values) + padding
    axis.set_xlim(0, 100)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Epoch", fontsize=13)
    axis.tick_params(labelsize=11.5, width=1.2, length=4)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(axis="x", visible=False)
    axis.grid(axis="y", visible=True)
    axis.spines[["top", "right"]].set_visible(False)

    figure.legend(
        handles=handles, labels=[LABELS[variant] for variant in ORDER],
        loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3,
        frameon=False, fontsize=8.5, handlelength=1.5,
        columnspacing=0.9, handletextpad=0.4, borderaxespad=0,
    )

    output = output_dir / spec["filename"]
    figure.savefig(output, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    configure_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("results/representation/dino_objective_ablation_100epoch_wandb"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "main_paper_panels"

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    run_records = {run["variant"]: run for run in manifest["runs"]}
    if missing := set(ORDER) - set(run_records):
        raise ValueError(f"Manifest is missing main-ablation runs: {sorted(missing)}")
    with (args.input_dir / manifest["curves_file"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [render_panel(spec, rows, run_records, output_dir) for spec in PANELS]
    print(f"Wrote {len(outputs)} fixed-geometry ablation panels to {output_dir}")


if __name__ == "__main__":
    main()
