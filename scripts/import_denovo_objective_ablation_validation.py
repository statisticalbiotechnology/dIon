#!/usr/bin/env python3
"""Import end-to-end 100-epoch objective-ablation validation results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path("/path/to/results/denovo_eval/denovo_objective_ablation_100epoch_mskb_final_validation")
FROZEN_ROOT = Path("/path/to/results/denovo_eval/denovo_frozen_encoder_transfer_100epoch_mskb_final_validation")
LEDGER = Path("results/paper/metrics_long.csv")
FIELDS = ("experiment_id", "task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name", "value", "status", "priority", "selection_role", "higher_is_better", "report_path", "notes")
RUNS = (
    ("dual_objective", "ablation_100ep_dual_objective", "dual_objective_100epoch", "gbx8pw3r", 73, 0.7824480533599854),
    ("pure_mixture", "ablation_100ep_pure_mixture", "pure_mixture_100epoch", "xdl3mt5t", 79, 0.7780551314353943),
    ("mixture_free_control", "ablation_100ep_mixture_free_control", "mixture_free_control_100epoch", "b5mhp9m9", 79, 0.7637779712677002),
    ("scratch_encoderld", "ablation_scratch_encoderld", "scratch_encoderld", "mn4nwa8l", 78, 0.7439097166061401),
)
FROZEN_RUNS = (
    ("dual_objective_100epoch", "frozen_ablation_100ep_dual_objective", "dual_objective_100epoch", "rgkvnzlz", 75, 0.6749201416969299),
    ("pure_mixture_100epoch", "frozen_ablation_100ep_pure_mixture", "pure_mixture_100epoch", "rdiykbew", 77, 0.6383786201477051),
    ("mixture_free_control_100epoch", "frozen_ablation_100ep_mixture_free_control", "mixture_free_control_100epoch", "x5p2tqls", 76, 0.6096246242523193),
    ("random_encoder", "frozen_ablation_random_encoder", "random_encoder", "e9gly5mw", 79, 0.5815694928169250),
)
KEY_FIELDS = ("task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_ROOT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()
    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    rows = [row for row in rows if not (
        row["task"] == "denovo"
        and row["cohort"] in {
            "objective_ablation_100epoch_mskb_final_validation",
            "frozen_encoder_transfer_100epoch_mskb_final_validation",
        }
    )]
    indexed = {tuple(row[field] for field in KEY_FIELDS): index for index, row in enumerate(rows)}
    for directory, model_id, objective, run_id, selected_epoch, expected_precision in RUNS:
        run_dir = args.root / directory
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if (manifest.get("objective") != objective or manifest.get("regime") != "end_to_end"
                or manifest.get("encoder_frozen") is not False or manifest.get("wandb_run_id") != run_id
                or manifest.get("max_peaks") != 200 or manifest.get("decoder_epoch_budget") != 80
                or manifest.get("precursor_conditioning") != "conditioned"):
            raise ValueError(f"Unexpected end-to-end ablation manifest: {directory}")
        if (metrics.get("split") != "validation" or metrics.get("development_set_diagnostic") is not True
                or metrics.get("held_out_test_run") is not False
                or metrics.get("regime") != "end_to_end" or metrics.get("objective") != objective
                or metrics.get("wandb_run_id") != run_id or metrics.get("selected_epoch") != selected_epoch
                or metrics.get("selection_mode") != "max"
                or metrics.get("peptide_precision_at_full_coverage") != expected_precision):
            raise ValueError(f"Unexpected selected validation metric: {directory}")
        row = {
            "experiment_id": "denovo_objective_ablation_100epoch_mskb_final_validation_20260924",
            "task": "denovo", "cohort": "objective_ablation_100epoch_mskb_final_validation",
            "split": "validation", "corpus": "mskb_final_val10k_seed42",
            "model_id": model_id, "conditioning": "conditioned",
            "representation": "end_to_end_100epoch_objective_ablation",
            "metric_name": "denovo/peptide_precision", "value": repr(expected_precision),
            "status": "complete", "priority": "primary", "selection_role": "development",
            "higher_is_better": "true", "report_path": str(run_dir / "metrics.json"),
            "notes": "MSKB-final development-validation diagnostic only: encoder and decoder were fine-tuned end to end for 80 epochs. Selected maximum autoregressively decoded peptide precision; not a held-out test or final-model result. The canonical 10,004-row validation split was evaluated as 10,016 distributed items because the sampler repeated 12 rows.",
        }
        row_key = tuple(row[field] for field in KEY_FIELDS)
        if row_key in indexed:
            rows[indexed[row_key]].update(row)
        else:
            indexed[row_key] = len(rows)
            rows.append(row)
    for directory, model_id, objective, run_id, selected_epoch, expected_precision in FROZEN_RUNS:
        run_dir = args.frozen_root / directory
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if (manifest.get("objective") != objective
                or manifest.get("regime") != "frozen_encoder_transfer"
                or manifest.get("encoder_frozen") is not True
                or manifest.get("wandb_run_id") != run_id
                or manifest.get("max_peaks") != 200
                or manifest.get("decoder_epoch_budget") != 80
                or manifest.get("precursor_conditioning") != "conditioned"):
            raise ValueError(f"Unexpected frozen-transfer manifest: {directory}")
        if (metrics.get("split") != "validation"
                or metrics.get("development_set_diagnostic") is not True
                or metrics.get("regime") != "frozen_encoder_transfer"
                or metrics.get("objective") != objective
                or metrics.get("wandb_run_id") != run_id
                or metrics.get("selected_epoch") != selected_epoch
                or metrics.get("selection_mode") != "max"
                or metrics.get("peptide_precision_at_full_coverage") != expected_precision):
            raise ValueError(f"Unexpected frozen-transfer metric: {directory}")
        row = {
            "experiment_id": "denovo_frozen_encoder_transfer_100epoch_mskb_final_validation_20260924",
            "task": "denovo", "cohort": "frozen_encoder_transfer_100epoch_mskb_final_validation",
            "split": "validation", "corpus": "mskb_final_val10k_seed42",
            "model_id": model_id, "conditioning": "conditioned",
            "representation": "frozen_100epoch_encoder_transfer",
            "metric_name": "denovo/peptide_precision", "value": repr(expected_precision),
            "status": "complete", "priority": "primary", "selection_role": "development",
            "higher_is_better": "true", "report_path": str(run_dir / "metrics.json"),
            "notes": "MSKB-final development-validation diagnostic only: the encoder was frozen and a fresh decoder trained for 80 epochs. Selected maximum autoregressively decoded peptide precision; not a held-out test or final-model result. The canonical 10,004-row validation split was evaluated as 10,016 distributed items because the sampler repeated 12 rows.",
        }
        row_key = tuple(row[field] for field in KEY_FIELDS)
        if row_key in indexed:
            raise ValueError(f"Duplicate frozen-transfer ledger key: {directory}")
        indexed[row_key] = len(rows)
        rows.append(row)
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Imported 4 end-to-end and 4 frozen-transfer validation metrics.")


if __name__ == "__main__":
    main()
