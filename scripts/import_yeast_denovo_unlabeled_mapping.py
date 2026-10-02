#!/usr/bin/env python3
"""Import the validated unlabeled yeast proteome-mapping benchmark to the ledger."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path("/path/to/results/denovo_eval/yeast_denovo_benchmark")
LEDGER = Path("results/paper/metrics_long.csv")
FIELDS = ("experiment_id", "task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name", "value", "status", "priority", "selection_role", "higher_is_better", "report_path", "notes")
MODELS = {"dion": ("v5_hybrid300_finetuned", "full_finetune"), "casanovo": ("casanovo_v5_2_1", "released_denovo"), "instanovo": ("instanovo_v1_2_2", "released_denovo")}
METRICS = (("emitted_fraction", "label_free/emitted_fraction", "primary"), ("mapped_of_emitted", "label_free/proteome_mapped_fraction_emitted", "primary"), ("mapped_of_all_spectra", "label_free/proteome_mapped_fraction_all", "primary"), ("mapped_tryptic_of_emitted", "label_free/proteome_mapped_fraction_tryptic_emitted", "secondary"), ("chance_reversed", "label_free/proteome_mapped_fraction_reversed_decoy", "secondary"), ("chance_shuffled", "label_free/proteome_mapped_fraction_shuffled_decoy", "secondary"), ("modified_call_fraction", "label_free/modified_call_fraction", "diagnostic"))
KEY_FIELDS = ("task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name")


def validate_roster(root: Path) -> None:
    rosters, declines = [], {}
    for source in MODELS:
        with (root / f"{source}_predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["scan_id"] for row in rows]
        if len(rows) != 39_460 or len(set(ids)) != 39_460 or any(row["true_sequence"] for row in rows):
            raise ValueError(f"Invalid unlabeled yeast prediction roster: {source}")
        rosters.append(set(ids))
        declines[source] = sum(row["no_prediction"].lower() == "true" for row in rows)
    if any(roster != rosters[0] for roster in rosters[1:]) or declines != {"dion": 0, "casanovo": 3_934, "instanovo": 0}:
        raise ValueError("Unexpected yeast prediction roster or decline accounting.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()
    validate_roster(args.root)
    report_path = args.root / "yeast_mapping_three_way.json"
    readme = (args.root / "README.md").read_text()
    if "CORRECTION, 2026-09-23" not in readme or "SUPERSEDED_BUGGY" not in readme:
        raise ValueError("Expected corrected yeast benchmark README notice is missing.")
    report = json.loads(report_path.read_text())
    if report.get("proteome", {}).get("proteins") != 6_067 or report.get("rule", {}).get("matching") != "exact substring vs concatenated proteome (protein-delimited)":
        raise ValueError("Unexpected yeast proteome report.")
    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    indexed = {tuple(row[field] for field in KEY_FIELDS): index for index, row in enumerate(rows)}
    for source, (model_id, representation) in MODELS.items():
        overall = report["models"][source]["overall"]
        if overall["spectra_total"] != 39_460:
            raise ValueError(f"Unexpected spectrum count for {source}")
        notes = "Corrected 2026-09-23 unlabeled yeast MS2 proteome-mapping proxy on PXD026806 MDR27022021_S01_2_DDA01 (39,460 spectra). No peptide labels are used: bracketed/parenthesised modifications are removed whole, I/L are collapsed, and calls are exact protein-delimited substrings of UniProt UP000002311. This is not peptide accuracy. It supersedes the earlier asymmetric modification-parsing error; Casanovo's explicit declines are retained in coverage and all-spectrum denominators."
        for source_metric, metric_name, priority in METRICS:
            value = overall.get(source_metric)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Missing numeric {source_metric} for {source}")
            row = {"experiment_id": "denovo_yeast_unlabeled_proteome_mapping_20260922", "task": "denovo", "cohort": "yeast_unlabeled_proteome_mapping", "split": "test", "corpus": "PXD026806_MDR27022021_S01_2_DDA01", "model_id": model_id, "conditioning": "conditioned" if source == "dion" else "not_applicable", "representation": representation, "metric_name": metric_name, "value": repr(float(value)), "status": "complete", "priority": priority, "selection_role": "selected", "higher_is_better": "true", "report_path": str(report_path), "notes": notes}
            row_key = tuple(row[field] for field in KEY_FIELDS)
            if row_key in indexed:
                rows[indexed[row_key]].update(row)
            else:
                indexed[row_key] = len(rows)
                rows.append(row)
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Imported 21 validated unlabeled yeast mapping metrics.")


if __name__ == "__main__":
    main()
