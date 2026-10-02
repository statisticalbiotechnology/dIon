#!/usr/bin/env python3
"""Import capped Kingdoms de novo test metrics from the four canonical W&B runs.

The 100k-per-species Kingdoms evaluation intentionally has no prediction table.
This importer identifies each run by project plus its exact selected checkpoint
basename, reads aggregate test *epoch* metrics only, and archives aggregate plus
per-species on-test-epoch values in a JSON sidecar. Step metrics are rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import wandb

FIELDS = (
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
)
AGGREGATE = {
    "denovo_tf_test_pep_prec_epoch": "denovo/peptide_precision",
    "denovo_tf_test_aa_prec_epoch": "denovo/aa_precision",
    "denovo_tf_test_aa_recall_epoch": "denovo/aa_recall",
}
SPECIES_SUFFIXES = ("n_spectra", "pep_prec", "aa_prec", "aa_recall")
RUNS = (
    ("user/denovo-kingdoms-mskb", "epoch=72-denovo_tf_val_pep_prec=0.80.ckpt", "mskb_hybrid300_finetuned"),
    ("user/denovo-kingdoms-mskb", "epoch=78-denovo_tf_val_pep_prec=0.74.ckpt", "mskb_scratch_encoderld"),
    ("user/denovo-kingdoms-v5", "epoch=37-denovo_tf_val_pep_prec=0.60.ckpt", "v5_hybrid300_finetuned"),
    ("user/denovo-kingdoms-v5", "epoch=41-denovo_tf_val_pep_prec=0.57.ckpt", "v5_scratch_encoderld"),
)


PENDING_1000 = (
    ("mskb_hybrid300_finetuned_maxpeaks1000", "Kingdoms 1000-peak evaluation job 2482474_0 pending."),
    ("v5_hybrid300_finetuned_maxpeaks1000", "Kingdoms 1000-peak evaluation job 2482475_0 pending."),
    ("casanovo_v5_2_1", "Released Casanovo v5.2.1 inference pending; unsupported charge >4 inputs are retained as no/wrong predictions in the full capped Kingdoms denominator."),
    ("instanovo_v1_2_2", "Released InstaNovo v1.2.2 inference pending on the full-charge capped Kingdoms denominator."),
)
RELEASED_DENOVO_BASELINES = {"casanovo_v5_2_1", "instanovo_v1_2_2"}

def finite(value: Any) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Expected finite numeric metric, received {value!r}")
    return float(value)


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def upsert(rows: list[dict[str, str]], indexed: dict[tuple[str, ...], int], row: dict[str, str]) -> None:
    row_key = key(row)
    if row_key in indexed:
        existing = rows[indexed[row_key]]
        if (
            existing["experiment_id"] == "denovo_kingdoms_species_cap100k_200peak_prediction_20260923"
            and row["experiment_id"] == "denovo_kingdoms_species_cap100k_wandb_20260915"
        ):
            return
        rows[indexed[row_key]].update(row)
    else:
        indexed[row_key] = len(rows)
        rows.append(row)


def selected_run(api: wandb.Api, project: str, checkpoint_basename: str):
    matches = [
        run for run in api.runs(project)
        if run.state == "finished" and Path(str(run.config.get("downstream_weights", ""))).name == checkpoint_basename
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one finished {project} run for {checkpoint_basename}, found {len(matches)}.")
    return matches[0]


def aggregate_epoch_metrics(run) -> dict[str, float]:
    records = list(run.scan_history(keys=["_step", *AGGREGATE], page_size=10_000))
    candidates = [record for record in records if all(isinstance(record.get(name), (int, float)) and math.isfinite(float(record[name])) for name in AGGREGATE)]
    if len(candidates) != 1:
        raise ValueError(f"{run.path} has {len(candidates)} complete aggregate test-epoch records, expected one.")
    return {metric: finite(candidates[0][metric]) for metric in AGGREGATE}


def species_epoch_metrics(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    records: dict[str, dict[str, float]] = {}
    prefix = "denovo_tf_test_species_"
    for metric_name, value in summary.items():
        if not metric_name.startswith(prefix):
            continue
        if metric_name.endswith("_step"):
            raise ValueError(f"Per-species step metric must not be imported: {metric_name}")
        suffix = next((item for item in SPECIES_SUFFIXES if metric_name.endswith(f"_{item}")), None)
        if suffix is None:
            continue
        species = metric_name[len(prefix):-(len(suffix) + 1)]
        records.setdefault(species, {})[suffix] = finite(value)
    if not records:
        raise ValueError("No per-species test metrics found in W&B summary.")
    required = set(SPECIES_SUFFIXES)
    malformed = {species: sorted(required - set(metrics)) for species, metrics in records.items() if set(metrics) != required}
    if malformed:
        raise ValueError(f"Incomplete per-species metric records: {malformed!r}")
    return dict(sorted(records.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument("--sidecar", type=Path, default=Path("results/denovo/kingdoms_fast_eval/wandb_epoch_metrics.json"))
    args = parser.parse_args()
    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    indexed = {key(row): i for i, row in enumerate(rows)}
    api = wandb.Api()
    archived: dict[str, Any] = {"benchmark": "Kingdoms run-disjoint deterministic 100k-per-species cap", "runs": []}

    for project, checkpoint_basename, model_id in RUNS:
        run = selected_run(api, project, checkpoint_basename)
        aggregate = aggregate_epoch_metrics(run)
        species = species_epoch_metrics(dict(run.summary))
        sidecar_record = {
            "project": project, "run_id": run.id, "run_name": run.name,
            "checkpoint_basename": checkpoint_basename, "aggregate_epoch_metrics": aggregate,
            "per_species_on_test_epoch": species,
        }
        archived["runs"].append(sidecar_record)
        notes = (
            "Deterministic 100k-per-species Kingdoms fast-triage test; aggregate test epoch metrics "
            "from W&B. Full per-species on-test-epoch metrics are archived in the linked JSON sidecar."
        )
        for source_metric, ledger_metric in AGGREGATE.items():
            upsert(rows, indexed, {
                "experiment_id": "denovo_kingdoms_species_cap100k_wandb_20260915",
                "task": "denovo", "cohort": "kingdoms_run_disjoint_v1_species_cap100k",
                "split": "test", "corpus": "kingdoms_run_disjoint_cap100k", "model_id": model_id,
                "conditioning": "conditioned", "representation": "full_finetune",
                "metric_name": ledger_metric, "value": repr(aggregate[source_metric]), "status": "complete",
                "priority": "primary" if ledger_metric == "denovo/peptide_precision" else "secondary",
                "selection_role": "selected", "higher_is_better": "true",
                "report_path": str(args.sidecar), "notes": notes,
            })

    for model_id, note in PENDING_1000:
        for _, ledger_metric in AGGREGATE.items():
            pending_key = (
                "denovo", "kingdoms_run_disjoint_v1_species_cap100k", "test",
                "kingdoms_run_disjoint_cap100k", model_id,
                "not_applicable" if model_id in RELEASED_DENOVO_BASELINES else "conditioned",
                "released_denovo" if model_id in RELEASED_DENOVO_BASELINES else "full_finetune",
                ledger_metric,
            )
            if pending_key in indexed and rows[indexed[pending_key]]["status"] == "complete":
                continue
            upsert(rows, indexed, {
                "experiment_id": "denovo_kingdoms_species_cap100k_pending_1000peaks_20260915",
                "task": "denovo", "cohort": "kingdoms_run_disjoint_v1_species_cap100k",
                "split": "test", "corpus": "kingdoms_run_disjoint_cap100k", "model_id": model_id,
                "conditioning": "not_applicable" if model_id in RELEASED_DENOVO_BASELINES else "conditioned",
                "representation": "released_denovo" if model_id in RELEASED_DENOVO_BASELINES else "full_finetune",
                "metric_name": ledger_metric, "value": "", "status": "pending",
                "priority": "primary" if ledger_metric == "denovo/peptide_precision" else "secondary",
                "selection_role": "leakage_concern" if model_id == "instanovo_v1_2_2" else "selected", "higher_is_better": "true", "report_path": "",
                "notes": note + (" Leakage concern: overlap with the published InstaNovo training data has not yet been ruled out." if model_id == "instanovo_v1_2_2" else "") + " Deterministic 100k-per-species fast-triage test cohort.",
            })

    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps(archived, indent=2, sort_keys=True) + "\n")
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {len(archived['runs'])} aggregate Kingdoms runs and archived {sum(len(r['per_species_on_test_epoch']) for r in archived['runs'])} per-species records without printing metrics.")


if __name__ == "__main__":
    main()
