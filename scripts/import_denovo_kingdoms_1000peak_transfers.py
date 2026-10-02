#!/usr/bin/env python3
"""Import transferred Kingdoms 1000-peak fast-eval epoch summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path("/path/to/results/denovo_eval/denovo_kingdoms_species_cap100k/1000_peaks")
SIDECAR = Path("results/denovo/kingdoms_fast_eval/transferred_1000peak_epoch_metrics.json")
LEDGER = Path("results/paper/metrics_long.csv")
RUNS = (
    ("hybrid300_maxpeaks1000_2482474_0", "gxi0mh21", "mskb_hybrid300_finetuned_maxpeaks1000", "denovo_mskb_final", "hybrid300", "epoch=77-denovo_tf_val_pep_prec=0.82.ckpt", "selected"),
    ("hybrid300_maxpeaks1000_2482475_0", "66wrb6it", "v5_hybrid300_finetuned_maxpeaks1000", "denovo_dnlv1", "hybrid300", "epoch=27-denovo_tf_val_pep_prec=0.62.ckpt", "selected"),
    ("scratch_encoderld_maxpeaks1000_2520002_3", "9yx87oca", "mskb_scratch_encoderld_maxpeaks1000", "denovo_mskb_final", "scratch_encoderld", "epoch=79-denovo_tf_val_pep_prec=0.76.ckpt", "not_selected"),
    ("scratch_encoderld_maxpeaks1000_2520003_3", "ogp914w0", "v5_scratch_encoderld_maxpeaks1000", "denovo_dnlv1", "scratch_encoderld", "epoch=26-denovo_tf_val_pep_prec=0.59.ckpt", "not_selected"),
)
METRICS = {
    "denovo_tf_test_pep_prec_epoch": "denovo/peptide_precision",
    "denovo_tf_test_aa_prec_epoch": "denovo/aa_precision",
    "denovo_tf_test_aa_recall_epoch": "denovo/aa_recall",
}
SUFFIXES = ("n_spectra", "pep_prec", "aa_prec", "aa_recall")
KEY_FIELDS = ("task", "cohort", "split", "corpus", "model_id", "conditioning", "representation", "metric_name")


def finite(value: object) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid epoch metric: {value!r}")
    return float(value)


def read_run(root: Path, spec: tuple[str, ...]) -> dict:
    directory, run_id, model_id, training, initialization, checkpoint, _ = spec
    matches = list((root / directory).glob(f"logs_*/wandb/run-*-{run_id}/files/wandb-summary.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one transferred summary for {run_id}; found {len(matches)}")
    summary_path = matches[0]
    config_path = summary_path.with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    weights = str(config["downstream_weights"]["value"])
    if (Path(weights).name != checkpoint or training not in weights or initialization not in weights
            or config["max_peaks"]["value"] != 1000):
        raise ValueError(f"Unexpected checkpoint or peak configuration for {model_id}")
    summary = json.loads(summary_path.read_text())
    aggregate = {name: finite(summary[name]) for name in METRICS}
    species: dict[str, dict[str, float]] = {}
    prefix = "denovo_tf_test_species_"
    for name, value in summary.items():
        if not name.startswith(prefix) or name.endswith("_step"):
            continue
        stem = name[len(prefix):]
        suffix = next((suffix for suffix in SUFFIXES if stem.endswith("_" + suffix)), None)
        if suffix:
            species_name = stem[:-(len(suffix) + 1)]
            species.setdefault(species_name, {})[suffix] = finite(value)
    if len(species) != 70 or any(set(values) != set(SUFFIXES) for values in species.values()):
        raise ValueError(f"Incomplete species metrics for {model_id}")
    count = sum(values["n_spectra"] for values in species.values())
    if count != 4_926_240:
        raise ValueError(f"Unexpected logged species count for {model_id}: {count}")
    return {
        "model_id": model_id, "run_id": run_id, "checkpoint_basename": checkpoint,
        "summary_path": str(summary_path),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "aggregate_epoch_metrics": aggregate,
        "per_species_on_test_epoch": dict(sorted(species.items())),
        "logged_species_n_spectra_sum": int(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()
    runs = [read_run(args.source_root, spec) for spec in RUNS]
    with args.ledger.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or len(rows) == 0:
        raise ValueError("Empty or malformed results ledger")
    indexed = {tuple(row[field] for field in KEY_FIELDS): index for index, row in enumerate(rows)}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate ledger keys")
    for spec, run in zip(RUNS, runs):
        model_id, selection = spec[2], spec[6]
        for source_metric, metric in METRICS.items():
            row = {
                "experiment_id": "denovo_kingdoms_species_cap100k_transferred_1000peaks_20260921",
                "task": "denovo", "cohort": "kingdoms_run_disjoint_v1_species_cap100k",
                "split": "test", "corpus": "kingdoms_run_disjoint_cap100k",
                "model_id": model_id, "conditioning": "conditioned",
                "representation": "full_finetune", "metric_name": metric,
                "value": repr(run["aggregate_epoch_metrics"][source_metric]),
                "status": "complete", "priority": "primary" if metric == "denovo/peptide_precision" else "secondary",
                "selection_role": selection, "higher_is_better": "true",
                "report_path": str(args.sidecar),
                "notes": "Transferred 1000-peak Kingdoms autoregressive beam-decoded test metric from W&B; teacher forcing is used for loss only. Per-spectrum prediction tables were not retained. Per-species summary metrics are archived in the sidecar. DDP sampler padding repeats the first eight Akkermansia records: logged species counts sum to 4,926,240, while the canonical denominator is 4,926,232. Scores are preserved as logged.",
            }
            key = tuple(row[field] for field in KEY_FIELDS)
            if key in indexed:
                if rows[indexed[key]]["status"] not in ("pending", "complete"):
                    raise ValueError(f"Unexpected existing ledger status: {key}")
                rows[indexed[key]] = row
            else:
                indexed[key] = len(rows)
                rows.append(row)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "benchmark": "Kingdoms run-disjoint deterministic 100k-per-species cap",
        "evaluation_mode": "autoregressive_decoding_without_prediction_table",
        "canonical_test_rows": 4_926_232,
        "logged_species_count_note": "DDP sampler padding repeats the first eight test records, all Akkermansia_muciniphila: 27,286 logged versus 27,278 canonical. Each run logs 4,926,240 species n_spectra; use 4,926,232 as the canonical denominator. Scores and counts are preserved as logged.",
        "runs": runs,
    }, indent=2, sort_keys=True) + "\n")
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {len(runs)} runs, {len(runs) * 70} per-species records and 12 ledger metrics")


if __name__ == "__main__":
    main()
