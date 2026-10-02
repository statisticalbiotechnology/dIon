#!/usr/bin/env python3
"""Import validated MSKB-final charge-<5 de novo predictions into the paper ledger.

The transferred package is the authority: each completed run must carry exactly
one prediction per spectrum in the fixed 196,979-row held-out subset. Values are
read from its validated summary JSONs, whose matching rule is Casanovo-compatible
``aa_match`` under the paper tokenizer. This script never reads W&B metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_ROWS = 196_979
FIELDS = (
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
)
METRICS = (
    ("peptide_precision_100pct_coverage", "denovo/peptide_precision", "primary"),
    ("aa_precision", "denovo/aa_precision", "secondary"),
    ("aa_recall", "denovo/aa_recall", "secondary"),
)
OPTIONAL_METRICS = (
    ("precision_coverage_auc", "denovo/precision_coverage_auc", "diagnostic"),
)

# Every transferred result is precursor-conditioned. Priority reflects the
# prespecified main-comparison trio; valid frozen/lower-priority runs remain
# visible for sensitivity or appendix reporting.
IDENTITY = {
    "mskb_hybrid300_finetuned": ("mskb_hybrid300_finetuned", "full_finetune", "primary"),
    "mskb_scratch_encoderld": ("mskb_scratch_encoderld", "full_finetune", "primary"),
    "v5_hybrid300_finetuned": ("v5_hybrid300_finetuned", "full_finetune", "primary"),
    "mskb_local60_gram_refined_frozen": ("mskb_local60_gram_refined_frozen", "frozen_encoder", "secondary"),
    "mskb_hybrid300_frozen": ("mskb_hybrid300_frozen", "frozen_encoder", "secondary"),
    "v5_scratch_encoderld": ("v5_scratch_encoderld", "full_finetune", "secondary"),
    "v5_local60_gram_refined_frozen": ("v5_local60_gram_refined_frozen", "frozen_encoder", "secondary"),
    "v5_hybrid300_frozen_best": ("v5_hybrid300_frozen_best", "frozen_encoder", "secondary"),
    "v5_hybrid300_frozen_late": ("v5_hybrid300_frozen_late", "frozen_encoder", "secondary"),
    "v5_local60_gram_refined_finetuned": ("v5_local60_gram_refined_finetuned", "full_finetune", "secondary"),
    # Transfer labels omit ``finetuned``; retain the canonical ledger identities
    # chosen before these replacement inference jobs completed.
    "mskb_hybrid300_maxpeaks1000": ("mskb_hybrid300_finetuned_maxpeaks1000", "full_finetune", "primary"),
    "v5_hybrid300_maxpeaks1000": ("v5_hybrid300_finetuned_maxpeaks1000", "full_finetune", "primary"),
}
SELECTED_MODELS = {
    "mskb_hybrid300_finetuned",
    "mskb_hybrid300_finetuned_maxpeaks1000",
    "mskb_scratch_encoderld",
    "v5_hybrid300_finetuned",
    "v5_hybrid300_finetuned_maxpeaks1000",
    "v5_scratch_encoderld",
}

PENDING = {
    "mskb_scratch_encoderld_maxpeaks1000": "Upstream scratch-1000 training job 2465043_0 pending; no valid held-out prediction exists.",
    "v5_scratch_encoderld_maxpeaks1000": "Upstream scratch-1000 training job 2465044_0 pending; no valid held-out prediction exists.",
    "casanovo_v5_2_1": "Released Casanovo v5.2.1 inference is pending on the MSKB-final charge-<5 held-out subset.",
}


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def upsert(rows: list[dict[str, str]], index: dict[tuple[str, ...], int], row: dict[str, str]) -> None:
    key = row_key(row)
    if key in index:
        rows[index[key]].update(row)
    else:
        index[key] = len(rows)
        rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root", type=Path,
        default=Path("/path/to/results/denovo_eval/denovo_mskb_final_charge_lt5"),
    )
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    args = parser.parse_args()

    package = args.package_root
    source = json.loads((package / "validated_results_ledger.json").read_text())
    if source.get("expected_psm_rows") != EXPECTED_ROWS:
        raise ValueError(f"Unexpected package row count: {source.get('expected_psm_rows')!r}")
    runs = source.get("runs")
    if not isinstance(runs, list) or len(runs) != len(IDENTITY):
        raise ValueError("Expected exactly the ten validated MSKB result entries.")

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Unexpected paper ledger schema.")
    index = {row_key(row): i for i, row in enumerate(rows)}
    imported: set[str] = set()

    for run in runs:
        label = run.get("label")
        if label not in IDENTITY or run.get("status") != "validated_complete":
            raise ValueError(f"Unexpected validated run: {label!r}")
        if run.get("psm_rows") != EXPECTED_ROWS or run.get("unique_spectra") != EXPECTED_ROWS:
            raise ValueError(f"{label} does not cover exactly {EXPECTED_ROWS} spectra.")
        summary_path = package / "summaries" / f"{label}.json"
        mztab_path = package / "mztab" / f"{label}.mzTab"
        if not summary_path.is_file() or not mztab_path.is_file():
            raise ValueError(f"{label} is missing transferred summary or mzTab.")
        summary = json.loads(summary_path.read_text())
        if summary.get("psm_rows") != EXPECTED_ROWS or summary.get("unique_spectra") != EXPECTED_ROWS:
            raise ValueError(f"{label} summary does not cover exactly {EXPECTED_ROWS} spectra.")
        if Path(str(run.get("validated_mztab"))).name != mztab_path.name:
            raise ValueError(f"{label} validated mzTab name disagrees with package.")
        model_id, representation, run_priority = IDENTITY[label]
        notes = (
            f"Validated Casanovo-compatible prediction-mode result on the fixed {EXPECTED_ROWS:,}-spectrum "
            "MSKB-final charge-<5 held-out subset; summary derived from the transferred full mzTab."
        )
        for summary_key, metric_name, metric_priority in METRICS:
            value = summary.get(summary_key)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{label} has no numeric {summary_key}.")
            upsert(rows, index, {
                "experiment_id": "denovo_mskb_final_charge_lt5_validated_20260915",
                "task": "denovo", "cohort": "mskb_final_charge_lt5_for_casanovo_v_gt_5_0",
                "split": "test", "corpus": "mskb_final_charge_lt5", "model_id": model_id,
                "conditioning": "conditioned", "representation": representation,
                "metric_name": metric_name, "value": repr(float(value)), "status": "complete",
                "priority": "primary" if run_priority == "primary" and metric_priority == "primary" else metric_priority,
                "selection_role": "selected" if model_id in SELECTED_MODELS else "not_selected", "higher_is_better": "true",
                "report_path": str(summary_path), "notes": notes,
            })
        for summary_key, metric_name, metric_priority in OPTIONAL_METRICS:
            value = summary.get(summary_key)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise ValueError(f"{label} has non-numeric optional {summary_key}.")
            upsert(rows, index, {
                "experiment_id": "denovo_mskb_final_charge_lt5_validated_20260915",
                "task": "denovo", "cohort": "mskb_final_charge_lt5_for_casanovo_v_gt_5_0",
                "split": "test", "corpus": "mskb_final_charge_lt5", "model_id": model_id,
                "conditioning": "conditioned", "representation": representation,
                "metric_name": metric_name, "value": repr(float(value)), "status": "complete",
                "priority": metric_priority,
                "selection_role": "selected" if model_id in SELECTED_MODELS else "not_selected", "higher_is_better": "true",
                "report_path": str(summary_path), "notes": notes,
            })
        imported.add(label)

    if imported != set(IDENTITY):
        raise ValueError(f"Imported labels differ from expected set: {sorted(imported)!r}")

    for model_id, note in PENDING.items():
        for _, metric_name, priority in METRICS:
            upsert(rows, index, {
                "experiment_id": "denovo_mskb_final_charge_lt5_pending_1000peaks_20260915",
                "task": "denovo", "cohort": "mskb_final_charge_lt5_for_casanovo_v_gt_5_0",
                "split": "test", "corpus": "mskb_final_charge_lt5", "model_id": model_id,
                "conditioning": "not_applicable" if model_id == "casanovo_v5_2_1" else "conditioned",
                "representation": "released_denovo" if model_id == "casanovo_v5_2_1" else "full_finetune",
                "metric_name": metric_name, "value": "", "status": "pending", "priority": priority,
                "selection_role": "selected" if model_id in SELECTED_MODELS or model_id == "casanovo_v5_2_1" else "not_selected", "higher_is_better": "true", "report_path": "", "notes": note,
            })

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {len(imported)} validated MSKB runs and retained {len(PENDING)} pending entries without printing metrics.")


if __name__ == "__main__":
    main()
