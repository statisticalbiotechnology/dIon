"""Initialize the paper ledger with current representation validation rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = [
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
]
METRICS = {
    "retrieval/broad_map": ("primary", "development"),
    "retrieval/same_charge_10ppm_map": ("secondary", "diagnostic"),
    "pair/same_charge_10ppm/roc_auc": ("primary", "development"),
    "pair/same_charge_10ppm/average_precision": ("primary", "development"),
}
CORPORA = ("ninespecies_v2", "bacterial", "kingdoms")
PENDING_MODELS = (
    ("casanovo_v4_peak_mean", "not_applicable", "casanovo_peak_mean"),
    ("random_hybrid", "conditioned", "backbone"),
    ("random_hybrid", "null", "backbone"),
    ("hybrid_100_last", "conditioned", "backbone"),
    ("hybrid_100_last", "null", "backbone"),
    ("hybrid_300_last", "conditioned", "backbone"),
    ("hybrid_300_last", "null", "backbone"),
    ("metric_scratch_10pct_val_loss", "conditioned", "metric_head"),
    ("metric_hybrid_100_frozen_10pct_val_loss", "conditioned", "metric_head"),
    ("metric_hybrid_300_frozen_10pct_val_loss", "conditioned", "metric_head"),
    ("metric_hybrid_100_10pct_val_loss", "conditioned", "metric_head"),
    ("metric_hybrid_300_10pct_val_loss", "conditioned", "metric_head"),
)


def _metric_source(report: dict, report_kind: str) -> dict:
    if "metrics_by_distance" in report:
        cosine = report["metrics_by_distance"]["cosine"]
        if report_kind == "retrieval":
            return cosine["macro"]
        return cosine["pair_sets"]["same_charge_10ppm"]["macro"]
    if report_kind == "retrieval":
        return report["macro"]
    return report["pair_sets"]["same_charge_10ppm"]["macro"]


def _completed_rows(
    path: Path,
    corpus: str,
    model_id: str,
    representation: str,
    notes: str,
) -> list[dict[str, str]]:
    report = json.loads(path.read_text())
    kind = "pairs" if "pairs" in path.name else "retrieval"
    source = _metric_source(report, kind)
    requested = (
        (("retrieval/broad_map", "map"), ("retrieval/same_charge_10ppm_map", "mass_controlled_map"))
        if kind == "retrieval"
        else (("pair/same_charge_10ppm/roc_auc", "roc_auc"), ("pair/same_charge_10ppm/average_precision", "average_precision"))
    )
    rows = []
    for metric_name, source_key in requested:
        value = source.get(source_key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        priority, selection_role = METRICS[metric_name]
        rows.append({
            "experiment_id": "representation_charge2to4_validation_20260909",
            "task": "representation",
            "cohort": "charge2to4",
            "split": "validation",
            "corpus": corpus,
            "model_id": model_id,
            "conditioning": "not_applicable",
            "representation": representation,
            "metric_name": metric_name,
            "value": repr(float(value)),
            "status": "complete",
            "priority": priority,
            "selection_role": selection_role,
            "higher_is_better": "true",
            "report_path": str(path),
            "notes": notes,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --force to rebuild this seed ledger.")

    root = Path("results/representation")
    gleams = {
        "ninespecies_v2": root / "gleams_v03_charge2to4_validation__20260909T000000Z/ninespecies_v2",
        "bacterial": root / "gleams_v03_charge2to4_validation__20260909T000000Z/bacterial",
        "kingdoms": root / "external/gleams_v03/kingdoms_validation_charge2to4",
    }
    binned = {
        corpus: root / f"external/binned_spectrum/{corpus}_validation_charge2to4"
        for corpus in CORPORA
    }
    rows: list[dict[str, str]] = []
    for corpus in CORPORA:
        rows += _completed_rows(
            gleams[corpus] / "gleams_retrieval.json", corpus, "gleams_v03", "gleams",
            "Released GLEAMS v0.3 reference on the canonical charge-2-4 cohort.",
        )
        rows += _completed_rows(
            gleams[corpus] / "gleams_pairs.json", corpus, "gleams_v03", "gleams",
            "Released GLEAMS v0.3 reference on the canonical charge-2-4 cohort.",
        )
        rows += _completed_rows(
            binned[corpus] / "binned_spectrum_retrieval.json", corpus, "binned_spectral_angle", "binned_spectrum",
            "Exploratory validation reference; binned runs used max_peaks=200. Final test baseline must be uncapped.",
        )
        rows += _completed_rows(
            binned[corpus] / "binned_spectrum_pairs.json", corpus, "binned_spectral_angle", "binned_spectrum",
            "Exploratory validation reference; binned runs used max_peaks=200. Final test baseline must be uncapped.",
        )
    for model_id, conditioning, representation in PENDING_MODELS:
        for corpus in CORPORA:
            for metric_name, (priority, selection_role) in METRICS.items():
                rows.append({
                    "experiment_id": "representation_charge2to4_validation_20260909",
                    "task": "representation",
                    "cohort": "charge2to4",
                    "split": "validation",
                    "corpus": corpus,
                    "model_id": model_id,
                    "conditioning": conditioning,
                    "representation": representation,
                    "metric_name": metric_name,
                    "value": "",
                    "status": "pending",
                    "priority": priority,
                    "selection_role": selection_role,
                    "higher_is_better": "true",
                    "report_path": "",
                    "notes": "Expected validation comparison cell.",
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} validation ledger rows: {args.output}")


if __name__ == "__main__":
    main()
