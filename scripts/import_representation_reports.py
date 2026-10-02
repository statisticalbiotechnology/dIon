"""Import completed registry-driven representation reports into the paper ledger.

The importer is idempotent: a completed report replaces the matching pending or
prior row identified by task/cohort/split/corpus/model/conditioning/readout/metric.
It intentionally ignores incomplete array cells, so it is safe while an array
is still running.
"""
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
    "retrieval/broad_map": ("primary", "development", "map"),
    "retrieval/same_charge_10ppm_map": ("secondary", "diagnostic", "mass_controlled_map"),
    "pair/same_charge_10ppm/roc_auc": ("primary", "development", "roc_auc"),
    "pair/same_charge_10ppm/average_precision": ("primary", "development", "average_precision"),
}


def report_metrics(path: Path, kind: str) -> list[tuple[str, float]]:
    report = json.loads(path.read_text())
    if "metrics_by_distance" in report:
        source = report["metrics_by_distance"]["cosine"]
        source = source["macro"] if kind == "retrieval" else source["pair_sets"]["same_charge_10ppm"]["macro"]
    else:
        source = report["macro"] if kind == "retrieval" else report["pair_sets"]["same_charge_10ppm"]["macro"]
    prefix = "retrieval/" if kind == "retrieval" else "pair/same_charge_10ppm/"
    pairs = []
    for metric_name, (_, _, source_key) in METRICS.items():
        if not metric_name.startswith(prefix):
            continue
        value = source.get(source_key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            pairs.append((metric_name, float(value)))
    return pairs


def representation(metadata: dict) -> str:
    kind = metadata["model"]["kind"]
    if kind == "metric_learning_checkpoint":
        return "metric_head"
    return str(metadata["model"].get("embedding_readout", "backbone"))


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument(
        "--cohort-override",
        help="Paper-facing cohort name to use instead of the registry metadata cohort.",
    )
    parser.add_argument(
        "--model-id-suffix",
        default="",
        help=(
            "Append a stable inference-variant suffix to imported model IDs. "
            "Use this when the same checkpoint is evaluated with materially "
            "different preprocessing, such as --max-peaks-override 0."
        ),
    )
    args = parser.parse_args()

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {key(row): index for index, row in enumerate(rows)}
    imported = 0
    skipped = 0
    for metadata_path in sorted(args.results_root.glob("*/*/*/run_metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "completed":
            skipped += 1
            continue
        retrieval_path = Path(metadata["reports"]["retrieval"])
        pairs_path = Path(metadata["reports"]["pairs"])
        if not retrieval_path.exists() or not pairs_path.exists():
            skipped += 1
            continue
        cohort = args.cohort_override or str(metadata["cohort"])
        common = {
            "experiment_id": f"representation_{cohort}_{metadata['split']}_{metadata_path.parents[3].name}",
            "task": "representation",
            "cohort": cohort,
            "split": str(metadata["split"]),
            "corpus": str(metadata["corpus"]),
            "model_id": f"{metadata['model_id']}{args.model_id_suffix}",
            "conditioning": str(metadata["conditioning"]),
            "representation": representation(metadata),
            "status": "complete",
            "higher_is_better": "true",
            "notes": "Registry-driven dIon representation evaluation; exact matched external cohort and preprocessing recorded in run_metadata.json.",
        }
        for kind, report_path in (("retrieval", retrieval_path), ("pairs", pairs_path)):
            for metric_name, value in report_metrics(report_path, kind):
                priority, selection_role, _ = METRICS[metric_name]
                row = dict(common, metric_name=metric_name, value=repr(value), priority=priority,
                           selection_role=selection_role, report_path=str(report_path))
                row_key = key(row)
                if row_key in by_key:
                    existing = rows[by_key[row_key]]
                    # Selection status is a human model-selection decision, not a
                    # property of a newly generated report. Preserve explicit
                    # decisions when refreshing the corresponding metric value.
                    if existing["selection_role"] in {
                        "not_selected",
                        "provisional_selected",
                        "locked_test",
                    }:
                        row["selection_role"] = existing["selection_role"]
                    existing.update(row)
                else:
                    by_key[row_key] = len(rows)
                    rows.append(row)
                imported += 1

    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {imported} metrics; skipped {skipped} incomplete cells: {args.ledger}")


if __name__ == "__main__":
    main()
