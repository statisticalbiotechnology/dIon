"""Import completed staged locked-test results into the paper ledger without printing values.

Incomplete representation cells are intentionally skipped: a result is imported only
when both retrieval and pair reports are present. Re-running is idempotent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

FIELDS = [
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path", "notes",
]
REP_METRICS = {
    "retrieval/broad_map": ("primary", "map"),
    "retrieval/same_charge_10ppm_map": ("secondary", "mass_controlled_map"),
    "pair/same_charge_10ppm/roc_auc": ("primary", "roc_auc"),
    "pair/same_charge_10ppm/average_precision": ("primary", "average_precision"),
}
DOWNSTREAM_METRICS = {
    "sqa": (("test_auc", "classification/roc_auc", True, "primary"),),
    "chimericity": (("test_auc", "classification/roc_auc", True, "primary"),),
    "oxidized_met": (
        ("test_auc", "classification/roc_auc", True, "primary"),
        ("test_matched_backbone_auc", "classification/matched_backbone_roc_auc", True, "secondary"),
    ),
    "retention_time": (
        ("retention_time_test_mae", "regression/mae", False, "primary"),
        ("retention_time_test_r2", "regression/r2", True, "secondary"),
        ("retention_time_test_spearman", "regression/spearman", True, "secondary"),
        ("retention_time_test_pearson", "regression/pearson", True, "diagnostic"),
        ("retention_time_test_delta_t95", "regression/delta_t95", False, "sensitivity"),
    ),
}
TASK_META = {
    "sqa": ("casanovo19pxd_balanced_charge1_10", "casanovo19pxd"),
    "chimericity": ("PXD024584_HYE_V1", "PXD024584_HYE"),
    "oxidized_met": ("PXD010613_oxidized_met_V1", "PXD010613"),
    "retention_time": ("run_aligned_retention_time_V1", "run_aligned_multispecies"),
}


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "task", "cohort", "split", "corpus", "model_id", "conditioning",
        "representation", "metric_name",
    ))


def metric_source(report: dict, kind: str) -> dict:
    if "metrics_by_distance" in report:
        cosine = report["metrics_by_distance"]["cosine"]
        return cosine["macro"] if kind == "retrieval" else cosine["pair_sets"]["same_charge_10ppm"]["macro"]
    return report["macro"] if kind == "retrieval" else report["pair_sets"]["same_charge_10ppm"]["macro"]


def corpus_from_path(path: Path) -> str | None:
    text = str(path).lower()
    for corpus in ("ninespecies_v2", "bacterial", "kingdoms"):
        if corpus in text:
            return corpus
    return None


def external_identity(path: Path) -> tuple[str, str, str, str] | None:
    parts = set(path.parts)
    text = str(path)
    charge_matched = "heldout_test_charge2to4" in text
    if "gleams_v03" in parts:
        return "charge2to4", "gleams_v03", "not_applicable", "gleams"
    if "casanovo_v4" in parts:
        if charge_matched:
            return "charge2to4", "casanovo_v4_peak_mean", "not_applicable", "casanovo_peak_mean"
        return "full_charge", "casanovo_v4", "not_applicable", "casanovo_v4_peak_mean"
    if "instanovo_fm" in parts:
        return ("charge2to4" if charge_matched else "full_charge",
                "instanovo_fm", "not_applicable", "instanovo_fm")
    if "binned_spectrum" in parts:
        return ("charge2to4", "binned_spectral_angle", "not_applicable", "binned_spectrum") if charge_matched else (
            "full_charge", "binned_spectrum", "not_applicable", "binned_spectrum")
    return None


def upsert(rows: list[dict[str, str]], indexed: dict[tuple[str, ...], int], row: dict[str, str]) -> None:
    row_key = key(row)
    if row_key in indexed:
        rows[indexed[row_key]].update(row)
    else:
        indexed[row_key] = len(rows)
        rows.append(row)


def import_external(staging: Path, rows: list[dict[str, str]], indexed: dict[tuple[str, ...], int]) -> Counter:
    counts: Counter = Counter()
    seen: set[Path] = set()
    for retrieval in sorted(staging.rglob("*_retrieval.json")):
        text = str(retrieval)
        if "locked_test" not in text and "heldout_test" not in text:
            continue
        identity = external_identity(retrieval)
        corpus = corpus_from_path(retrieval)
        pairs = retrieval.with_name(retrieval.name.replace("_retrieval.json", "_pairs.json"))
        if identity is None or corpus is None or not pairs.is_file() or retrieval in seen:
            continue
        seen.add(retrieval)
        cohort, model_id, conditioning, representation = identity
        for kind, report_path in (("retrieval", retrieval), ("pairs", pairs)):
            source = metric_source(json.loads(report_path.read_text()), kind)
            prefix = "retrieval/" if kind == "retrieval" else "pair/same_charge_10ppm/"
            for metric_name, (priority, source_key) in REP_METRICS.items():
                if not metric_name.startswith(prefix):
                    continue
                value = source.get(source_key)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    raise ValueError(f"Missing {source_key} in {report_path}")
                upsert(rows, indexed, {
                    "experiment_id": f"locked_test_{cohort}_{model_id}", "task": "representation",
                    "cohort": cohort, "split": "test", "corpus": corpus,
                    "model_id": model_id, "conditioning": conditioning, "representation": representation,
                    "metric_name": metric_name, "value": repr(float(value)), "status": "complete",
                    "priority": priority, "selection_role": "locked_test", "higher_is_better": "true",
                    "report_path": str(report_path),
                    "notes": "Locked held-out representation evaluation; hidden in dashboard by default.",
                })
                counts["external_representation_metrics"] += 1
    return counts


def import_dion_representation(staging: Path, rows: list[dict[str, str]], indexed: dict[tuple[str, ...], int]) -> Counter:
    counts: Counter = Counter()
    metadata_paths = []
    for pattern in (
        "locked_test_dion_controls_representation_*/*/*/*/run_metadata.json",
        "locked_test_metric_learning_label_efficiency_*/*/*/*/run_metadata.json",
        "locked_test_representation_*/*/*/*/run_metadata.json",
    ):
        metadata_paths.extend(staging.glob(pattern))
    for metadata_path in sorted(metadata_paths):
        metadata = json.loads(metadata_path.read_text())
        reports = metadata.get("reports", {})
        retrieval, pairs = Path(reports.get("retrieval", "")), Path(reports.get("pairs", ""))
        if metadata.get("status") != "completed" or not retrieval.is_file() or not pairs.is_file():
            continue
        representation = "metric_head" if metadata["model"]["kind"] == "metric_learning_checkpoint" else metadata["model"].get("embedding_readout", "backbone")
        is_label_efficiency = metadata_path.parents[3].name.startswith(
            "locked_test_metric_learning_label_efficiency_"
        )
        notes = (
            "Prespecified 1%/10% SupCon label-efficiency held-out evaluation; "
            "each checkpoint was selected by its fixed validation-loss monitor."
            if is_label_efficiency
            else "Frozen validation-selected dIon control; locked held-out evaluation."
        )
        for kind, report_path in (("retrieval", retrieval), ("pairs", pairs)):
            source = metric_source(json.loads(report_path.read_text()), kind)
            prefix = "retrieval/" if kind == "retrieval" else "pair/same_charge_10ppm/"
            for metric_name, (priority, source_key) in REP_METRICS.items():
                if not metric_name.startswith(prefix):
                    continue
                value = source.get(source_key)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    raise ValueError(f"Missing {source_key} in {report_path}")
                upsert(rows, indexed, {
                    "experiment_id": f"locked_test_{('full_charge' if metadata['cohort'] == 'primary' else metadata['cohort'])}_{metadata['model_id']}", "task": "representation",
                    "cohort": "full_charge" if metadata["cohort"] == "primary" else str(metadata["cohort"]), "split": "test", "corpus": str(metadata["corpus"]),
                    "model_id": str(metadata["model_id"]), "conditioning": str(metadata["conditioning"]),
                    "representation": str(representation), "metric_name": metric_name,
                    "value": repr(float(value)), "status": "complete", "priority": priority,
                    "selection_role": "locked_test", "higher_is_better": "true", "report_path": str(report_path),
                    "notes": notes,
                })
                counts["dion_representation_metrics"] += 1
    return counts


def auxiliary_selection_role(model_id: str, conditioning: str) -> str:
    """Return the prespecified paper-selection role for auxiliary-task rows."""
    selected_controls = {
        "binned1024_precursor_metadata", "binned1024_conditioned_frozen",
        "casanovo_v4_peak_mean", "instanovo_fm", "scratch",
        "random_conditioned_finetuned", "random_conditioned_frozen",
    }
    if model_id == "hybrid_300_last":
        return "selected" if conditioning == "conditioned" else "not_selected"
    if model_id in selected_controls:
        return "selected"
    return "not_selected"


def model_identity(label: str, task: str) -> tuple[str, str, str]:
    if label.startswith("casanovo_v4_"):
        model_id, conditioning = "casanovo_v4_peak_mean", "not_applicable"
    elif label.startswith("instanovo_fm_"):
        model_id, conditioning = "instanovo_fm", "not_applicable"
    elif label.startswith("hybrid300last_"):
        model_id = "hybrid_300_last"
        conditioning = "null" if "_null_" in label else "conditioned"
    else:
        model_id = "scratch" if label.startswith("scratch_") or label.startswith("random_") else (
            "binned1024_spectrum_only" if "spectrum_only" in label else "binned1024_precursor_metadata")
        conditioning = "not_used" if "spectrum_only" in label else "conditioned"
    if task == "retention_time":
        return model_id, conditioning, "linear_ordinal_frozen" if "ordinal" in label else "linear_regression_frozen"
    if task == "sqa":
        return model_id, conditioning, "sqa_frozen" if label.endswith("_frozen") else "sqa_finetuned"
    return model_id, conditioning, "binary_frozen" if label.endswith("_frozen") else "binary_finetuned"


def import_downstream(staging: Path, selection: Path, root_glob: str, experiment_id: str, note: str, rows: list[dict[str, str]], indexed: dict[tuple[str, ...], int]) -> Counter:
    counts: Counter = Counter()
    selections = json.loads(selection.read_text())["downstream"]
    for root in sorted(staging.glob(root_glob)):
        if (root / "INVALIDATED.txt").is_file():
            continue
        for summary_path in sorted(root.glob("*/**/wandb-summary.json")):
            run_dir = summary_path
            while run_dir.parent != root:
                run_dir = run_dir.parent
            index = int(run_dir.name.rsplit("_", 1)[1])
            item = selections[index]
            task = item["task"]
            if item["model_label"] not in run_dir.name:
                raise ValueError(f"Selection mismatch: {run_dir}")
            summary = json.loads(summary_path.read_text())
            model_id, conditioning, representation = model_identity(item["model_label"], task)
            cohort, corpus = TASK_META[task]
            obsolete = task == "retention_time"
            notes = note + (
                " Obsolete task retained for audit; excluded from paper completion and reporting scope."
                if obsolete else ""
            )
            for source_key, metric_name, higher_is_better, priority in DOWNSTREAM_METRICS[task]:
                value = summary.get(source_key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"Missing finite {source_key} in {summary_path}")
                upsert(rows, indexed, {
                    "experiment_id": experiment_id, "task": task, "cohort": cohort,
                    "split": "test", "corpus": corpus, "model_id": model_id,
                    "conditioning": conditioning, "representation": representation,
                    "metric_name": metric_name, "value": repr(float(value)),
                    "status": "obsolete" if obsolete else "complete",
                    "priority": priority,
                    "selection_role": "not_selected" if obsolete else auxiliary_selection_role(model_id, conditioning),
                    "higher_is_better": str(higher_is_better).lower(), "report_path": str(summary_path),
                    "notes": notes,
                })
                counts["downstream_metrics"] += 1
    return counts


def import_sqa(source: Path, rows: list[dict[str, str]], indexed: dict[tuple[str, ...], int]) -> Counter:
    counts: Counter = Counter()
    with source.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    for item in source_rows:
        if item.get("split_id") != "test" or item.get("metric_name") != "test_auc":
            continue
        representation = "sqa_frozen" if item["freeze_encoder"].lower() == "true" else "sqa_finetuned"
        upsert(rows, indexed, {
            "experiment_id": item["experiment_id"], "task": "sqa", "cohort": item["dataset_id"],
            "split": "test", "corpus": "casanovo19pxd", "model_id": item["model_id"],
            "conditioning": item["precursor_conditioning"], "representation": representation,
            "metric_name": "classification/roc_auc", "value": item["metric_value"], "status": "complete",
            "priority": "primary", "selection_role": auxiliary_selection_role(item["model_id"], item["precursor_conditioning"]), "higher_is_better": "true",
            "report_path": str(source.parent / item["report_path"]),
            "notes": "Canonical completed SQA held-out evaluation; validation-selected checkpoint.",
        })
        counts["sqa_metrics"] += 1
    if not counts:
        raise ValueError(f"No SQA test_auc rows found in {source}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=Path("/path/to/results"))
    parser.add_argument("--selection", type=Path, default=Path("configs/evaluation/locked_test_dion_controls.json"))
    parser.add_argument("--external-downstream-selection", type=Path, default=Path("configs/evaluation/locked_test_external_downstream.json"))
    parser.add_argument("--hybrid300last-downstream-selection", type=Path, default=Path("configs/evaluation/locked_test_hybrid300last_downstream.json"))
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument(
        "--sqa-source", type=Path,
        default=Path("results/sqa/casanovo_19pxd_auc_selected__20260908T160857Z/metrics.csv"),
    )
    args = parser.parse_args()
    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != FIELDS:
        raise ValueError(f"Unexpected ledger schema: {args.ledger}")
    # Earlier imports used a temporary split label. Canonicalize before
    # constructing upsert keys so re-import remains idempotent.
    for row in rows:
        if row["split"] == "locked_test":
            row["split"] = "test"
    indexed = {key(row): i for i, row in enumerate(rows)}
    counts = Counter()
    counts.update(import_external(args.staging_root, rows, indexed))
    counts.update(import_dion_representation(args.staging_root, rows, indexed))
    counts.update(import_downstream(
        args.staging_root, args.selection, "locked_test_dion_controls_downstream_*",
        "locked_test_dion_controls", "Frozen validation-selected dIon control; locked held-out evaluation.", rows, indexed,
    ))
    if args.external_downstream_selection.is_file():
        counts.update(import_downstream(
            args.staging_root, args.external_downstream_selection, "locked_test_external_downstream*",
            "locked_test_external_downstream", "Frozen validation-selected external embedding head; locked held-out evaluation.", rows, indexed,
        ))
    if args.hybrid300last_downstream_selection.is_file():
        counts.update(import_downstream(
            args.staging_root, args.hybrid300last_downstream_selection, "locked_test_downstream_hybrid300cachefixed_*",
            "locked_test_hybrid300last_downstream", "Hybrid-300-last checkpoint selected by validation monitor; locked held-out evaluation.", rows, indexed,
        ))
    counts.update(import_sqa(args.sqa_source, rows, indexed))
    with args.ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Imported locked-test metrics without printing values: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
