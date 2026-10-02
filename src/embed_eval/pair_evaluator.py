"""Configuration-driven peptide-ion pair-discrimination evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from src.embed_eval.baselines import binned_spectrum_details, similarity_metadata
from src.embed_eval.evaluator import resolve_evaluation_config
from src.embed_eval.extraction import (
    extract_embeddings,
    extract_precursor_metadata_embeddings,
)
from src.embed_eval.interventions import InputIntervention
from src.embed_eval.pair_data import (
    PairBenchmarkDataset,
    StreamingPairBenchmarkDataset,
    build_pair_eval_collate,
    load_manifest,
)
from src.embed_eval.pair_metrics import evaluate_pair_discrimination


def _availability_by_set_species(
    manifest: dict[str, object] | None,
) -> dict[str, dict[str, dict[str, object]]]:
    """Extract protocol availability recorded by the static benchmark builder."""
    if not manifest:
        return {}
    pair_sets = manifest.get("pair_sets", {})
    if not isinstance(pair_sets, dict):
        return {}
    return {
        str(pair_set): {
            str(species): dict(values)
            for species, values in by_species.items()
            if isinstance(values, dict)
        }
        for pair_set, by_species in pair_sets.items()
        if isinstance(by_species, dict)
    }


def extract_pair_embedding_cache(
    embedder: torch.nn.Module | None,
    global_args,
    config: dict,
    *,
    mode: str,
    device: torch.device,
    input_intervention_name: str = "none",
    intervention_seed: int = 0,
    precursor_conditioning: str | None = None,
):
    """Resolve one static-pair configuration and extract its aligned cache."""
    config = resolve_evaluation_config(config, mode)
    dataset_cfg = config["dataset"]
    selection_cfg = config.get("selection", {})
    extraction_cfg = config.get("extraction", {})
    dataset_class = (
        StreamingPairBenchmarkDataset
        if extraction_cfg.get("stream_parquet", True)
        and input_intervention_name == "none"
        else PairBenchmarkDataset
    )
    dataset = dataset_class(
        dataset_cfg["spectra_path"],
        dataset_cfg["pairs_path"],
        seed=int(selection_cfg.get("seed", 0)),
        max_pairs_per_partition_per_label=selection_cfg.get(
            "max_pairs_per_partition_per_label"
        ),
    )
    embedding_baseline = getattr(global_args, "embedding_baseline", "model")
    if embedding_baseline == "precursor_metadata":
        if input_intervention_name != "none":
            raise ValueError("Input interventions require model embeddings, not metadata-only embeddings.")
        cache = extract_precursor_metadata_embeddings(dataset)
    else:
        if embedder is None:
            raise ValueError("A model embedder is required when no baseline is selected.")
        input_intervention = (
            None
            if input_intervention_name == "none"
            else InputIntervention(input_intervention_name, dataset, intervention_seed)
        )
        cache = extract_embeddings(
            embedder,
            dataset,
            collate_fn=build_pair_eval_collate(global_args),
            batch_size=int(extraction_cfg.get("batch_size", 256)),
            num_workers=int(extraction_cfg.get("num_workers", 0)),
            device=device,
            pin_memory=bool(extraction_cfg.get("pin_memory", False))
            and device.type == "cuda",
            show_progress=bool(extraction_cfg.get("show_progress", False)),
            progress_desc="Embedding pair benchmark",
            input_intervention=input_intervention,
            precursor_conditioning=(
                getattr(global_args, "precursor_conditioning", "conditioned")
                if precursor_conditioning is None
                else precursor_conditioning
            ),
        )
    return config, dataset, cache


def evaluate_pair_embedder(
    embedder: torch.nn.Module | None,
    global_args,
    config: dict,
    *,
    mode: str,
    device: torch.device,
    input_intervention_name: str = "none",
    intervention_seed: int = 0,
    precursor_conditioning: str | None = None,
) -> dict[str, object]:
    """Extract model or precursor-metadata embeddings for static pair protocols."""
    config, dataset, cache = extract_pair_embedding_cache(
        embedder,
        global_args,
        config,
        mode=mode,
        device=device,
        input_intervention_name=input_intervention_name,
        intervention_seed=intervention_seed,
        precursor_conditioning=precursor_conditioning,
    )
    embedding_baseline = getattr(global_args, "embedding_baseline", "model")
    manifest = load_manifest(config["dataset"].get("manifest_path"))
    metrics_device = (
        device
        if config.get("metrics_device", "model") == "model"
        else torch.device("cpu")
    )
    report = evaluate_pair_discrimination(
        cache.values,
        cache.spectrum_ids,
        dataset.pairs.to_pylist(),
        config.get("pair_metrics", {}),
        availability_by_set_species=_availability_by_set_species(manifest),
        device=metrics_device,
    )
    report.update(
        {
            "name": config["name"],
            "mode": mode,
            "selection": asdict(dataset.selection),
            "embedding_dimension": int(cache.values.shape[1]),
            "embedding_source": embedding_baseline,
            "precursor_conditioning": (
                getattr(global_args, "precursor_conditioning", "conditioned")
                if precursor_conditioning is None
                else precursor_conditioning
            ),
            "input_intervention": input_intervention_name,
            "intervention_seed": int(intervention_seed),
            "embedding_source_details": (
                {
                    "features": ["log_precursor_mass", "precursor_charge"],
                    "normalization": "population_zscore_over_selected_evaluation_spectra",
                    "precursor_mass": "observed_precursor_mz_times_charge",
                }
                if embedding_baseline == "precursor_metadata"
                else (
                    binned_spectrum_details(global_args)
                    if embedding_baseline == "binned_spectrum"
                    else None
                )
            ),
            "similarity": (
                similarity_metadata(
                    str(config.get("retrieval", {}).get("metric", "cosine"))
                    if "retrieval" in config
                    else str(config.get("pair_metrics", {}).get("metric", "cosine"))
                )
                if embedding_baseline == "binned_spectrum"
                else None
            ),
            "manifest": manifest,
        }
    )
    return report


def flatten_pair_report_for_logging(
    report: dict[str, object], *, macro_only: bool = False
) -> dict[str, float]:
    """Flatten concise pair-quality scalars for online experiment logging."""
    name = str(report["name"])
    metrics = {}
    metric_names = (
        {"roc_auc", "average_precision"}
        if macro_only
        else {
            "roc_auc",
            "average_precision",
            "fnr_at_balanced_fdr_0p01",
            "fnr_at_balanced_fdr_0p05",
            "fnr_at_balanced_fdr_0p10",
            "hard_negative_anchor_coverage",
        }
    )
    pair_sets = report.get("pair_sets", {})
    for pair_set, result in pair_sets.items():
        if macro_only and pair_set != "same_charge_10ppm":
            continue
        scopes = ("macro",) if macro_only else ("partitions", "macro", "pooled")
        for scope in scopes:
            scoped = result.get(scope, {})
            if scope == "partitions":
                items = scoped.items()
            else:
                items = [(scope, scoped)]
            for partition, values in items:
                for metric_name in metric_names:
                    value = values.get(metric_name)
                    if isinstance(value, (float, int)) and math.isfinite(float(value)):
                        metrics[
                            f"pair_eval/{name}/{pair_set}/{metric_name}"
                        ] = float(value)
    return metrics


def write_pair_report(
    report: dict[str, object], config: dict, checkpoint_path: str | None
) -> Path | None:
    """Write one standalone pair-evaluation JSON report when enabled."""
    output_cfg = config.get("output", {})
    if not output_cfg.get("write_report", True):
        return None
    configured_path = output_cfg.get("report_path")
    if configured_path:
        output_path = Path(configured_path)
    elif checkpoint_path:
        readout = str(report.get("embedding_readout", "backbone"))
        report_tag = output_cfg.get("report_tag")
        suffix = ".pair_eval.json" if readout == "backbone" else f".pair_eval_{readout}.json"
        if report_tag:
            suffix = f".{report_tag}{suffix}"
        output_path = Path(checkpoint_path).with_suffix(suffix)
    else:
        source = str(report.get("embedding_source", "model"))
        output_path = Path(f"{report['name']}_{source}.pair_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output_path
