"""Configuration-driven peptide embedding evaluation orchestration."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from src.embed_eval.baselines import binned_spectrum_details, similarity_metadata
from src.embed_eval.data import (
    PeptideRetrievalDataset,
    StreamingPeptideRetrievalDataset,
    build_embedding_eval_collate,
)
from src.embed_eval.extraction import (
    extract_embeddings,
    extract_precursor_metadata_embeddings,
)
from src.embed_eval.interventions import InputIntervention
from src.embed_eval.peptide_metrics import evaluate_peptide_embeddings


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_evaluation_config(config: dict, mode: str) -> dict:
    """Apply standalone or online overrides without mutating parsed YAML."""
    if mode not in {"standalone", "online"}:
        raise ValueError(f"Unsupported embedding evaluation mode: {mode!r}")
    resolved = copy.deepcopy(config)
    overrides = resolved.get(mode, {}).get("overrides", {})
    _deep_update(resolved, overrides)
    return resolved


def extract_embedding_cache(
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
    """Resolve one retrieval configuration and extract its aligned cache."""
    config = resolve_evaluation_config(config, mode)
    dataset_cfg = config["dataset"]
    selection_cfg = config.get("selection", {})
    extraction_cfg = config.get("extraction", {})
    dataset_class = (
        StreamingPeptideRetrievalDataset
        if extraction_cfg.get("stream_parquet", True)
        and input_intervention_name == "none"
        else PeptideRetrievalDataset
    )
    dataset = dataset_class(
        dataset_cfg["parquet_path"],
        peptide_id_column=dataset_cfg.get("peptide_id_column", "peptide_id"),
        partition_column=dataset_cfg.get("partition_column", "species"),
        seed=int(selection_cfg.get("seed", 0)),
        max_peptides_per_partition=selection_cfg.get(
            "max_peptides_per_partition"
        ),
        max_spectra_per_peptide=selection_cfg.get("max_spectra_per_peptide"),
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
            collate_fn=build_embedding_eval_collate(global_args),
            batch_size=int(extraction_cfg.get("batch_size", 256)),
            num_workers=int(extraction_cfg.get("num_workers", 0)),
            device=device,
            pin_memory=bool(extraction_cfg.get("pin_memory", False))
            and device.type == "cuda",
            show_progress=bool(extraction_cfg.get("show_progress", False)),
            input_intervention=input_intervention,
            precursor_conditioning=(
                getattr(global_args, "precursor_conditioning", "conditioned")
                if precursor_conditioning is None
                else precursor_conditioning
            ),
        )
    return config, dataset, cache


def evaluate_embedder(
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
    """Extract and evaluate model or precursor-metadata embeddings."""
    config, dataset, cache = extract_embedding_cache(
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
    metrics_device = device if config.get("metrics_device", "model") == "model" else torch.device("cpu")
    report = evaluate_peptide_embeddings(
        cache.values,
        cache.peptide_ids,
        cache.partition_ids,
        config,
        device=metrics_device,
        precursor_mz=cache.precursor_mz,
        precursor_charges=cache.precursor_charges,
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
        }
    )
    return report


def flatten_report_for_logging(
    report: dict[str, object], *, macro_only: bool = False
) -> dict[str, float]:
    """Flatten one report into W&B-safe scalar metric names."""
    name = str(report["name"])
    metrics = {}
    if not macro_only:
        for partition, values in report["partitions"].items():
            for metric_name, value in values.items():
                if isinstance(value, float) and math.isfinite(value):
                    metrics[f"embed_eval/{name}/{partition}/{metric_name}"] = value
    online_metric_names = {"map"}
    for metric_name, value in report["macro"].items():
        if macro_only and metric_name not in online_metric_names:
            continue
        if isinstance(value, float) and math.isfinite(value):
            logged_name = "broad_map" if macro_only and metric_name == "map" else metric_name
            metrics[f"embed_eval/{name}/{logged_name}"] = value
    return metrics


def write_report(report: dict[str, object], config: dict, checkpoint_path: str | None) -> Path | None:
    """Write exactly one JSON report, unless output is configured as print-only."""
    output_cfg = config.get("output", {})
    if not output_cfg.get("write_report", True):
        return None
    configured_path = output_cfg.get("report_path")
    if configured_path:
        output_path = Path(configured_path)
    elif checkpoint_path:
        checkpoint = Path(checkpoint_path)
        report_tag = output_cfg.get("report_tag")
        if report_tag:
            output_path = checkpoint.with_name(
                f"{checkpoint.stem}.{report_tag}.embedding_eval.json"
            )
        else:
            output_path = checkpoint.with_suffix(".embedding_eval.json")
    else:
        source = str(report.get("embedding_source", "model"))
        output_path = Path(f"{report['name']}_{source}.embedding_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output_path
