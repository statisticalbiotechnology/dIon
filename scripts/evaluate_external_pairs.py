"""Evaluate an external embedding NPZ with dIon's pair-discrimination metrics."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq
import torch
import yaml

from src.embed_eval.external import load_external_embedding_cache
from src.embed_eval.pair_data import load_manifest
from src.embed_eval.pair_evaluator import _availability_by_set_species
from src.embed_eval.pair_metrics import evaluate_pair_discrimination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument("--pairs-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=["cosine", "euclidean"])
    args = parser.parse_args()

    raw_config = yaml.safe_load(args.evaluation_config.read_text())
    if not isinstance(raw_config, dict) or "pair_discrimination_evaluation" not in raw_config:
        raise ValueError("--evaluation-config must contain pair_discrimination_evaluation.")
    base_metrics = raw_config["pair_discrimination_evaluation"].get("pair_metrics", {})
    cache = load_external_embedding_cache(
        args.embeddings_npz,
        peptide_id_key="peptide_ion_id",
        partition_id_key="species",
        spectrum_id_key="spectrum_id",
    )
    pairs = pq.read_table(args.pairs_path).to_pylist()
    manifest = load_manifest(args.manifest_path)
    availability = _availability_by_set_species(manifest)
    reports = {}
    for metric in args.metrics:
        metric_config = copy.deepcopy(base_metrics)
        metric_config["metric"] = metric
        reports[metric] = evaluate_pair_discrimination(
            cache.values,
            cache.spectrum_ids,
            pairs,
            metric_config,
            availability_by_set_species=availability,
            device=torch.device("cpu"),
        )
    report = {
        "name": raw_config["pair_discrimination_evaluation"]["name"],
        "embedding_source": "external",
        "external_artifact": str(args.embeddings_npz.resolve()),
        "embedding_dimension": int(cache.values.shape[1]),
        "spectra": int(cache.values.shape[0]),
        "pairs_path": str(args.pairs_path.resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "metrics_by_distance": reports,
        "notes": [
            "All metrics use pairs whose two spectra passed external preprocessing.",
            "Cosine is the common DINO-vs-GLEAMS metric; Euclidean is GLEAMS' native embedding distance.",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote external pair-discrimination report: {args.output_report}")


if __name__ == "__main__":
    main()
