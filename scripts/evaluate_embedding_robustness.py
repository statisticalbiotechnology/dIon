"""Evaluate perturbed spectrum queries against clean embedding galleries."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import yaml

from src.embed_eval.evaluator import extract_embedding_cache
from src.embed_eval.loading import load_checkpoint_embedder
from src.embed_eval.pair_evaluator import (
    _availability_by_set_species,
    extract_pair_embedding_cache,
)
from src.embed_eval.pair_data import load_manifest
from src.embed_eval.pair_metrics import evaluate_pair_discrimination
from src.embed_eval.peptide_metrics import evaluate_clean_gallery_robustness
from src.parse_args import parse_args_and_config


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively apply suite-local diagnostic subset overrides."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_section(path: str, section: str) -> dict:
    config = yaml.safe_load(Path(path).read_text()) or {}
    if section not in config:
        raise ValueError(f"{path} must contain {section!r}.")
    return config[section]


def _stable_permutation(size: int, seed: int, partition: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        int.from_bytes(
            hashlib.sha256(f"{seed}\x1f{partition}".encode()).digest()[:8], "big"
        ) % (2**31)
    )
    permutation = torch.randperm(size, generator=generator)
    if size > 1 and torch.equal(permutation, torch.arange(size)):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def _alignment_metrics(query_values, clean_values, partition_ids, seed: int) -> dict[str, float]:
    """Compare same-spectrum cross-view distance with random same-species distance."""
    query_values = F.normalize(query_values.float(), dim=1)
    clean_values = F.normalize(clean_values.float(), dim=1)
    same = 1.0 - (query_values * clean_values).sum(dim=1)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, partition in enumerate(partition_ids):
        groups[str(partition)].append(index)
    random_distances = []
    for partition, indices in groups.items():
        if len(indices) < 2:
            continue
        local = torch.tensor(indices, dtype=torch.long)
        randomized = local[_stable_permutation(len(indices), seed, partition)]
        random_distances.append(1.0 - (query_values[local] * clean_values[randomized]).sum(dim=1))
    random = torch.cat(random_distances)
    same_mean = float(same.mean().item())
    random_mean = float(random.mean().item())
    return {
        "same_spectrum_cosine_distance": same_mean,
        "random_same_species_cosine_distance": random_mean,
        "same_to_random_distance_ratio": same_mean / max(random_mean, 1e-12),
    }


def _macro_summary(reports: list[dict], key: str) -> dict[str, dict[str, float]]:
    samples: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for name, value in report[key].items():
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                samples[name].append(float(value))
    return {
        name: {
            "mean": sum(values) / len(values),
            "std": (sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)) ** 0.5,
            "n": len(values),
        }
        for name, values in sorted(samples.items())
    }


def _assert_aligned(clean_cache, query_cache, name: str) -> None:
    if (
        clean_cache.peptide_ids != query_cache.peptide_ids
        or clean_cache.partition_ids != query_cache.partition_ids
        or clean_cache.spectrum_ids != query_cache.spectrum_ids
    ):
        raise RuntimeError(f"{name} extraction order changed between clean and perturbed views.")


def main() -> None:
    global_args, pretrain_config, _, probing_config = parse_args_and_config()
    suite_key = "embedding_robustness_evaluation"
    if not probing_config or suite_key not in probing_config:
        raise ValueError(f"--probing_config must contain {suite_key!r}.")
    if global_args.embedding_baseline != "model":
        raise ValueError("Clean-gallery robustness requires model embeddings.")
    suite = probing_config[suite_key]
    retrieval_config = _load_section(suite["retrieval_config_path"], "embedding_evaluation")
    pair_config = _load_section(suite["pair_config_path"], "pair_discrimination_evaluation")
    _deep_update(retrieval_config, suite.get("retrieval_overrides", {}))
    _deep_update(pair_config, suite.get("pair_overrides", {}))
    _, embedder = load_checkpoint_embedder(global_args, pretrain_config)
    device = torch.device("cuda" if global_args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    retrieval_config, _, clean_retrieval = extract_embedding_cache(
        embedder, global_args, retrieval_config, mode="standalone", device=device
    )
    pair_config, pair_dataset, clean_pair = extract_pair_embedding_cache(
        embedder, global_args, pair_config, mode="standalone", device=device
    )
    manifest = load_manifest(pair_config["dataset"].get("manifest_path"))
    availability = _availability_by_set_species(manifest)
    pair_records = pair_dataset.pairs.to_pylist()
    metrics_device = device if pair_config.get("metrics_device", "model") == "model" else torch.device("cpu")
    base_seed = int(suite.get("seed", 0))
    results = {}
    for entry in suite["interventions"]:
        name = str(entry["name"])
        repeats = int(entry.get("repeats", 1))
        retrieval_reports = []
        pair_reports = []
        for repeat in range(repeats):
            seed = base_seed + repeat
            _, _, query_retrieval = extract_embedding_cache(
                embedder, global_args, retrieval_config, mode="standalone", device=device,
                input_intervention_name=name, intervention_seed=seed,
            )
            _, _, query_pair = extract_pair_embedding_cache(
                embedder, global_args, pair_config, mode="standalone", device=device,
                input_intervention_name=name, intervention_seed=seed,
            )
            _assert_aligned(clean_retrieval, query_retrieval, "retrieval")
            _assert_aligned(clean_pair, query_pair, "pair")
            retrieval = evaluate_clean_gallery_robustness(
                query_retrieval.values,
                clean_retrieval.values,
                clean_retrieval.peptide_ids,
                clean_retrieval.partition_ids,
                clean_retrieval.precursor_mz,
                clean_retrieval.precursor_charges,
                retrieval_config,
                device=metrics_device,
            )
            retrieval["alignment"] = _alignment_metrics(
                query_retrieval.values, clean_retrieval.values,
                clean_retrieval.partition_ids, seed,
            )
            pair = evaluate_pair_discrimination(
                query_pair.values,
                clean_pair.spectrum_ids,
                pair_records,
                pair_config.get("pair_metrics", {}),
                availability_by_set_species=availability,
                device=metrics_device,
                comparison_values=clean_pair.values,
            )
            retrieval_reports.append(retrieval)
            pair_reports.append(pair)
        results[name] = {
            "retrieval_reports": retrieval_reports,
            "pair_reports": pair_reports,
            "retrieval_macro": _macro_summary(retrieval_reports, "macro"),
            "alignment": _macro_summary(retrieval_reports, "alignment"),
            "pair_macro": {
                pair_set: _macro_summary(
                    [{"macro": report["pair_sets"][pair_set]["macro"]} for report in pair_reports],
                    "macro",
                )
                for pair_set in pair_reports[0]["pair_sets"]
            },
        }
    report = {
        "checkpoint_path": global_args.encoder_weights,
        "embedding_readout": global_args.embedding_readout,
        "protocol": "perturbed_query_vs_clean_gallery_bidirectional_pairs",
        "suite": suite,
        "interventions": results,
    }
    configured = suite.get("output", {}).get("report_path")
    output_path = (
        Path(configured)
        if configured
        else Path(global_args.encoder_weights).with_suffix(".clean_gallery_robustness.json")
    )
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if suite.get("output", {}).get("print_report", True):
        print(json.dumps({
            name: {
                "retrieval_macro": result["retrieval_macro"],
                "alignment": result["alignment"],
                "pair_macro": result["pair_macro"],
            }
            for name, result in results.items()
        }, indent=2, sort_keys=True))
    print(f"Wrote clean-gallery robustness report: {output_path}")


if __name__ == "__main__":
    main()
