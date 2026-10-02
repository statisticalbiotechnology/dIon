"""Standalone input-reliance evaluation for a dIon embedding checkpoint."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.embed_eval.evaluator import evaluate_embedder, resolve_evaluation_config
from src.embed_eval.loading import load_checkpoint_embedder
from src.embed_eval.pair_evaluator import evaluate_pair_embedder
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


def _summary(reports: list[dict], getter) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for key, value in getter(report).items():
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                values[key].append(float(value))
    return {
        key: {
            "mean": sum(samples) / len(samples),
            "std": (sum((value - sum(samples) / len(samples)) ** 2 for value in samples) / len(samples)) ** 0.5,
            "n": len(samples),
        }
        for key, samples in sorted(values.items())
    }


def main() -> None:
    global_args, pretrain_config, _, probing_config = parse_args_and_config()
    key = "embedding_intervention_evaluation"
    if not probing_config or key not in probing_config:
        raise ValueError(f"--probing_config must contain {key!r}.")
    if global_args.embedding_baseline != "model":
        raise ValueError("Input interventions require --embedding_baseline model.")
    suite = probing_config[key]
    retrieval_config = _load_section(suite["retrieval_config_path"], "embedding_evaluation")
    pair_config = _load_section(suite["pair_config_path"], "pair_discrimination_evaluation")
    _deep_update(retrieval_config, suite.get("retrieval_overrides", {}))
    _deep_update(pair_config, suite.get("pair_overrides", {}))
    _, embedder = load_checkpoint_embedder(global_args, pretrain_config)
    device = torch.device("cuda" if global_args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    base_seed = int(suite.get("seed", 0))
    results = {}
    for intervention in suite["interventions"]:
        name = str(intervention["name"])
        repeats = int(intervention.get("repeats", 1))
        if repeats < 1:
            raise ValueError("Each intervention requires repeats >= 1.")
        retrieval_reports = []
        pair_reports = []
        for repeat in range(repeats):
            seed = base_seed + repeat
            retrieval_reports.append(evaluate_embedder(
                embedder, global_args, retrieval_config, mode="standalone", device=device,
                input_intervention_name=name, intervention_seed=seed,
            ))
            pair_reports.append(evaluate_pair_embedder(
                embedder, global_args, pair_config, mode="standalone", device=device,
                input_intervention_name=name, intervention_seed=seed,
            ))
        results[name] = {
            "retrieval_reports": retrieval_reports,
            "pair_reports": pair_reports,
            "retrieval_macro": _summary(retrieval_reports, lambda report: report["macro"]),
            "pair_macro": {
                pair_set: _summary(
                    pair_reports,
                    lambda report, pair_set=pair_set: report["pair_sets"][pair_set]["macro"],
                )
                for pair_set in pair_reports[0]["pair_sets"]
            },
        }
    report = {
        "checkpoint_path": global_args.encoder_weights,
        "embedding_readout": global_args.embedding_readout,
        "suite": suite,
        "interventions": results,
    }
    configured = suite.get("output", {}).get("report_path")
    output_path = Path(configured) if configured else Path(global_args.encoder_weights).with_suffix(".input_interventions.json")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if suite.get("output", {}).get("print_report", True):
        print(json.dumps({name: {"retrieval_macro": value["retrieval_macro"], "pair_macro": value["pair_macro"]} for name, value in results.items()}, indent=2, sort_keys=True))
    print(f"Wrote input-intervention report: {output_path}")


if __name__ == "__main__":
    main()
