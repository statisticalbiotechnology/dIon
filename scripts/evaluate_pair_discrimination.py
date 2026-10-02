"""Standalone peptide-ion pair-discrimination evaluation for a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.embed_eval.evaluator import resolve_evaluation_config
from src.embed_eval.loading import (
    load_checkpoint_embedder,
    load_metric_learning_checkpoint_embedder,
    load_peak_only_binned_spectrum_embedder,
    load_random_checkpoint_embedder,
)
from src.embed_eval.pair_evaluator import evaluate_pair_embedder, write_pair_report
from src.parse_args import parse_args_and_config


def _parse_script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--random_encoder",
        action="store_true",
        help="Evaluate a seeded architecture-matched random DINO encoder.",
    )
    parser.add_argument(
        "--random_encoder_seed",
        type=int,
        default=None,
        help="Initialization seed for --random_encoder; defaults to --seed.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    script_args = _parse_script_args()
    global_args, pretrain_config, downstream_config, probing_config = parse_args_and_config()
    key = "pair_discrimination_evaluation"
    if not probing_config or key not in probing_config:
        raise ValueError(
            "--probing_config must contain a 'pair_discrimination_evaluation' section."
        )
    embedder = None
    if global_args.embedding_baseline == "model":
        if script_args.random_encoder:
            if global_args.encoder_weights or global_args.downstream_weights:
                raise ValueError(
                    "--random_encoder cannot be combined with checkpoint weights."
                )
            random_seed = (
                global_args.seed
                if script_args.random_encoder_seed is None
                else script_args.random_encoder_seed
            )
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
            _, embedder = load_random_checkpoint_embedder(global_args, pretrain_config)
        elif global_args.downstream_task == "metric_learning":
            if not downstream_config or "metric_learning" not in downstream_config:
                raise ValueError("Metric-learning evaluation requires its downstream config.")
            embedder = load_metric_learning_checkpoint_embedder(
                global_args, pretrain_config, downstream_config["metric_learning"]
            )
        else:
            _, embedder = load_checkpoint_embedder(global_args, pretrain_config)
    elif global_args.embedding_baseline == "binned_spectrum":
        _, embedder = load_peak_only_binned_spectrum_embedder(global_args)
    device = torch.device(
        "cuda"
        if global_args.accelerator == "gpu" and torch.cuda.is_available()
        else "cpu"
    )
    config = probing_config[key]
    standalone_config = resolve_evaluation_config(config, "standalone")
    report = evaluate_pair_embedder(
        embedder,
        global_args,
        config,
        mode="standalone",
        device=device,
    )
    report["embedding_readout"] = global_args.embedding_readout
    if (
        global_args.embedding_baseline == "binned_spectrum"
        and not standalone_config.get("output", {}).get("report_path")
    ):
        standalone_config.setdefault("output", {})["report_path"] = str(
            Path(global_args.output_dir)
            / f"{report['name']}_binned_spectrum.pair_eval.json"
        )
    checkpoint_path = (
        None
        if script_args.random_encoder
        else (
            global_args.downstream_weights
            if global_args.downstream_task == "metric_learning"
            else global_args.encoder_weights
        )
    )
    output_path = write_pair_report(
        report,
        standalone_config,
        checkpoint_path if embedder is not None else None,
    )
    if standalone_config.get("output", {}).get("print_report", True):
        print(json.dumps(report, indent=2, sort_keys=True))
    if output_path is not None:
        print(f"Wrote pair-discrimination report: {output_path}")


if __name__ == "__main__":
    main()
