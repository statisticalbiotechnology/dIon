"""Standalone cached dense de novo probe evaluation for a dIon checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.callbacks.dense_denovo_probe_callback import DenseDeNovoProbeCallback
from src.embed_eval.loading import (
    load_checkpoint_encoder,
    load_downstream_encoder,
    load_random_encoder,
)
from src.parse_args import parse_args_and_config
from src.probe_conditioning import get_probe_conditioning_modes


def _parse_script_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--random_encoder",
        action="store_true",
        help="Evaluate a seeded, untrained DINO encoder instead of loading --encoder_weights.",
    )
    parser.add_argument(
        "--random_encoder_seed",
        type=int,
        default=None,
        help="Initialization seed for --random_encoder; defaults to --seed.",
    )
    parser.add_argument(
        "--downstream_encoder_weights",
        type=str,
        default=None,
        help="Supervised downstream checkpoint from which to load only encoder.* weights.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    script_args = _parse_script_args()
    global_args, pretrain_config, _, probing_config = parse_args_and_config()
    if not probing_config or "dense_denovo_probe" not in probing_config:
        raise ValueError("--probing_config must contain a 'dense_denovo_probe' section.")
    conditioning_modes = get_probe_conditioning_modes(
        probing_config,
        "dense_denovo_probe",
        global_args.precursor_conditioning,
    )
    if len(conditioning_modes) != 1:
        raise ValueError(
            "Standalone dense evaluation writes one report per invocation; "
            "configure exactly one dense_denovo_probe "
            "online_precursor_conditioning_modes value."
        )
    selected_sources = int(script_args.random_encoder) + int(bool(script_args.downstream_encoder_weights)) + int(bool(global_args.encoder_weights))
    if selected_sources != 1:
        raise ValueError(
            "Provide exactly one encoder source: --encoder_weights, --random_encoder, "
            "or --downstream_encoder_weights."
        )
    if script_args.random_encoder:
        random_seed = global_args.seed if script_args.random_encoder_seed is None else script_args.random_encoder_seed
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        wrapper, encoder = load_random_encoder(global_args, pretrain_config)
        encoder_source = f"random_seed_{random_seed}"
    elif script_args.downstream_encoder_weights:
        wrapper = None
        encoder = load_downstream_encoder(
            global_args, pretrain_config, script_args.downstream_encoder_weights
        )
        encoder_source = str(Path(script_args.downstream_encoder_weights).resolve())
    else:
        wrapper, encoder = load_checkpoint_encoder(global_args, pretrain_config)
        encoder_source = str(Path(global_args.encoder_weights).resolve())
    device = torch.device(
        "cuda" if global_args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    callback = DenseDeNovoProbeCallback(
        probing_config,
        global_args,
        embedder_batch_size=getattr(wrapper, "batch_size", global_args.batch_size),
        precursor_conditioning=conditioning_modes[0],
    )
    metrics = callback.evaluate_encoder(encoder.to(device), device, show_progress=True)
    report = {
        "checkpoint_path": encoder_source if not script_args.random_encoder else None,
        "encoder_source": encoder_source,
        "encoder_source_type": (
            "random" if script_args.random_encoder else
            "downstream_checkpoint" if script_args.downstream_encoder_weights else
            "pretraining_checkpoint"
        ),
        "encoder_precursor_conditioning": callback.encoder_precursor_conditioning,
        "cache_dtype": callback.cache_dtype_name,
        "metrics": metrics,
    }
    output_cfg = callback.cfg.get("output", {})
    if output_cfg.get("write_report", True):
        output_path = Path(
            output_cfg.get("report_path")
            or (
                Path("dense_denovo_random_encoder_probe.json")
                if script_args.random_encoder
                else (
                    Path(script_args.downstream_encoder_weights).with_suffix(".dense_denovo_probe.json")
                    if script_args.downstream_encoder_weights
                    else Path(global_args.encoder_weights).with_suffix(".dense_denovo_probe.json")
                )
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote dense de novo probe report: {output_path}")
    if output_cfg.get("print_report", True):
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
