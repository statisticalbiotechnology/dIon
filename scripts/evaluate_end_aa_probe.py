"""Standalone end-amino-acid probe evaluation for a dIon checkpoint."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Permit direct ``python scripts/evaluate_end_aa_probe.py`` execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.callbacks.linprobe_callback import EndAAProbeCallback
from src.embed_eval.loading import load_checkpoint_embedder
from src.parse_args import parse_args_and_config


def _seed_everything(seed: int) -> None:
    """Make repeated standalone probe comparisons use the same linear-head seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_report(report: dict, output_config: dict, checkpoint_path: str) -> Path | None:
    """Write one JSON report unless standalone report output is disabled."""
    if not output_config.get("write_report", True):
        return None
    configured_path = output_config.get("report_path")
    output_path = (
        Path(configured_path)
        if configured_path
        else Path(checkpoint_path).with_suffix(
            f".end_aa_probe_{report['mass_input']}.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    global_args, pretrain_config, _, probing_config = parse_args_and_config()
    if not probing_config or "end_aa_pred" not in probing_config:
        raise ValueError("--probing_config must contain an 'end_aa_pred' section.")

    _seed_everything(int(global_args.seed))
    wrapper, embedder = load_checkpoint_embedder(global_args, pretrain_config)
    device = torch.device(
        "cuda"
        if global_args.accelerator == "gpu" and torch.cuda.is_available()
        else "cpu"
    )
    callback = EndAAProbeCallback(
        probing_config,
        global_args,
        embedder_batch_size=getattr(wrapper, "batch_size", global_args.batch_size),
    )
    metrics = callback.evaluate_embedder(embedder.to(device), device, show_progress=True)
    output_config = callback.cfg.get("output", {})
    report = {
        "checkpoint_path": str(Path(global_args.encoder_weights).resolve()),
        "mass_input": callback.mass_input,
        "metrics": metrics,
    }
    output_path = _write_report(report, output_config, global_args.encoder_weights)
    if output_config.get("print_report", True):
        print(json.dumps(report, indent=2, sort_keys=True))
    if output_path is not None:
        print(f"Wrote end-AA probe report: {output_path}")


if __name__ == "__main__":
    main()
