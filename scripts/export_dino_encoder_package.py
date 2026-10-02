#!/usr/bin/env python3
"""Export a standalone EMA-teacher encoder package from a DINO checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from src.models.custom.encoder import encoder_larger_deeper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrain-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_encoder_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu")
    if "state_dict" not in payload:
        raise ValueError("Expected a Lightning checkpoint containing state_dict")

    state_dict = payload["state_dict"]
    prefix = "teacher.backbone."
    state = {
        key[len(prefix) :]: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(f"Checkpoint contains no {prefix!r} parameters")
    return state


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    with args.pretrain_config.open() as handle:
        task = yaml.safe_load(handle)["dion"]

    state = load_encoder_state(args.checkpoint)
    encoder = encoder_larger_deeper(
        use_mass=True,
        use_charge=True,
        use_energy=False,
        dropout=float(task.get("dropout", 0.15)),
        cls_token=False,
        max_charge=10,
    )
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Standalone encoder verification failed: missing={missing}, "
            f"unexpected={unexpected}"
        )
    encoder.eval()

    package = args.output_dir / "dino_encoder"
    package.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1] / "src/models/custom"
    shutil.copy2(source_root / "encoder.py", package / "encoder.py")
    shutil.copy2(source_root / "model_parts.py", package / "model_parts.py")
    (package / "encoder.py").write_text(
        (package / "encoder.py")
        .read_text()
        .replace("import src.models.custom.model_parts as mp", "from . import model_parts as mp")
    )
    (package / "__init__.py").write_text(
        "from .encoder import Encoder, encoder_larger_deeper\n\n"
        "__all__ = ['Encoder', 'encoder_larger_deeper']\n"
    )

    torch.save({"state_dict": state}, args.output_dir / "encoder_larger_deeper_epoch100.pt")
    metadata = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_branch": "teacher.backbone",
        "architecture": "encoder_larger_deeper",
        "running_units": encoder.running_units,
        "use_mass": True,
        "use_charge": True,
        "max_charge": 10,
        "dropout": float(task.get("dropout", 0.15)),
        "weights_format": "PyTorch payload with state_dict key",
        "source_pretrain_config": str(args.pretrain_config.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Standalone DINO Encoder\n\n"
        "This package contains only the EMA-teacher `encoder_larger_deeper`\n"
        "backbone exported from the hybrid DINO checkpoint. It is intended as\n"
        "a dense encoder for downstream de novo sequencing.\n\n"
        "```python\n"
        "import torch\n"
        "from dino_encoder import encoder_larger_deeper\n\n"
        "encoder = encoder_larger_deeper(\n"
        "    use_mass=True, use_charge=True, use_energy=False,\n"
        "    dropout=0.15, cls_token=False, max_charge=10,\n"
        ")\n"
        "payload = torch.load(\n"
        "    'encoder_larger_deeper_epoch100.pt',\n"
        "    map_location='cpu',\n"
        ")\n"
        "encoder.load_state_dict(payload['state_dict'])\n"
        "encoder.eval()\n"
        "```\n\n"
        "The forward input is a padded `[batch, peaks, 2]` tensor containing\n"
        "m/z and intensity, plus `[batch]` `mass`, `[batch]` `charge`, and an\n"
        "optional `key_padding_mask`. Use canonical precursor mass, not raw\n"
        "m/z. The encoder returns the dense token representation dictionary\n"
        "used by the de novo cross-attention decoder.\n"
    )
    print(f"Verified {len(state)} encoder tensors")
    print(f"Wrote standalone encoder package: {args.output_dir}")


if __name__ == "__main__":
    main()
