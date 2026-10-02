"""Validate a frozen locked-test selection manifest and emit Slurm array cells."""
from __future__ import annotations
import argparse, json, shlex
from pathlib import Path
import yaml


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("status") != "frozen":
        raise ValueError("Selection manifest status must be 'frozen' before locked-test use.")
    if not data.get("selection_basis"):
        raise ValueError("Selection manifest must record validation selection_basis.")
    return data


def representation(data: dict, registry_path: Path) -> list[tuple[str, str, str]]:
    registry = yaml.safe_load(registry_path.read_text())
    cells=[]
    for item in data.get("representation", {}).get("models", []):
        model_id=item["model_id"]
        model=registry["models"].get(model_id)
        if model is None: raise ValueError(f"Unknown registry model {model_id!r}.")
        for conditioning in item.get("conditionings", model["conditioning_modes"]):
            if conditioning not in model["conditioning_modes"]:
                raise ValueError(f"{model_id} does not support {conditioning}.")
            for corpus in item.get("corpora", ["bacterial", "ninespecies_v2", "kingdoms"]):
                cells.append((model_id, conditioning, corpus))
    if not cells: raise ValueError("Frozen manifest contains no representation selections.")
    return cells


def downstream(data: dict, index: int) -> str:
    items=data.get("downstream", [])
    if index < 0 or index >= len(items): raise ValueError(f"Downstream index {index} is outside 0..{len(items)-1}.")
    item=items[index]
    required=("task","model_label","master_config","downstream_config","encoder_checkpoint","downstream_checkpoint","freeze_encoder","precursor_conditioning","decoder_model","wandb_project")
    missing=[key for key in required if key not in item or item[key] in (None, "")]
    if missing: raise ValueError(f"Downstream selection {index} is missing {missing}.")
    for key, value in item.items():
        rendered = str(int(value)) if isinstance(value, bool) else str(value)
        print(f"{key.upper()}={shlex.quote(rendered)}")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--registry", type=Path, default=Path("configs/evaluation/representation_benchmark_registry.yaml"))
    p.add_argument("--list-representation", action="store_true")
    p.add_argument("--downstream-index", type=int)
    a=p.parse_args(); data=load(a.selection)
    if a.list_representation:
        for cell in representation(data, a.registry): print("\t".join(cell))
    elif a.downstream_index is not None: downstream(data, a.downstream_index)
    else: p.error("select --list-representation or --downstream-index")
if __name__ == "__main__": main()
