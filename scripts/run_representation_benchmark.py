"""Run one registry-defined dIon representation benchmark cell.

The registry separates candidate identity, source preprocessing, and benchmark
cohort. A Slurm array supplies one ``model x conditioning x corpus`` cell;
this runner materializes resolved configs and writes reports outside checkpoint
directories.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs/evaluation/representation_benchmark_registry.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path_value: str | None, *, hash_file: bool) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_file:
        identity["sha256"] = _sha256(path)
    return identity


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_extraction_batch_size(value: str, accelerator: str) -> int:
    if value != "auto":
        return int(value)
    if accelerator != "gpu":
        return 256
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("--extraction-batch-size=auto requires nvidia-smi.")
    memories = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not memories:
        raise RuntimeError("nvidia-smi returned no GPU memory values.")
    return 1024 if min(memories) >= 70_000 else 512


def _configure_retrieval(
    template_path: Path,
    cohort_item: dict[str, Any],
    batch_size: int,
    output_path: Path,
) -> dict[str, Any]:
    config = _load_yaml(template_path)
    section = config.get("embedding_evaluation")
    if not isinstance(section, dict):
        raise ValueError(f"{template_path} lacks embedding_evaluation.")
    if cohort_item.get("retrieval_parquet"):
        section.setdefault("dataset", {})["parquet_path"] = cohort_item["retrieval_parquet"]
    section["name"] = f"{section.get('name', 'retrieval')}_resolved"
    section.setdefault("extraction", {})["batch_size"] = batch_size
    section.setdefault("output", {})["write_report"] = True
    section["output"]["report_path"] = str(output_path)
    section["output"]["print_report"] = False
    return config


def _configure_pairs(
    template_path: Path,
    cohort_item: dict[str, Any],
    batch_size: int,
    output_path: Path,
) -> dict[str, Any]:
    config = _load_yaml(template_path)
    section = config.get("pair_discrimination_evaluation")
    if not isinstance(section, dict):
        raise ValueError(f"{template_path} lacks pair_discrimination_evaluation.")
    replacements = {
        "pair_spectra": "spectra_path",
        "pairs": "pairs_path",
        "pair_manifest": "manifest_path",
    }
    for registry_key, config_key in replacements.items():
        if cohort_item.get(registry_key):
            section.setdefault("dataset", {})[config_key] = cohort_item[registry_key]
    section["name"] = f"{section.get('name', 'pairs')}_resolved"
    section.setdefault("extraction", {})["batch_size"] = batch_size
    section.setdefault("output", {})["write_report"] = True
    section["output"]["report_path"] = str(output_path)
    section["output"]["print_report"] = False
    return config


def _base_command(model: dict[str, Any], conditioning: str, accelerator: str) -> list[str]:
    preprocessing = model["preprocessing"]
    command = [
        sys.executable,
        "--config",
        str(model["master_config"]),
        "--pretrain_config",
        str(model["pretrain_config"]),
        "--pretraining_task",
        str(model["pretraining_task"]),
        "--embedding_baseline",
        "model",
        "--accelerator",
        accelerator,
        "--precision",
        "bf16-mixed",
        "--num_devices",
        "1",
        "--num_nodes",
        "1",
        "--log_wandb",
        "0",
        "--max_peaks",
        str(preprocessing["max_peaks"]),
        "--use_mass",
        str(int(bool(preprocessing["use_mass"]))),
        "--use_charge",
        str(int(bool(preprocessing["use_charge"]))),
        "--use_energy",
        str(int(bool(preprocessing["use_energy"]))),
        "--max_charge",
        str(preprocessing["max_charge"]),
        "--peak_filter_method",
        str(preprocessing["peak_filter_method"]),
        "--intensity_scaling",
        str(preprocessing["intensity_scaling"]),
        "--min_mz",
        str(preprocessing["min_mz"]),
        "--max_mz",
        str(preprocessing["max_mz"]),
        "--min_intensity",
        str(preprocessing["min_intensity"]),
        "--min_peaks",
        str(preprocessing["min_peaks"]),
        "--remove_precursor_tol",
        str(preprocessing["remove_precursor_tol"]),
        "--precursor_conditioning",
        conditioning,
    ]
    kind = model["kind"]
    if kind == "random_dino":
        command += [
            "--random_encoder",
            "--random_encoder_seed",
            str(model["random_seed"]),
            "--embedding_readout",
            str(model.get("embedding_readout", "backbone")),
        ]
    elif kind == "dino_checkpoint":
        command += ["--encoder_weights", str(model["checkpoint"])]
        command += ["--embedding_readout", str(model.get("embedding_readout", "backbone"))]
    elif kind == "metric_learning_checkpoint":
        command += [
            "--downstream_task",
            "metric_learning",
            "--downstream_config",
            str(model["downstream_config"]),
            "--downstream_weights",
            str(model["checkpoint"]),
        ]
    else:
        raise ValueError(f"Unsupported model kind {kind!r}.")
    return command


def _write_yaml(path: Path, content: dict[str, Any]) -> None:
    with path.open("w") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def _model_matches_set(model: dict[str, Any], model_set: str) -> bool:
    if model_set == "all":
        return True
    if model_set == "raw_dino":
        return model["kind"] in {"random_dino", "dino_checkpoint"}
    if model_set == "metric_learning":
        return model["kind"] == "metric_learning_checkpoint"
    raise ValueError(f"Unknown model set {model_set!r}.")


def _cells(registry: dict[str, Any], cohort: str, split: str, model_set: str) -> list[tuple[str, str, str]]:
    split_entry = registry["cohorts"][cohort][split]
    if split_entry.get("status") != "ready":
        raise ValueError(
            f"{cohort}/{split} is not runnable: {split_entry.get('reason', 'unknown reason')}"
        )
    corpora = sorted(key for key in split_entry if key not in {"status", "reason"})
    cells: list[tuple[str, str, str]] = []
    for model_id, model in registry["models"].items():
        if not _model_matches_set(model, model_set):
            continue
        for conditioning in model["conditioning_modes"]:
            for corpus in corpora:
                cells.append((model_id, conditioning, corpus))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model")
    parser.add_argument("--conditioning", choices=["conditioned", "null"])
    parser.add_argument("--corpus", choices=["ninespecies_v2", "bacterial", "kingdoms"])
    parser.add_argument("--model-set", default="all", choices=["all", "raw_dino", "metric_learning"])
    parser.add_argument("--list-cells", action="store_true", help="Print model, conditioning, corpus rows for a Slurm array.")
    parser.add_argument("--cohort", default="charge2to4", choices=["primary", "charge2to4"])
    parser.add_argument("--split", default="validation", choices=["validation", "locked_test"])
    parser.add_argument("--tasks", default="both", choices=["retrieval", "pairs", "both"])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--extraction-batch-size", default="auto")
    parser.add_argument(
        "--max-peaks-override",
        type=int,
        help="Override the registry preprocessing cap for an explicitly labelled inference study; zero means uncapped.",
    )
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = _load_yaml(args.registry)
    if args.list_cells:
        if args.split == "locked_test" and not args.allow_locked_test:
            raise ValueError("Locked test requires explicit --allow-locked-test.")
        for model_id, conditioning, corpus in _cells(
            registry, args.cohort, args.split, args.model_set
        ):
            print(f"{model_id}\t{conditioning}\t{corpus}")
        return
    if not args.model or not args.conditioning or not args.corpus or args.output_root is None:
        parser.error("--model, --conditioning, --corpus, and --output-root are required unless --list-cells is used.")
    try:
        model = copy.deepcopy(registry["models"][args.model])
        split_entry = registry["cohorts"][args.cohort][args.split]
    except KeyError as exc:
        raise ValueError(f"Unknown registry entry: {exc}") from exc
    if args.conditioning not in model["conditioning_modes"]:
        raise ValueError(
            f"{args.model} supports {model['conditioning_modes']}, not {args.conditioning}."
        )
    if args.max_peaks_override is not None:
        if args.max_peaks_override < 0:
            raise ValueError("--max-peaks-override must be non-negative.")
        model["preprocessing"]["max_peaks"] = args.max_peaks_override
    if args.split == "locked_test" and not args.allow_locked_test:
        raise ValueError("Locked test requires explicit --allow-locked-test.")
    if split_entry.get("status") != "ready":
        raise ValueError(
            f"{args.cohort}/{args.split} is not runnable: {split_entry.get('reason', 'unknown reason')}"
        )
    cohort_item = copy.deepcopy(split_entry[args.corpus])
    for path_key in ("retrieval_parquet", "pair_spectra", "pairs"):
        if cohort_item.get(path_key) and not Path(cohort_item[path_key]).is_file():
            raise FileNotFoundError(f"Missing {path_key}: {cohort_item[path_key]}")

    batch_size = _resolve_extraction_batch_size(args.extraction_batch_size, args.accelerator)
    run_dir = args.output_root / args.model / args.conditioning / args.corpus
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_tasks = ("retrieval", "pairs") if args.tasks == "both" else (args.tasks,)
    output_paths = {"retrieval": run_dir / "retrieval.json", "pairs": run_dir / "pairs.json"}
    existing = [path for task, path in output_paths.items() if task in requested_tasks and path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing reports: {existing}")

    resolved = run_dir / "resolved_configs"
    resolved.mkdir(exist_ok=True)
    commands: dict[str, list[str]] = {}
    if "retrieval" in requested_tasks:
        retrieval_config = _configure_retrieval(
            PROJECT_ROOT / cohort_item["retrieval_config"], cohort_item, batch_size, output_paths["retrieval"]
        )
        retrieval_path = resolved / "retrieval.yaml"
        _write_yaml(retrieval_path, retrieval_config)
        commands["retrieval"] = [
            *_base_command(model, args.conditioning, args.accelerator),
            "--probing_config", str(retrieval_path),
            "--output_dir", str(run_dir),
            "scripts/evaluate_embeddings.py",
        ]
    if "pairs" in requested_tasks:
        pair_config = _configure_pairs(
            PROJECT_ROOT / cohort_item["pair_config"], cohort_item, batch_size, output_paths["pairs"]
        )
        pair_path = resolved / "pairs.yaml"
        _write_yaml(pair_path, pair_config)
        commands["pairs"] = [
            *_base_command(model, args.conditioning, args.accelerator),
            "--probing_config", str(pair_path),
            "--output_dir", str(run_dir),
            "scripts/evaluate_pair_discrimination.py",
        ]

    # The interpreter invokes a script, so move it directly after sys.executable.
    commands = {
        task: [command[0], command[-1], *command[1:-1]]
        for task, command in commands.items()
    }
    experiment_metadata = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "registry": str(args.registry.resolve()),
        "registry_sha256": _sha256(args.registry),
        "cohort": args.cohort,
        "split": args.split,
        "locked_test_enabled": bool(args.allow_locked_test),
        "max_peaks_override": args.max_peaks_override,
    }
    experiment_path = args.output_root / "experiment_metadata.json"
    if not experiment_path.exists():
        experiment_path.write_text(json.dumps(experiment_metadata, indent=2, sort_keys=True) + "\n")

    metadata = {
        "status": "started",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "registry": str(args.registry.resolve()),
        "model_id": args.model,
        "model": model,
        "cohort": args.cohort,
        "split": args.split,
        "corpus": args.corpus,
        "conditioning": args.conditioning,
        "extraction_batch_size": batch_size,
        "max_peaks_override": args.max_peaks_override,
        "checkpoint": _file_identity(model.get("checkpoint"), hash_file=args.hash_checkpoint),
        "retrieval_manifest": _file_identity(cohort_item.get("retrieval_manifest"), hash_file=True),
        "pair_manifest": _file_identity(cohort_item.get("pair_manifest"), hash_file=True),
        "commands": {task: [str(value) for value in command] for task, command in commands.items()},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    for task, command in commands.items():
        print(f"[{task}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    metadata["status"] = "completed" if not args.dry_run else "dry_run"
    metadata["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata["reports"] = {
        task: str(output_paths[task].resolve())
        for task in requested_tasks
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Completed representation benchmark cell: {run_dir}")


if __name__ == "__main__":
    main()
