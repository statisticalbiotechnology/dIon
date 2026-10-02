"""Run the uncapped peak-only binned baseline on one raw representation cohort."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "configs/master_binned_spectrum_embedding.yaml"


def _patched_config(source: Path, key: str, report: Path, work: Path) -> Path:
    payload = yaml.safe_load(source.read_text())
    section = payload.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"{source} is missing {key}.")
    section.setdefault("output", {})["report_path"] = str(report)
    destination = work / source.name
    destination.write_text(yaml.safe_dump(payload, sort_keys=False))
    return destination


def _patched_config_with_dataset_root(source: Path, key: str, dataset_root: Path, report: Path, work: Path) -> Path:
    """Patch a canonical config to evaluate a charge-filtered dataset root."""
    payload = yaml.safe_load(source.read_text())
    section = payload.get(key)
    if not isinstance(section, dict) or not isinstance(section.get("dataset"), dict):
        raise ValueError(f"{source} is missing {key}.dataset.")
    dataset = section["dataset"]
    if key == "embedding_evaluation":
        dataset["parquet_path"] = str(dataset_root / "retrieval.parquet")
    else:
        pair_root = dataset_root / "pair_discrimination"
        dataset.update({
            "spectra_path": str(pair_root / "spectra.parquet"),
            "pairs_path": str(pair_root / "pairs.parquet"),
            "manifest_path": str(dataset_root / "manifest.json"),
        })
    section.setdefault("output", {})["report_path"] = str(report)
    destination = work / source.name
    destination.write_text(yaml.safe_dump(payload, sort_keys=False))
    return destination


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-config", type=Path, required=True)
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-peaks", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, help="Optional charge-filtered dataset root.")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="binned_full_charge_", dir="/tmp") as temp:
        work = Path(temp)
        if args.dataset_root:
            retrieval = _patched_config_with_dataset_root(args.retrieval_config, "embedding_evaluation", args.dataset_root, args.results_dir / "binned_spectrum_retrieval.json", work)
            pairs = _patched_config_with_dataset_root(args.pair_config, "pair_discrimination_evaluation", args.dataset_root, args.results_dir / "binned_spectrum_pairs.json", work)
        else:
            retrieval = _patched_config(args.retrieval_config, "embedding_evaluation", args.results_dir / "binned_spectrum_retrieval.json", work)
            pairs = _patched_config(args.pair_config, "pair_discrimination_evaluation", args.results_dir / "binned_spectrum_pairs.json", work)
        common = ["--config", str(MASTER), "--embedding_baseline", "binned_spectrum", "--accelerator", args.accelerator, "--use_mass", "0", "--use_charge", "0", "--max_peaks", str(args.max_peaks), "--batch_size", str(args.batch_size)]
        _run([sys.executable, "-u", "scripts/evaluate_embeddings.py", "--probing_config", str(retrieval), *common])
        _run([sys.executable, "-u", "scripts/evaluate_pair_discrimination.py", "--probing_config", str(pairs), *common])


if __name__ == "__main__":
    main()
