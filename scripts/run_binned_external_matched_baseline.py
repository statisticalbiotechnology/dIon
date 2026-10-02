"""Run peak-only binned-spectrum evaluation on an external matched cohort.

The external model may reject spectra during its own preprocessing. This runner
keeps the binned reference fair by evaluating the exact retained retrieval and
pair artifacts, while reusing the canonical dIon evaluators and metrics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_CONFIG = PROJECT_ROOT / "configs/master_binned_spectrum_embedding.yaml"


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _write_matched_probe_configs(
    *,
    retrieval_config: Path,
    pair_config: Path,
    work_dir: Path,
    results_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Create temporary configs pointing only at immutable matched artifacts."""
    retrieval = _load_yaml(retrieval_config)
    retrieval_eval = retrieval.get("embedding_evaluation")
    if not isinstance(retrieval_eval, dict):
        raise ValueError(f"{retrieval_config} lacks embedding_evaluation.")
    retrieval_eval["name"] = f"{retrieval_eval['name']}_external_matched"
    retrieval_eval["dataset"]["parquet_path"] = str(work_dir / "retrieval_matched.parquet")
    retrieval_eval.setdefault("output", {})["report_path"] = str(
        results_dir / "binned_spectrum_retrieval.json"
    )

    pair = _load_yaml(pair_config)
    pair_eval = pair.get("pair_discrimination_evaluation")
    if not isinstance(pair_eval, dict):
        raise ValueError(f"{pair_config} lacks pair_discrimination_evaluation.")
    pair_eval["name"] = f"{pair_eval['name']}_external_matched"
    pair_eval["dataset"].update(
        {
            "spectra_path": str(work_dir / "pairs_matched_spectra.parquet"),
            "pairs_path": str(work_dir / "pairs_matched_pairs.parquet"),
            "manifest_path": str(work_dir / "pairs_matched_manifest.json"),
        }
    )
    pair_eval.setdefault("output", {})["report_path"] = str(
        results_dir / "binned_spectrum_pairs.json"
    )

    retrieval_out = output_dir / "retrieval_probe.yaml"
    pair_out = output_dir / "pair_probe.yaml"
    retrieval_out.write_text(yaml.safe_dump(retrieval, sort_keys=False))
    pair_out.write_text(yaml.safe_dump(pair, sort_keys=False))
    return retrieval_out, pair_out


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the binned spectral cosine baseline on an external matched cohort."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--retrieval-config", type=Path, required=True)
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-peaks", type=int, default=200)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-pairs", action="store_true")
    args = parser.parse_args()

    if args.skip_retrieval and args.skip_pairs:
        raise ValueError("At least one of retrieval or pairs must be enabled.")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    required = []
    if not args.skip_retrieval:
        required.append(args.work_dir / "retrieval_matched.parquet")
    if not args.skip_pairs:
        required.extend(
            [
                args.work_dir / "pairs_matched_spectra.parquet",
                args.work_dir / "pairs_matched_pairs.parquet",
                args.work_dir / "pairs_matched_manifest.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing external matched artifacts:\n" + "\n".join(missing))

    with tempfile.TemporaryDirectory(prefix="binned_external_matched_") as tmp:
        retrieval_probe, pair_probe = _write_matched_probe_configs(
            retrieval_config=args.retrieval_config,
            pair_config=args.pair_config,
            work_dir=args.work_dir,
            results_dir=args.results_dir,
            output_dir=Path(tmp),
        )
        common = [
            "--config",
            str(MASTER_CONFIG),
            "--embedding_baseline",
            "binned_spectrum",
            "--accelerator",
            args.accelerator,
            "--use_mass",
            "0",
            "--use_charge",
            "0",
            "--max_peaks",
            str(args.max_peaks),
            "--batch_size",
            str(args.batch_size),
        ]
        if not args.skip_retrieval:
            _run(
                [
                    sys.executable,
                    "-u",
                    "scripts/evaluate_embeddings.py",
                    "--probing_config",
                    str(retrieval_probe),
                    *common,
                ]
            )
        if not args.skip_pairs:
            _run(
                [
                    sys.executable,
                    "-u",
                    "scripts/evaluate_pair_discrimination.py",
                    "--probing_config",
                    str(pair_probe),
                    *common,
                ]
            )


if __name__ == "__main__":
    main()
