#!/usr/bin/env python3
"""Build exact Kingdoms peptide precision--coverage curves and a paper PDF.

The two in-house CSVs are re-scored with dIon's canonical PA1.1 peptide
matcher.  Casanovo and InstaNovo curves are the previously validated outputs
of their corresponding released-model evaluators.  All curves retain the
canonical 4,926,232-spectrum denominator; a no-prediction row is an
incorrect prediction ranked at negative infinity.

Example
-------
PYTHONPATH=. "$PY" -u scripts/plot_kingdoms_denovo_precision_coverage.py

The default paths point at the transferred Kingdoms prediction artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from tqdm import tqdm

from src import casanovo_eval as evaluate
from src.data.unified_tokenizer import PeptideTokenizer


CANONICAL_DENOMINATOR = 4_926_232
ROOT = Path("/path/to/results/denovo_eval")
INHOUSE_ROOT = ROOT / "denovo_kingdoms_species_cap100k/200_peaks_prediction_mode"
OUTPUT_ROOT = ROOT / "denovo_kingdoms_species_cap100k/precision_coverage"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    color: str
    marker: str
    curve: Path | None = None
    predictions: Path | None = None


MODELS = (
    ModelSpec(
        "scratch_dnlv1",
        "Scratch (dIon-de-novo-labeled-v1 (DNLv1))",
        "#c779c3",
        "s",
        predictions=INHOUSE_ROOT / "v5_scratch_encoderld/predictions.csv",
    ),
    ModelSpec(
        "dion_dnlv1",
        "dIon (dIon-de-novo-labeled-v1 (DNLv1))",
        "#087fba",
        "o",
        predictions=INHOUSE_ROOT / "v5_hybrid300_finetuned/predictions.csv",
    ),
    ModelSpec(
        "casanovo_v5_2_1",
        "Casanovo 5.2.1",
        "#dc8b00",
        "^",
        curve=ROOT
        / "casanovo_v5_2_1/kingdoms_species_cap100k_full_denominator_sharded/precision_coverage.csv",
    ),
    ModelSpec(
        "instanovo_v1_2_2",
        "InstaNovo 1.2.2",
        "#4d9d68",
        "D",
        curve=ROOT
        / "instanovo_v1_2_2/kingdoms_species_cap100k_full_charge_sharded/precision_coverage.csv",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tokenizer-manifest", type=Path, default=Path("configs/tokenizers/pa11.json"))
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--recompute-inhouse",
        action="store_true",
        help="Recompute the two in-house curves even when their cached CSVs exist.",
    )
    return parser.parse_args()


def _bool(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"Expected boolean CSV value, got {value!r}")


def _write_curve(path: Path, indices: np.ndarray, scores: np.ndarray, matches: np.ndarray) -> dict[str, object]:
    order = np.argsort(-scores, kind="stable")
    ranked_indices = indices[order]
    ranked_scores = scores[order]
    ranked_matches = matches[order]
    coverage = np.arange(1, len(ranked_matches) + 1, dtype=np.float64) / CANONICAL_DENOMINATOR
    precision = np.cumsum(ranked_matches, dtype=np.int64) / np.arange(1, len(ranked_matches) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("rank", "source_spectrum_index", "confidence", "peptide_match", "coverage", "peptide_precision"),
        )
        writer.writeheader()
        for position, (index, score, match) in enumerate(
            tqdm(
                zip(ranked_indices, ranked_scores, ranked_matches, strict=True),
                total=len(ranked_indices),
                desc=f"Write {path.stem}",
                unit="row",
            ),
            start=1,
        ):
            writer.writerow(
                {
                    "rank": position,
                    "source_spectrum_index": int(index),
                    "confidence": repr(float(score)),
                    "peptide_match": int(match),
                    "coverage": repr(float(coverage[position - 1])),
                    "peptide_precision": repr(float(precision[position - 1])),
                }
            )
    return {
        "canonical_denominator": CANONICAL_DENOMINATOR,
        "emitted_predictions": int(np.isfinite(scores).sum()),
        "no_prediction_count": int((~np.isfinite(scores)).sum()),
        "peptide_matches": int(matches.sum()),
        "peptide_precision_at_full_coverage": float(precision[-1]),
        "peptide_precision_coverage_auc": float(auc(coverage, precision)),
    }


def score_inhouse_predictions(
    predictions: Path,
    output_curve: Path,
    tokenizer_manifest: Path,
    batch_size: int,
) -> dict[str, object]:
    """Score a canonical-indexed in-house prediction CSV without reordering it."""
    tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(
        tokenizer_manifest, replace_isoleucine_with_leucine=True
    )
    masses = tokenizer.residues
    indices = np.arange(CANONICAL_DENOMINATOR, dtype=np.int64)
    scores = np.full(CANONICAL_DENOMINATOR, -np.inf, dtype=np.float64)
    matches = np.zeros(CANONICAL_DENOMINATOR, dtype=np.uint8)
    seen = np.zeros(CANONICAL_DENOMINATOR, dtype=bool)
    pending_indices: list[int] = []
    pending_truth: list[list[str]] = []
    pending_prediction: list[list[str] | None] = []

    def flush() -> None:
        if not pending_indices:
            return
        batch_matches, _, _ = evaluate.aa_match_batch(pending_truth, pending_prediction, masses)
        for index, match in zip(pending_indices, batch_matches, strict=True):
            matches[index] = int(match[1])
        pending_indices.clear()
        pending_truth.clear()
        pending_prediction.clear()

    with predictions.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "canonical_index",
            "true_sequence",
            "predicted_sequence",
            "peptide_confidence",
            "no_prediction",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{predictions} is missing columns: {sorted(missing)}")
        for row in tqdm(reader, total=CANONICAL_DENOMINATOR, desc=f"Read + match {predictions.parent.name}", unit="row"):
            index = int(row["canonical_index"])
            if index < 0 or index >= CANONICAL_DENOMINATOR or seen[index]:
                raise ValueError(f"Invalid or duplicate canonical_index {index} in {predictions}")
            seen[index] = True
            no_prediction = _bool(row["no_prediction"])
            if no_prediction:
                if row["predicted_sequence"]:
                    raise ValueError(f"No-prediction row {index} unexpectedly contains a peptide")
                continue
            score = float(row["peptide_confidence"])
            if not np.isfinite(score):
                raise ValueError(f"Emitted prediction {index} has non-finite confidence")
            truth = tokenizer.preprocess_sequence(row["true_sequence"])
            prediction = tokenizer.preprocess_sequence(row["predicted_sequence"])
            scores[index] = score
            pending_indices.append(index)
            pending_truth.append(truth)
            pending_prediction.append(prediction)
            if len(pending_indices) >= batch_size:
                flush()
    flush()
    if not np.all(seen):
        raise ValueError(f"{predictions} lacks {int((~seen).sum()):,} canonical rows")
    report = _write_curve(output_curve, indices, scores, matches)
    report.update(
        {
            "prediction_source": str(predictions.resolve()),
            "scoring_mode": "dion_pa11_canonical_mass_matcher",
            "confidence_definition": "mean emitted amino-acid probability, including stop-token confidence, minus 1 when the peptide does not fit the precursor-m/z tolerance",
        }
    )
    return report


def validate_inhouse_report(predictions: Path, report: dict[str, object]) -> None:
    """Bound the canonical re-score against the transferred distributed metric."""
    manifest_path = predictions.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("canonical_denominator") != CANONICAL_DENOMINATOR:
        raise ValueError(f"Unexpected canonical denominator in {manifest_path}")
    expected = float(manifest["peptide_precision_at_full_coverage"])
    observed = float(report["peptide_precision_at_full_coverage"])
    delta = observed - expected
    distributed_bound = 8 / (CANONICAL_DENOMINATOR + 8)
    if abs(delta) > distributed_bound:
        raise ValueError(
            f"Canonical re-score differs from {manifest_path} by {delta:.16g}, "
            f"outside the documented distributed-evaluation bound {distributed_bound:.16g}"
        )
    report["validated_against_manifest"] = str(manifest_path.resolve())
    report["manifest_peptide_precision_at_full_coverage"] = expected
    report["manifest_minus_canonical_precision"] = expected - observed
    report["manifest_minus_canonical_effective_matches"] = (
        (expected - observed) * CANONICAL_DENOMINATOR
    )
    report["manifest_comparison"] = (
        "Canonical CSV re-score is authoritative; the transferred manifest value is an "
        "online distributed aggregate and agrees within the documented eight-record bound."
    )


def read_curve(path: Path, points: int = 4_000) -> tuple[np.ndarray, np.ndarray]:
    """Read a large curve while retaining evenly spaced rank locations for plotting."""
    n_rows = sum(1 for _ in path.open()) - 1
    if n_rows != CANONICAL_DENOMINATOR:
        raise ValueError(f"{path} has {n_rows:,} rows; expected {CANONICAL_DENOMINATOR:,}")
    keep = set(np.linspace(0, n_rows - 1, min(points, n_rows), dtype=np.int64).tolist())
    coverage, precision = [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for position, row in enumerate(reader):
            if position in keep:
                coverage.append(float(row["coverage"]))
                precision.append(float(row["peptide_precision"]))
    return np.asarray(coverage), np.asarray(precision)


def plot_curves(curves: dict[str, Path], output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9.6, 6.6), constrained_layout=True)
    for spec in MODELS:
        coverage, precision = read_curve(curves[spec.key])
        axis.plot(
            coverage,
            precision,
            label=spec.label,
            color=spec.color,
            linewidth=2.8,
            marker=spec.marker,
            markersize=4.3,
            markevery=max(1, len(coverage) // 14),
        )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Coverage", fontsize=20, labelpad=8)
    axis.set_ylabel("Peptide precision", fontsize=20, labelpad=8)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(axis="y", color="#d5dbe1", linewidth=1.15)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    legend = axis.legend(loc="best", frameon=True, fontsize=13)
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_alpha(0.96)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    curves: dict[str, Path] = {}
    reports: dict[str, dict[str, object]] = {}
    for spec in MODELS:
        curve = args.output_root / f"{spec.key}_precision_coverage.csv" if spec.predictions else spec.curve
        assert curve is not None
        if spec.predictions is not None:
            report_path = args.output_root / f"{spec.key}_metrics.json"
            if args.recompute_inhouse or not curve.exists():
                reports[spec.key] = score_inhouse_predictions(
                    spec.predictions, curve, args.tokenizer_manifest, args.batch_size
                )
                validate_inhouse_report(spec.predictions, reports[spec.key])
                report_path.write_text(json.dumps(reports[spec.key], indent=2, sort_keys=True) + "\n")
            elif not report_path.exists():
                raise FileNotFoundError(f"Cached curve exists but report is absent: {report_path}")
            else:
                reports[spec.key] = json.loads(report_path.read_text())
        curves[spec.key] = curve
    plot_path = args.output_root / "kingdoms_denovo_peptide_precision_coverage.pdf"
    plot_curves(curves, plot_path)
    print(json.dumps({"curves": {key: str(path) for key, path in curves.items()}, "plot": str(plot_path), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
