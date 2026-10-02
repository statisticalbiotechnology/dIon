#!/usr/bin/env python3
"""Score released InstaNovo v1.2 CSV predictions against annotated MGF labels.

The official v1.2 transformer writes one CSV row per emitted MGF ordinal in
``prediction_id``. This scorer maps its UNIMOD vocabulary to dIon canonical
tokens and evaluates them using the same canonical residue masses and matching
criteria as every other model. It ranks by descending ``log_probs`` and
preserves inputs without emitted predictions as explicit zero-confidence peptide
errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import casanovo_eval as evaluate
from src.data.unified_tokenizer import PeptideTokenizer

DEFAULT_TOKENIZER_MANIFEST = PROJECT_ROOT / "configs/tokenizers/pa11.json"

UNIMOD_TO_TOKEN = {
    "M[UNIMOD:35]": "M[Oxidation]",
    "C[UNIMOD:4]": "C[Carbamidomethyl]",
    "N[UNIMOD:7]": "N[Deamidated]",
    "Q[UNIMOD:7]": "Q[Deamidated]",
    "S[UNIMOD:21]": "S[Phospho]",
    "T[UNIMOD:21]": "T[Phospho]",
    "Y[UNIMOD:21]": "Y[Phospho]",
    "[UNIMOD:1]-": "[Acetyl]-",
    "[UNIMOD:5]-": "[Carbamyl]-",
    "[UNIMOD:385]-": "[Ammonia-loss]-",
}
UNIMOD_TOKEN = re.compile(r"(?:[A-Z])?\[UNIMOD:\d+\]")
PREDICTION_TOKEN = re.compile(
    r"[A-Z]\[(?:Oxidation|Carbamidomethyl|Deamidated|Phospho)\]"
    r"|\[(?:Acetyl|Carbamyl|Ammonia-loss)\]-"
    r"|[A-Z]"
)
# Canonical phospho delta used by the dIon/Casanovo matching protocol.
# InstaNovo's checkpoint stores a rounded 79.966000 delta, but model-specific
# vocabulary rounding must not alter a cross-model evaluation.
CANONICAL_PHOSPHO_MASS_DELTA = 79.966331



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument(
        "--mass-checkpoint",
        type=Path,
        help="Optional provenance only; it never supplies matching masses or tokenization.",
    )
    parser.add_argument(
        "--tokenizer-manifest",
        type=Path,
        default=DEFAULT_TOKENIZER_MANIFEST,
        help="dIon canonical numeric-mass tokenizer used for every prediction source.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-coverage-output", type=Path)
    parser.add_argument("--full-denominator-count", type=int)
    parser.add_argument("--supported-to-full-index", type=Path)
    args = parser.parse_args()
    if (args.full_denominator_count is None) != (args.supported_to_full_index is None):
        parser.error("--full-denominator-count and --supported-to-full-index must be supplied together.")
    if args.full_denominator_count is not None and args.full_denominator_count < 1:
        parser.error("--full-denominator-count must be positive.")
    return args


def mgf_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    sequence: str | None = None
    with path.open() as handle, tqdm(
        total=path.stat().st_size, unit="B", unit_scale=True,
        desc="Read InstaNovo MGF labels",
    ) as progress:
        for raw_line in handle:
            progress.update(len(raw_line.encode("ascii")))
            line = raw_line.rstrip("\n")
            if line == "BEGIN IONS":
                sequence = None
            elif line.startswith("SEQ="):
                sequence = line.removeprefix("SEQ=")
            elif line == "END IONS":
                if sequence is None:
                    raise ValueError(f"MGF record without SEQ in {path}.")
                sequences.append(sequence)
    return sequences


def normalize_prediction(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("InstaNovo emitted an empty peptide prediction.")
    for source, target in UNIMOD_TO_TOKEN.items():
        normalized = normalized.replace(source, target)
    unknown = UNIMOD_TOKEN.search(normalized)
    if unknown is not None:
        raise ValueError(f"Unsupported InstaNovo UNIMOD token: {unknown.group(0)!r}")
    return normalized


def tokenize_prediction(value: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    while position < len(value):
        match = PREDICTION_TOKEN.match(value, position)
        if match is None:
            raise ValueError(f"Unsupported normalized InstaNovo peptide syntax at {position}: {value!r}")
        tokens.append(match.group(0))
        position = match.end()
    return tokens


def load_predictions(path: Path, expected_count: int) -> tuple[np.ndarray, list[str], np.ndarray, list[int]]:
    predictions: list[str | None] = [None] * expected_count
    scores = np.empty(expected_count, dtype=np.float64)
    seen = np.zeros(expected_count, dtype=bool)
    with path.open(newline="") as handle, tqdm(
        total=path.stat().st_size, unit="B", unit_scale=True,
        desc="Read InstaNovo predictions",
    ) as progress:
        reader = csv.DictReader(handle)
        required = {"prediction_id", "predictions", "log_probs"}
        fields = set(reader.fieldnames or [])
        missing_columns = required.difference(fields)
        if missing_columns:
            raise ValueError(f"Prediction CSV lacks required columns: {sorted(missing_columns)}")
        for row in reader:
            progress.update(sum(len(str(value)) + 1 for value in row.values()))
            try:
                index = int(row["prediction_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid InstaNovo prediction_id: {row.get('prediction_id')!r}") from exc
            if index < 0 or index >= expected_count or seen[index]:
                raise ValueError(f"Invalid or duplicate InstaNovo prediction_id: {index}")
            predictions[index] = normalize_prediction(row["predictions"])
            try:
                scores[index] = float(row["log_probs"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid InstaNovo log_probs for prediction_id {index}") from exc
            if not np.isfinite(scores[index]):
                raise ValueError(f"Non-finite InstaNovo log_probs for prediction_id {index}")
            seen[index] = True
    indices = np.flatnonzero(seen)
    return indices, [predictions[index] for index in indices], scores[indices], np.flatnonzero(~seen).tolist()



def main() -> None:
    args = parse_args()
    truth = mgf_sequences(args.mgf)
    indices, predicted, scores, missing = load_predictions(args.predictions_csv, len(truth))
    tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(
        args.tokenizer_manifest,
        replace_isoleucine_with_leucine=True,
    )
    truth_tokens = [tokenizer.preprocess_sequence(truth[index]) for index in indices]
    prediction_tokens = [tokenizer.preprocess_sequence(value) for value in predicted]
    masses = tokenizer.residues
    order = np.argsort(-scores, kind="stable")

    matches = []
    n_true = n_pred = 0
    match_batch_size = 100_000
    for start in tqdm(
        range(0, len(order), match_batch_size),
        total=(len(order) + match_batch_size - 1) // match_batch_size,
        desc="Match InstaNovo peptides", unit="batch",
    ):
        batch_order = order[start : start + match_batch_size]
        batch_matches, batch_n_true, batch_n_pred = evaluate.aa_match_batch(
            [truth_tokens[index] for index in batch_order],
            [prediction_tokens[index] for index in batch_order],
            masses,
        )
        matches.extend(batch_matches)
        n_true += batch_n_true
        n_pred += batch_n_pred
    peptide_matches = np.asarray([match[1] for match in matches], dtype=np.float64)
    ranked_indices = indices[order].astype(np.int64, copy=False)
    ranked_scores = scores[order]

    denominator = len(peptide_matches)
    unsupported_charge_count = 0
    no_prediction_indices = np.empty(0, dtype=np.int64)
    if args.full_denominator_count is not None:
        mapping = np.load(args.supported_to_full_index, allow_pickle=False)
        if mapping.ndim != 1 or mapping.dtype.kind not in "iu":
            raise ValueError("--supported-to-full-index must be a one-dimensional integer NumPy array.")
        if len(mapping) != len(truth):
            raise ValueError(f"Mapping length {len(mapping)} does not match MGF count {len(truth)}.")
        mapping = mapping.astype(np.int64, copy=False)
        if args.full_denominator_count < len(mapping):
            raise ValueError("Full denominator cannot be smaller than MGF count.")
        if np.any(mapping < 0) or np.any(mapping >= args.full_denominator_count):
            raise ValueError("Supported-to-full mapping contains an out-of-range canonical index.")
        if len(np.unique(mapping)) != len(mapping):
            raise ValueError("Supported-to-full mapping must be one-to-one.")
        denominator = args.full_denominator_count
        unsupported_charge_count = denominator - len(mapping)
        ranked_indices = mapping[ranked_indices]
        included = np.zeros(denominator, dtype=bool)
        included[ranked_indices] = True
        no_prediction_indices = np.flatnonzero(~included).astype(np.int64, copy=False)
        peptide_matches = np.concatenate((peptide_matches, np.zeros(len(no_prediction_indices))))
        ranked_scores = np.concatenate((ranked_scores, np.full(len(no_prediction_indices), -np.inf)))
        ranked_indices = np.concatenate((ranked_indices, no_prediction_indices))
    elif missing:
        denominator = len(truth)
        missing_indices = np.asarray(missing, dtype=np.int64)
        peptide_matches = np.concatenate((peptide_matches, np.zeros(len(missing_indices))))
        ranked_scores = np.concatenate((ranked_scores, np.full(len(missing_indices), -np.inf)))
        ranked_indices = np.concatenate((ranked_indices, missing_indices))

    # Full-coverage AA metrics must include true residues from inputs without
    # emitted predictions; their prediction is represented as None.
    aa_precision_at_full_coverage: float | None = None
    aa_recall_at_full_coverage: float | None = None
    full_n_true = n_true
    full_n_pred = n_pred
    if args.full_denominator_count is None:
        all_truth_tokens = [tokenizer.preprocess_sequence(sequence) for sequence in truth]
        all_prediction_tokens: list[list[str] | None] = [None] * len(truth)
        for index, tokens in zip(indices, prediction_tokens, strict=True):
            all_prediction_tokens[int(index)] = tokens
        full_matches, full_n_true, full_n_pred = evaluate.aa_match_batch(
            all_truth_tokens, all_prediction_tokens, masses
        )
        aa_precision_at_full_coverage, aa_recall_at_full_coverage, _ = evaluate.aa_match_metrics(
            full_matches, full_n_true, full_n_pred
        )

    coverage = np.arange(1, len(peptide_matches) + 1, dtype=np.float64) / denominator
    precision = np.cumsum(peptide_matches) / np.arange(1, len(peptide_matches) + 1)
    if args.precision_coverage_output is not None:
        args.precision_coverage_output.parent.mkdir(parents=True, exist_ok=True)
        with args.precision_coverage_output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("rank", "source_spectrum_index", "confidence", "peptide_match", "coverage", "peptide_precision"),
            )
            writer.writeheader()
            for rank, (index, score) in enumerate(
                tqdm(
                    zip(ranked_indices, ranked_scores, strict=True),
                    total=len(ranked_indices), desc="Write precision-coverage curve", unit="row",
                ),
                start=1,
            ):
                writer.writerow({
                    "rank": rank,
                    "source_spectrum_index": int(index),
                    "confidence": repr(float(score)),
                    "peptide_match": int(peptide_matches[rank - 1]),
                    "coverage": repr(float(coverage[rank - 1])),
                    "peptide_precision": repr(float(precision[rank - 1])),
                })

    report = {
        "metric": "peptide_precision_coverage_auc",
        "metric_definition": "trapezoidal AUC of descending-InstaNovo-log_probs peptide precision versus coverage",
        "prediction_source": "instanovo_v1_2_2_csv",
        "mgf": str(args.mgf.resolve()),
        "predictions_csv": str(args.predictions_csv.resolve()),
        "mass_checkpoint": str(args.mass_checkpoint.resolve()) if args.mass_checkpoint else None,
        "tokenizer_manifest": str(args.tokenizer_manifest.resolve()),
        "scoring_mode": "dion_pa11_canonical_mass_matcher",
        "mgf_spectrum_count": len(truth),
        "predicted_spectrum_count": len(predicted),
        "missing_mgf_indices": missing,
        "missing_supported_prediction_count": len(missing),
        "full_denominator_count": denominator,
        "unsupported_no_prediction_count": int(unsupported_charge_count),
        "total_no_prediction_count": int(len(no_prediction_indices)) if args.full_denominator_count is not None else len(missing),
        "supported_to_full_index": str(args.supported_to_full_index.resolve()) if args.supported_to_full_index else None,
        "precision_coverage_curve": str(args.precision_coverage_output.resolve()) if args.precision_coverage_output else None,
        "peptide_precision_coverage_auc": float(auc(coverage, precision)),
        "peptide_precision_at_full_coverage": float(precision[-1]),
        "peptide_matches": int(peptide_matches.sum()),
        "aa_precision_at_full_coverage": aa_precision_at_full_coverage,
        "aa_recall_at_full_coverage": aa_recall_at_full_coverage,
        "n_aa_true": int(full_n_true),
        "n_aa_pred": int(full_n_pred),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
