"""Score Casanovo mzTab predictions with dIon's canonical de novo matcher."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--mztab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Optional provenance only: checkpoint used to generate the mzTab. "
            "It never supplies matching masses or tokenization."
        ),
    )
    parser.add_argument(
        "--tokenizer-manifest",
        type=Path,
        default=DEFAULT_TOKENIZER_MANIFEST,
        help="dIon canonical numeric-mass tokenizer used for every prediction source.",
    )
    parser.add_argument(
        "--precision-coverage-output",
        type=Path,
        help="Optional CSV containing the score-descending peptide precision curve.",
    )
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Score the released prediction table when mzTab omits invalid spectra.",
    )
    parser.add_argument(
        "--full-denominator-count",
        type=int,
        help=(
            "Optional canonical total count. Rows absent because a baseline does not "
            "support their precursor charge are appended as zero-confidence peptide errors."
        ),
    )
    parser.add_argument(
        "--supported-to-full-index",
        type=Path,
        help=(
            "NumPy int64 mapping from supported-MGF ordinal to the canonical full "
            "test ordinal. Required with --full-denominator-count."
        ),
    )
    args = parser.parse_args()
    if (args.full_denominator_count is None) != (args.supported_to_full_index is None):
        parser.error("--full-denominator-count and --supported-to-full-index must be supplied together.")
    if args.full_denominator_count is not None and args.full_denominator_count < 1:
        parser.error("--full-denominator-count must be positive.")
    return args


def _mgf_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    current: str | None = None
    with path.open() as handle, tqdm(
        total=path.stat().st_size, unit="B", unit_scale=True,
        desc="Read Casanovo MGF labels",
    ) as progress:
        for line in handle:
            progress.update(len(line.encode("ascii")))
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                current = None
            elif line.startswith("SEQ="):
                current = line.removeprefix("SEQ=")
            elif line == "END IONS":
                if current is None:
                    raise ValueError(f"MGF record without SEQ in {path}.")
                sequences.append(current)
    return sequences


_SPECTRUM_INDEX = re.compile(r"(?:^|:)index=(\d+)$")


def _mztab_predictions(
    path: Path, expected_count: int, *, allow_missing: bool, use_proforma: bool,
    full_to_supported_index: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[int]]:
    header: list[str] | None = None
    predictions: list[str | None] = [None] * expected_count
    scores = np.empty(expected_count, dtype=np.float64)
    seen = np.zeros(expected_count, dtype=bool)
    with path.open() as handle, tqdm(
        desc="Parse Casanovo mzTab", unit="PSM",
    ) as progress:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if row[0] == "PSH":
                header = row[1:]
                continue
            if row[0] != "PSM":
                continue
            if header is None:
                raise ValueError(f"mzTab PSM before PSH header in {path}.")
            values = dict(zip(header, row[1:], strict=True))
            match = _SPECTRUM_INDEX.search(values["spectra_ref"])
            if match is None:
                raise ValueError(f"Unparseable spectra_ref: {values['spectra_ref']!r}")
            reported_index = int(match.group(1))
            if full_to_supported_index is None:
                index = reported_index
            else:
                if reported_index >= len(full_to_supported_index):
                    raise ValueError(f"Out-of-range canonical mzTab spectrum index: {reported_index}")
                index = int(full_to_supported_index[reported_index])
                if index < 0:
                    raise ValueError(f"mzTab reports unsupported canonical spectrum index: {reported_index}")
            if index >= expected_count or seen[index]:
                raise ValueError(f"Invalid or duplicate mzTab spectrum index: {reported_index}")
            predictions[index] = (
                values.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
                if use_proforma
                else values["sequence"]
            )
            if not predictions[index]:
                raise ValueError(f"mzTab has no usable peptide for spectrum index {index}")
            scores[index] = float(values["search_engine_score[1]"])
            seen[index] = True
            progress.update(1)
    missing = np.flatnonzero(~seen).tolist()
    if missing and not allow_missing:
        raise ValueError(
            f"mzTab is missing {len(missing)} MGF spectrum indices: {missing[:10]}. "
            "Pass --allow-missing-predictions only for a released prediction table "
            "that intentionally omits invalid spectra."
        )
    indices = np.flatnonzero(seen)
    return indices, [predictions[index] for index in indices], scores[indices], missing


def main() -> None:
    args = _parse_args()
    truth = _mgf_sequences(args.mgf)
    full_to_supported_index: np.ndarray | None = None
    if args.full_denominator_count is not None:
        mapping = np.load(args.supported_to_full_index, allow_pickle=False)
        if mapping.ndim != 1 or mapping.dtype.kind not in "iu" or len(mapping) != len(truth):
            raise ValueError("Invalid supported-to-full index mapping for the input MGF.")
        if np.any(mapping < 0) or np.any(mapping >= args.full_denominator_count):
            raise ValueError("Supported-to-full mapping contains an out-of-range canonical index.")
        if len(np.unique(mapping)) != len(mapping):
            raise ValueError("Supported-to-full mapping must be one-to-one.")
        full_to_supported_index = np.full(args.full_denominator_count, -1, dtype=np.int64)
        full_to_supported_index[mapping.astype(np.int64, copy=False)] = np.arange(len(mapping), dtype=np.int64)
    indices, predicted, scores, missing = _mztab_predictions(
        args.mztab,
        len(truth),
        allow_missing=args.allow_missing_predictions,
        use_proforma=args.checkpoint is not None,
        full_to_supported_index=full_to_supported_index,
    )

    tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(
        args.tokenizer_manifest,
        replace_isoleucine_with_leucine=True,
    )
    masses = tokenizer.residues
    truth_for_match = [tokenizer.preprocess_sequence(truth[index]) for index in indices]
    predicted_for_match = [tokenizer.preprocess_sequence(value) for value in predicted]
    scoring_mode = "dion_pa11_canonical_mass_matcher"

    order = np.argsort(-scores, kind="stable")
    matches = []
    n_true = n_pred = 0
    match_batch_size = 100_000
    for start in tqdm(
        range(0, len(order), match_batch_size),
        total=(len(order) + match_batch_size - 1) // match_batch_size,
        desc="Match Casanovo peptides", unit="batch",
    ):
        batch_order = order[start : start + match_batch_size]
        batch_matches, batch_n_true, batch_n_pred = evaluate.aa_match_batch(
            [truth_for_match[index] for index in batch_order],
            [predicted_for_match[index] for index in batch_order],
            masses,
        )
        matches.extend(batch_matches)
        n_true += batch_n_true
        n_pred += batch_n_pred
    peptide_matches = np.asarray([match[1] for match in matches], dtype=np.float64)
    ranked_source_indices = indices[order].astype(np.int64, copy=False)
    ranked_scores = scores[order]

    full_denominator_count = len(peptide_matches)
    unsupported_charge_count = 0
    no_prediction_full_indices = np.empty(0, dtype=np.int64)
    if args.full_denominator_count is not None:
        mapping = np.load(args.supported_to_full_index, allow_pickle=False)
        if mapping.ndim != 1 or mapping.dtype.kind not in "iu":
            raise ValueError("--supported-to-full-index must be a one-dimensional integer NumPy array.")
        if len(mapping) != len(truth):
            raise ValueError(
                f"Mapping length {len(mapping)} does not match supported MGF count {len(truth)}."
            )
        mapping = mapping.astype(np.int64, copy=False)
        if args.full_denominator_count < len(mapping):
            raise ValueError("Full denominator cannot be smaller than supported MGF count.")
        if np.any(mapping < 0) or np.any(mapping >= args.full_denominator_count):
            raise ValueError("Supported-to-full mapping contains an out-of-range canonical index.")
        if len(np.unique(mapping)) != len(mapping):
            raise ValueError("Supported-to-full mapping must be one-to-one.")
        full_denominator_count = args.full_denominator_count
        unsupported_charge_count = full_denominator_count - len(mapping)
        ranked_source_indices = mapping[ranked_source_indices]
        included = np.zeros(full_denominator_count, dtype=bool)
        included[ranked_source_indices] = True
        no_prediction_full_indices = np.flatnonzero(~included).astype(np.int64, copy=False)
        # This complement includes unsupported charge rows and supported inputs
        # Casanovo declined to emit after its own preprocessing. Both are
        # explicit zero-confidence peptide errors at the bottom of the ranking.
        peptide_matches = np.concatenate((peptide_matches, np.zeros(len(no_prediction_full_indices))))
        ranked_scores = np.concatenate((ranked_scores, np.full(len(no_prediction_full_indices), -np.inf)))
        ranked_source_indices = np.concatenate((ranked_source_indices, no_prediction_full_indices))
    elif missing:
        # A released decoder can decline spectra after its own peak filtering.
        # Preserve the MGF denominator rather than reporting only emitted PSMs.
        full_denominator_count = len(truth)
        missing_indices = np.asarray(missing, dtype=np.int64)
        peptide_matches = np.concatenate((peptide_matches, np.zeros(len(missing_indices))))
        ranked_scores = np.concatenate((ranked_scores, np.full(len(missing_indices), -np.inf)))
        ranked_source_indices = np.concatenate((ranked_source_indices, missing_indices))

    # The confidence-ranked curve above only needs emitted predictions plus
    # appended zero-confidence peptide errors. Full-coverage AA metrics also
    # need the true residues of released-decoder omissions, so score the
    # original MGF order with missing predictions represented as None.
    aa_precision_at_full_coverage: float | None = None
    aa_recall_at_full_coverage: float | None = None
    full_n_true = n_true
    full_n_pred = n_pred
    if args.full_denominator_count is None:
        all_predictions: list[str | list[str] | None] = [None] * len(truth)
        for index, peptide in zip(indices, predicted_for_match, strict=True):
            all_predictions[int(index)] = peptide
        all_truth = [tokenizer.preprocess_sequence(sequence) for sequence in truth]
        full_matches, full_n_true, full_n_pred = evaluate.aa_match_batch(
            all_truth, all_predictions, masses
        )
        aa_precision_at_full_coverage, aa_recall_at_full_coverage, _ = evaluate.aa_match_metrics(
            full_matches, full_n_true, full_n_pred
        )

    coverage = np.arange(1, len(peptide_matches) + 1, dtype=np.float64) / full_denominator_count
    precision = np.cumsum(peptide_matches) / np.arange(1, len(peptide_matches) + 1)
    if args.precision_coverage_output is not None:
        args.precision_coverage_output.parent.mkdir(parents=True, exist_ok=True)
        with args.precision_coverage_output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "rank", "source_spectrum_index", "confidence",
                    "peptide_match", "coverage", "peptide_precision",
                ),
            )
            writer.writeheader()
            for ranked_position, (source_index, confidence) in enumerate(
                tqdm(
                    zip(ranked_source_indices, ranked_scores, strict=True),
                    total=len(ranked_source_indices),
                    desc="Write precision-coverage curve", unit="row",
                ),
                start=0,
            ):
                writer.writerow(
                    {
                        "rank": ranked_position + 1,
                        "source_spectrum_index": int(source_index),
                        "confidence": repr(float(confidence)),
                        "peptide_match": int(peptide_matches[ranked_position]),
                        "coverage": repr(float(coverage[ranked_position])),
                        "peptide_precision": repr(float(precision[ranked_position])),
                    }
                )

    report = {
        "metric": "peptide_precision_coverage_auc",
        "metric_definition": "trapezoidal AUC of score-descending peptide precision versus coverage",
        "mgf": str(args.mgf.resolve()),
        "mztab": str(args.mztab.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "tokenizer_manifest": str(args.tokenizer_manifest.resolve()),
        "scoring_mode": scoring_mode,
        "precision_coverage_curve": (
            str(args.precision_coverage_output.resolve())
            if args.precision_coverage_output is not None
            else None
        ),
        "mgf_spectrum_count": len(truth),
        "predicted_spectrum_count": len(predicted),
        "missing_mgf_indices": missing,
        "full_denominator_count": full_denominator_count,
        "unsupported_no_prediction_count": int(unsupported_charge_count),
        "missing_supported_prediction_count": len(missing),
        "total_no_prediction_count": int(len(no_prediction_full_indices)) if args.full_denominator_count is not None else len(missing),
        "supported_to_full_index": (
            str(args.supported_to_full_index.resolve())
            if args.supported_to_full_index is not None else None
        ),
        "mz_tab_index_contract": "each reported spectra_ref ms_run[1]:index=i appears exactly once",
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
