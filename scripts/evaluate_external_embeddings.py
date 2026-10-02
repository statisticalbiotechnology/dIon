"""Evaluate an external embedding NPZ using dIon's shared peptide metrics.

The artifact is expected to contain ``embedding``, precursor metadata, and
label columns retained by ``scripts/embed_gleams_benchmark.py``. It never
loads TensorFlow or the external model itself.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from tqdm.auto import tqdm

from src.embed_eval.data import PeptideRetrievalDataset
from src.embed_eval.evaluator import resolve_evaluation_config
from src.embed_eval.extraction import CachedEmbeddings
from src.embed_eval.peptide_metrics import evaluate_peptide_embeddings


def _aligned_cache(
    artifact_path: Path,
    config: dict,
    *,
    peptide_id_key: str,
    partition_id_key: str,
    alignment_keys: list[str],
) -> tuple[CachedEmbeddings, PeptideRetrievalDataset]:
    """Order external embeddings exactly as the dIon retrieval dataset."""
    dataset_cfg = config["dataset"]
    selection_cfg = config.get("selection", {})
    dataset = PeptideRetrievalDataset(
        dataset_cfg["parquet_path"],
        peptide_id_column=dataset_cfg.get("peptide_id_column", peptide_id_key),
        partition_column=dataset_cfg.get("partition_column", partition_id_key),
        seed=int(selection_cfg.get("seed", 0)),
        max_peptides_per_partition=selection_cfg.get("max_peptides_per_partition"),
        max_spectra_per_peptide=selection_cfg.get("max_spectra_per_peptide"),
    )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        required = {"embedding", "precursor_mz", "precursor_charge"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise ValueError(f"{artifact_path} is missing external embedding arrays: {missing}")
        values = np.asarray(artifact["embedding"], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("External embedding array must be finite, non-empty, and shaped [N, D].")
        if any(len(artifact[key]) != values.shape[0] for key in required - {"embedding"}):
            raise ValueError("External embedding precursor metadata does not align with embeddings.")
        missing_artifact = sorted(set(alignment_keys) - set(artifact.files))
        missing_table = sorted(set(alignment_keys) - set(dataset.table.column_names))
        if not missing_artifact and not missing_table:
            # NpzFile access decompresses an archive member. Materialize each
            # key once; indexing artifact[key] per row would repeatedly decode it.
            artifact_columns = [artifact[key] for key in alignment_keys]
            artifact_keys = [
                tuple(str(column[index]) for column in artifact_columns)
                for index in tqdm(
                    range(values.shape[0]),
                    desc="Index external alignment keys",
                    unit="row",
                    dynamic_ncols=True,
                )
            ]
            if len(set(artifact_keys)) != len(artifact_keys):
                raise ValueError("External artifact alignment keys are not unique.")
            index_by_key = {key: index for index, key in enumerate(artifact_keys)}
            target_columns = [dataset.table[key].to_pylist() for key in alignment_keys]
            target_keys = [
                tuple(str(column[index]) for column in target_columns)
                for index in tqdm(
                    range(dataset.table.num_rows),
                    desc="Align retained benchmark rows",
                    unit="row",
                    dynamic_ncols=True,
                )
            ]
            try:
                artifact_indices = [index_by_key[key] for key in target_keys]
            except KeyError as exc:
                raise ValueError(
                    "Matched Parquet row is absent from external embedding artifact: "
                    f"{exc.args[0]!r}"
                ) from exc
        elif "export_row_index" in artifact.files:
            # materialize_external_matched_retrieval preserves the source table
            # order. PeptideRetrievalDataset then removes singleton peptide
            # groups, so map its selected source_row_index values back to the
            # retained external positions rather than requiring equal row counts.
            export_rows = np.asarray(artifact["export_row_index"], dtype=np.int64)
            if export_rows.ndim != 1 or len(np.unique(export_rows)) != len(export_rows):
                raise ValueError("export_row_index must be one-dimensional and unique.")
            if "source_row_index" in dataset.table.column_names:
                source_columns = ["source_row_index"]
                composite_key = "species" in dataset.table.column_names
                if composite_key:
                    source_columns.insert(0, "species")
                source_table = pq.read_table(dataset.path, columns=source_columns)
                if composite_key:
                    source_keys = list(zip(
                        (str(value) for value in source_table["species"].to_pylist()),
                        (int(value) for value in source_table["source_row_index"].to_pylist()),
                    ))
                    selected_keys = list(zip(
                        (str(value) for value in dataset.table["species"].to_pylist()),
                        (int(value) for value in dataset.table["source_row_index"].to_pylist()),
                    ))
                else:
                    source_keys = [int(value) for value in source_table["source_row_index"].to_pylist()]
                    selected_keys = [int(value) for value in dataset.table["source_row_index"].to_pylist()]
                if len(set(source_keys)) != len(source_keys):
                    raise ValueError("Matched Parquet source row identity is not unique.")
                source_position = {value: index for index, value in enumerate(source_keys)}
                try:
                    selected_positions = [source_position[value] for value in selected_keys]
                except KeyError as exc:
                    raise ValueError(
                        f"Selected retrieval row is absent from matched Parquet: {exc.args[0]!r}"
                    ) from exc
                external_position = {int(value): index for index, value in enumerate(export_rows)}
                try:
                    artifact_indices = [external_position[position] for position in selected_positions]
                except KeyError as exc:
                    raise ValueError(
                        "Selected matched-Parquet row is absent from the external artifact: "
                        f"position={exc.args[0]!r}"
                    ) from exc
            else:
                if values.shape[0] != dataset.table.num_rows:
                    raise ValueError(
                        "Positional external alignment requires source_row_index or one "
                        "embedding for every selected matched-Parquet row."
                    )
                artifact_indices = list(range(values.shape[0]))
            selected_indices = np.asarray(artifact_indices, dtype=np.int64)
            matched_mz = np.asarray(dataset.table["precursor_mz"].to_pylist(), dtype=np.float32)
            matched_charge = np.asarray(dataset.table["precursor_charge"].to_pylist(), dtype=np.float32)
            if not np.allclose(matched_mz, np.asarray(artifact["precursor_mz"])[selected_indices], rtol=0.0, atol=1e-5) or not np.array_equal(
                matched_charge, np.asarray(artifact["precursor_charge"], dtype=np.float32)[selected_indices]
            ):
                raise ValueError("Matched Parquet order does not agree with external precursor metadata.")
            print("Using validated retained export order for external alignment.")
        else:
            raise ValueError(
                "External alignment keys must exist in both artifact and matched Parquet, "
                "or the artifact must retain export_row_index from the standard exporter; "
                f"artifact_missing={missing_artifact}, table_missing={missing_table}."
            )
    indices = torch.tensor(artifact_indices, dtype=torch.long)
    values_tensor = torch.from_numpy(values)
    partition_ids = (
        [str(value) for value in dataset.table[dataset.partition_column].to_pylist()]
        if dataset.partition_column is not None
        else ["all"] * dataset.table.num_rows
    )
    return CachedEmbeddings(
        values=values_tensor[indices],
        peptide_ids=[
            str(value)
            for value in dataset.table[dataset.peptide_id_column].to_pylist()
        ],
        partition_ids=partition_ids,
        spectrum_ids=[str(index) for index in range(dataset.table.num_rows)],
        precursor_mz=torch.tensor(
            dataset.table["precursor_mz"].to_pylist(), dtype=torch.float32
        ),
        precursor_charges=torch.tensor(
            dataset.table["precursor_charge"].to_pylist(), dtype=torch.float32
        ),
    ), dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        default=None,
        help=(
            "Optional retained-row Parquet override. Disables config subsetting so "
            "every externally embedded spectrum is evaluated exactly once."
        ),
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--peptide-id-key", default="peptide_id")
    parser.add_argument("--partition-id-key", default="species")
    parser.add_argument(
        "--alignment-keys",
        nargs="+",
        default=["species", "source_row_index"],
        help="Unique raw-row keys used to reproduce dIon dataset ordering.",
    )
    parser.add_argument("--metrics", nargs="+", default=["cosine", "euclidean"])
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Metric device; auto uses CUDA when available.",
    )
    args = parser.parse_args()

    metric_device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    print(f"Loading external artifact and retained benchmark rows (metrics device: {metric_device}).")
    raw_config = yaml.safe_load(args.evaluation_config.read_text())
    if not isinstance(raw_config, dict) or "embedding_evaluation" not in raw_config:
        raise ValueError("--evaluation-config must contain embedding_evaluation.")
    config = resolve_evaluation_config(raw_config["embedding_evaluation"], "standalone")
    if args.dataset_parquet is not None:
        config["dataset"]["parquet_path"] = str(args.dataset_parquet)
        config.setdefault("selection", {})["max_peptides_per_partition"] = None
        config["selection"]["max_spectra_per_peptide"] = None
    cache, dataset = _aligned_cache(
        args.embeddings_npz,
        config,
        peptide_id_key=args.peptide_id_key,
        partition_id_key=args.partition_id_key,
        alignment_keys=args.alignment_keys,
    )
    print(
        f"Aligned {cache.values.shape[0]:,} external embeddings to "
        f"{dataset.table.num_rows:,} retained benchmark rows."
    )
    reports = {}
    for metric in args.metrics:
        print(f"Evaluating {metric} retrieval metrics.")
        metric_config = copy.deepcopy(config)
        metric_config["retrieval"]["metric"] = metric
        reports[metric] = evaluate_peptide_embeddings(
            cache.values,
            cache.peptide_ids,
            cache.partition_ids,
            metric_config,
            device=metric_device,
            precursor_mz=cache.precursor_mz,
            precursor_charges=cache.precursor_charges,
        )
    report = {
        "name": config["name"],
        "embedding_source": "external",
        "external_artifact": str(args.embeddings_npz.resolve()),
        "embedding_dimension": int(cache.values.shape[1]),
        "spectra": int(cache.values.shape[0]),
        "selection": {
            "source_rows": dataset.selection.source_rows,
            "selected_rows": dataset.selection.selected_rows,
            "selected_groups": dataset.selection.selected_groups,
            "selected_groups_by_partition": dataset.selection.selected_groups_by_partition,
            "alignment_keys": args.alignment_keys,
        },
        "metrics_by_distance": reports,
        "notes": [
            "All metrics use the same externally retained spectra and dIon row order.",
            "Cosine is the common DINO-vs-GLEAMS metric; Euclidean is GLEAMS' native embedding distance.",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote external embedding evaluation report: {args.output_report}")


if __name__ == "__main__":
    main()
