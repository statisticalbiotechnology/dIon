"""Labelled spectrum data loading and deterministic evaluation subsetting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.collate_functions import pad_peaks


REQUIRED_COLUMNS = {
    "mz_array",
    "intensity_array",
    "precursor_mz",
    "precursor_charge",
}


def _stable_rank(seed: int, *parts: object) -> bytes:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).digest()


@dataclass(frozen=True)
class DatasetSelection:
    """Summary of the fixed subset used for one evaluation event."""

    source_rows: int
    selected_rows: int
    selected_groups: int
    selected_groups_by_partition: dict[str, int]


class PeptideRetrievalDataset(Dataset):
    """Parquet spectrum dataset used by dIon peptide retrieval evaluation."""

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        peptide_id_column: str,
        partition_column: str | None,
        seed: int,
        max_peptides_per_partition: int | None,
        max_spectra_per_peptide: int | None,
    ) -> None:
        self.path = Path(parquet_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Embedding evaluation Parquet not found: {self.path}")

        table = pq.read_table(self.path)
        required_columns = REQUIRED_COLUMNS | {peptide_id_column}
        if partition_column is not None:
            required_columns.add(partition_column)
        missing = sorted(required_columns - set(table.column_names))
        if missing:
            raise ValueError(
                f"{self.path} is missing required embedding-evaluation columns: {missing}"
            )
        if max_peptides_per_partition is not None and max_peptides_per_partition < 1:
            raise ValueError("max_peptides_per_partition must be positive or null.")
        if max_spectra_per_peptide is not None and max_spectra_per_peptide < 2:
            raise ValueError("max_spectra_per_peptide must be at least two or null.")

        peptide_ids = table[peptide_id_column].to_pylist()
        partition_ids = (
            table[partition_column].to_pylist()
            if partition_column is not None
            else ["all"] * table.num_rows
        )
        selected_indices, selection = self._select_indices(
            peptide_ids=peptide_ids,
            partition_ids=partition_ids,
            seed=seed,
            max_peptides_per_partition=max_peptides_per_partition,
            max_spectra_per_peptide=max_spectra_per_peptide,
        )
        if not selected_indices:
            raise ValueError("No repeated peptide groups remain after evaluation subsetting.")
        self.table = table.take(pa.array(selected_indices, type=pa.int64()))
        self.peptide_id_column = peptide_id_column
        self.partition_column = partition_column
        self.selection = selection

    @staticmethod
    def _select_indices(
        *,
        peptide_ids: list[Any],
        partition_ids: list[Any],
        seed: int,
        max_peptides_per_partition: int | None,
        max_spectra_per_peptide: int | None,
    ) -> tuple[list[int], DatasetSelection]:
        groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (peptide_id, partition_id) in enumerate(
            zip(peptide_ids, partition_ids, strict=True)
        ):
            if peptide_id is None or partition_id is None:
                continue
            peptide_key = str(peptide_id)
            partition_key = str(partition_id)
            if peptide_key:
                groups[partition_key][peptide_key].append(index)

        selected_indices = []
        selected_groups_by_partition = {}
        for partition_key in sorted(groups):
            repeated = {
                peptide_key: indices
                for peptide_key, indices in groups[partition_key].items()
                if len(indices) >= 2
            }
            peptide_keys = sorted(
                repeated,
                key=lambda peptide_key: _stable_rank(seed, partition_key, peptide_key),
            )
            if max_peptides_per_partition is not None:
                peptide_keys = peptide_keys[:max_peptides_per_partition]

            kept_groups = 0
            for peptide_key in peptide_keys:
                indices = sorted(
                    repeated[peptide_key],
                    key=lambda index: _stable_rank(
                        seed, partition_key, peptide_key, index
                    ),
                )
                if max_spectra_per_peptide is not None:
                    indices = indices[:max_spectra_per_peptide]
                if len(indices) < 2:
                    continue
                selected_indices.extend(indices)
                kept_groups += 1
            selected_groups_by_partition[partition_key] = kept_groups

        selection = DatasetSelection(
            source_rows=len(peptide_ids),
            selected_rows=len(selected_indices),
            selected_groups=sum(selected_groups_by_partition.values()),
            selected_groups_by_partition=selected_groups_by_partition,
        )
        return selected_indices, selection

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "mz_array": self.table["mz_array"][index].as_py(),
            "intensity_array": self.table["intensity_array"][index].as_py(),
            "precursor_mz": self.table["precursor_mz"][index].as_py(),
            "precursor_charge": self.table["precursor_charge"][index].as_py(),
            "peptide_id": self.table[self.peptide_id_column][index].as_py(),
            # Preserve selected-row identity through pad_peaks. It is needed
            # for deterministic interventions when a collate operation drops
            # spectra with too few retained peaks.
            "_embedding_row_index": index,
            "spectrum_id": (
                self.table["spectrum_id"][index].as_py()
                if "spectrum_id" in self.table.column_names
                else f"{self.path.name}:{index}"
            ),
        }
        if self.partition_column is not None:
            item["partition_id"] = self.table[self.partition_column][index].as_py()
        else:
            item["partition_id"] = "all"
        return item



class StreamingPeptideRetrievalDataset(IterableDataset):
    """Stream selected peak lists from Parquet without materializing them all.

    Selection intentionally uses the exact same metadata-only algorithm as
    :class:`PeptideRetrievalDataset`. Raw peak columns are read one Parquet
    record batch at a time, keeping full-corpus extraction memory-bounded.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        peptide_id_column: str,
        partition_column: str | None,
        seed: int,
        max_peptides_per_partition: int | None,
        max_spectra_per_peptide: int | None,
        parquet_batch_size: int = 8192,
    ) -> None:
        self.path = Path(parquet_path)
        self.peptide_id_column = peptide_id_column
        self.partition_column = partition_column
        self.parquet_batch_size = int(parquet_batch_size)
        if self.parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be positive.")

        metadata_columns = ["precursor_mz", "precursor_charge", peptide_id_column]
        if partition_column is not None:
            metadata_columns.append(partition_column)
        schema = pq.ParquetFile(self.path).schema_arrow
        if "spectrum_id" in schema.names:
            metadata_columns.append("spectrum_id")
        missing = sorted(set(metadata_columns) - set(schema.names))
        if missing:
            raise ValueError(
                f"{self.path} is missing required embedding-evaluation columns: {missing}"
            )
        # Metadata is compact enough to retain and is needed by precursor-only
        # baselines/interventions. Peak-list columns never enter this table.
        metadata = pq.read_table(self.path, columns=metadata_columns)
        peptide_ids = metadata[peptide_id_column].to_pylist()
        partition_ids = (
            metadata[partition_column].to_pylist()
            if partition_column is not None
            else ["all"] * metadata.num_rows
        )
        selected_indices, self.selection = PeptideRetrievalDataset._select_indices(
            peptide_ids=peptide_ids,
            partition_ids=partition_ids,
            seed=seed,
            max_peptides_per_partition=max_peptides_per_partition,
            max_spectra_per_peptide=max_spectra_per_peptide,
        )
        if not selected_indices:
            raise ValueError("No repeated peptide groups remain after evaluation subsetting.")
        self.table = metadata.take(pa.array(selected_indices, type=pa.int64()))
        self._selected_indices = frozenset(selected_indices)
        # Extraction streams source order for efficient Parquet reads, then
        # restores this legacy deterministic selection order in its cache.
        self._selection_rank_by_source_index = {
            source_index: rank for rank, source_index in enumerate(selected_indices)
        }
        parquet = pq.ParquetFile(self.path)
        offset = 0
        self._row_group_offsets = []
        for row_group in range(parquet.metadata.num_row_groups):
            self._row_group_offsets.append(offset)
            offset += parquet.metadata.row_group(row_group).num_rows

    def __len__(self) -> int:
        return self.selection.selected_rows

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        parquet = pq.ParquetFile(self.path)
        columns = [
            "mz_array",
            "intensity_array",
            "precursor_mz",
            "precursor_charge",
            self.peptide_id_column,
        ]
        if self.partition_column is not None:
            columns.append(self.partition_column)
        if "spectrum_id" in parquet.schema_arrow.names:
            columns.append("spectrum_id")

        # Assign whole row groups to workers so raw Parquet data is not read
        # redundantly by every worker.
        for row_group in range(worker_id, parquet.metadata.num_row_groups, worker_count):
            row_offset = self._row_group_offsets[row_group]
            batch_offset = 0
            for batch in parquet.iter_batches(
                batch_size=self.parquet_batch_size,
                row_groups=[row_group],
                columns=columns,
            ):
                for local_index, row in enumerate(batch.to_pylist()):
                    source_index = row_offset + batch_offset + local_index
                    if source_index not in self._selected_indices:
                        continue
                    yield {
                        "mz_array": row["mz_array"],
                        "intensity_array": row["intensity_array"],
                        "precursor_mz": row["precursor_mz"],
                        "precursor_charge": row["precursor_charge"],
                        "peptide_id": row[self.peptide_id_column],
                        "partition_id": (
                            row[self.partition_column]
                            if self.partition_column is not None
                            else "all"
                        ),
                        "spectrum_id": row.get("spectrum_id", f"{self.path.name}:{source_index}"),
                        "_embedding_row_index": source_index,
                    }
                batch_offset += batch.num_rows


def build_embedding_eval_collate(global_args):
    """Build the same peak preprocessing used for training and linear probes."""
    return partial(
        pad_peaks,
        max_peaks=global_args.max_peaks,
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
        filter_method=global_args.peak_filter_method,
        intensity_scaling=global_args.intensity_scaling,
        min_mz=global_args.min_mz,
        max_mz=global_args.max_mz,
        min_intensity=global_args.min_intensity,
        remove_precursor_tol=global_args.remove_precursor_tol,
        min_peaks=global_args.min_peaks,
    )
