"""Static peptide-ion pair-benchmark loading for embedding evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from src.collate_functions import pad_peaks
from src.embed_eval.data import REQUIRED_COLUMNS


PAIR_COLUMNS = {
    "pair_set",
    "species",
    "left_spectrum_id",
    "right_spectrum_id",
    "label",
}
SPECTRUM_ID_COLUMN = "spectrum_id"


def stable_rank(seed: int, *parts: object) -> bytes:
    """Return a process-independent deterministic ordering key."""
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


@dataclass(frozen=True)
class PairBenchmarkSelection:
    """Fixed protocol-aware subset summary for one pair-evaluation event."""

    source_pairs: int
    selected_pairs: int
    selected_spectra: int
    selected_pairs_by_set_species_label: dict[str, dict[str, dict[str, int]]]


class PairBenchmarkDataset(Dataset):
    """Spectra referenced by one deterministic peptide-ion pair selection."""

    def __init__(
        self,
        spectra_path: str | Path,
        pairs_path: str | Path,
        *,
        seed: int,
        max_pairs_per_partition_per_label: int | None,
    ) -> None:
        self.spectra_path = Path(spectra_path)
        self.pairs_path = Path(pairs_path)
        if not self.spectra_path.exists():
            raise FileNotFoundError(f"Pair benchmark spectra not found: {self.spectra_path}")
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Pair benchmark pairs not found: {self.pairs_path}")
        if (
            max_pairs_per_partition_per_label is not None
            and max_pairs_per_partition_per_label < 1
        ):
            raise ValueError(
                "max_pairs_per_partition_per_label must be positive or null."
            )

        pairs = pq.read_table(self.pairs_path)
        missing_pairs = sorted(PAIR_COLUMNS - set(pairs.column_names))
        if missing_pairs:
            raise ValueError(
                f"{self.pairs_path} is missing required pair columns: {missing_pairs}"
            )
        selected_pair_indices, self.selection = self._select_pair_indices(
            pairs,
            seed=seed,
            max_pairs_per_partition_per_label=max_pairs_per_partition_per_label,
        )
        if not selected_pair_indices:
            raise ValueError("No pair records remain after benchmark subsetting.")
        self.pairs = pairs.take(pa.array(selected_pair_indices, type=pa.int64()))

        spectra = pq.read_table(self.spectra_path)
        required_spectra = REQUIRED_COLUMNS | {SPECTRUM_ID_COLUMN, "species", "peptide_ion_id"}
        missing_spectra = sorted(required_spectra - set(spectra.column_names))
        if missing_spectra:
            raise ValueError(
                f"{self.spectra_path} is missing required pair-benchmark columns: "
                f"{missing_spectra}"
            )
        requested_ids = {
            str(value)
            for column_name in ("left_spectrum_id", "right_spectrum_id")
            for value in self.pairs[column_name].to_pylist()
        }
        spectrum_ids = [str(value) for value in spectra[SPECTRUM_ID_COLUMN].to_pylist()]
        selected_spectrum_indices = [
            index for index, spectrum_id in enumerate(spectrum_ids) if spectrum_id in requested_ids
        ]
        available_ids = {spectrum_ids[index] for index in selected_spectrum_indices}
        missing_ids = sorted(requested_ids - available_ids)
        if missing_ids:
            raise ValueError(
                f"{self.pairs_path} references {len(missing_ids)} spectra absent from "
                f"{self.spectra_path}; first={missing_ids[0]!r}"
            )
        self.table = spectra.take(pa.array(selected_spectrum_indices, type=pa.int64()))
        self.selection = PairBenchmarkSelection(
            source_pairs=self.selection.source_pairs,
            selected_pairs=self.selection.selected_pairs,
            selected_spectra=self.table.num_rows,
            selected_pairs_by_set_species_label=(
                self.selection.selected_pairs_by_set_species_label
            ),
        )

    @staticmethod
    def _select_pair_indices(
        pairs: pa.Table,
        *,
        seed: int,
        max_pairs_per_partition_per_label: int | None,
    ) -> tuple[list[int], PairBenchmarkSelection]:
        pair_sets = pairs["pair_set"].to_pylist()
        species = pairs["species"].to_pylist()
        labels = pairs["label"].to_pylist()
        groups: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, (pair_set, partition, label) in enumerate(
            zip(pair_sets, species, labels, strict=True)
        ):
            if pair_set is None or partition is None:
                continue
            label_int = int(label)
            if label_int not in {0, 1}:
                raise ValueError(f"Pair label must be 0 or 1, found {label!r}.")
            groups[(str(pair_set), str(partition))][label_int].append(index)

        selected = []
        selected_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for (pair_set, partition), label_groups in sorted(groups.items()):
            if set(label_groups) != {0, 1}:
                raise ValueError(
                    f"{pair_set}/{partition} must contain both labels, found "
                    f"{sorted(label_groups)}."
                )
            ordered_by_label = {}
            for label in (0, 1):
                ordered = sorted(
                    label_groups[label],
                    key=lambda index: stable_rank(
                        seed, pair_set, partition, label, index
                    ),
                )
                if max_pairs_per_partition_per_label is not None:
                    ordered = ordered[:max_pairs_per_partition_per_label]
                ordered_by_label[label] = ordered
            balanced_count = min(
                len(ordered_by_label[0]), len(ordered_by_label[1])
            )
            if balanced_count < 1:
                raise ValueError(
                    f"{pair_set}/{partition} has no balanced positive/negative pairs."
                )
            for label in (0, 1):
                kept = ordered_by_label[label][:balanced_count]
                selected.extend(kept)
                selected_counts[pair_set][partition][str(label)] = len(kept)

        selection = PairBenchmarkSelection(
            source_pairs=pairs.num_rows,
            selected_pairs=len(selected),
            selected_spectra=0,
            selected_pairs_by_set_species_label={
                pair_set: {
                    partition: dict(label_counts)
                    for partition, label_counts in partitions.items()
                }
                for pair_set, partitions in selected_counts.items()
            },
        )
        return selected, selection

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "mz_array": self.table["mz_array"][index].as_py(),
            "intensity_array": self.table["intensity_array"][index].as_py(),
            "precursor_mz": self.table["precursor_mz"][index].as_py(),
            "precursor_charge": self.table["precursor_charge"][index].as_py(),
            "peptide_id": self.table["peptide_ion_id"][index].as_py(),
            "partition_id": self.table["species"][index].as_py(),
            "spectrum_id": self.table[SPECTRUM_ID_COLUMN][index].as_py(),
            "_embedding_row_index": index,
        }



class StreamingPairBenchmarkDataset(IterableDataset):
    """Stream selected pair-spectrum peaks from Parquet record batches.

    Pair selection remains byte-for-byte equivalent to ``PairBenchmarkDataset``.
    Only pair metadata and selected spectrum metadata are retained in memory;
    ragged peak-list columns are streamed during model extraction.
    """

    def __init__(
        self,
        spectra_path: str | Path,
        pairs_path: str | Path,
        *,
        seed: int,
        max_pairs_per_partition_per_label: int | None,
        parquet_batch_size: int = 8192,
    ) -> None:
        self.spectra_path = Path(spectra_path)
        self.pairs_path = Path(pairs_path)
        self.parquet_batch_size = int(parquet_batch_size)
        if self.parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be positive.")
        if not self.spectra_path.exists() or not self.pairs_path.exists():
            raise FileNotFoundError("Pair benchmark spectra or pairs Parquet is missing.")

        pair_schema = pq.ParquetFile(self.pairs_path).schema_arrow
        pairs = pq.read_table(
            self.pairs_path,
            columns=[name for name in pair_schema.names if name in PAIR_COLUMNS],
        )
        missing_pairs = sorted(PAIR_COLUMNS - set(pairs.column_names))
        if missing_pairs:
            raise ValueError(f"{self.pairs_path} is missing required pair columns: {missing_pairs}")
        selected_indices, self.selection = PairBenchmarkDataset._select_pair_indices(
            pairs,
            seed=seed,
            max_pairs_per_partition_per_label=max_pairs_per_partition_per_label,
        )
        if not selected_indices:
            raise ValueError("No pair records remain after benchmark subsetting.")
        self.pairs = pairs.take(pa.array(selected_indices, type=pa.int64()))
        self._requested_ids = {
            str(value)
            for column_name in ("left_spectrum_id", "right_spectrum_id")
            for value in self.pairs[column_name].to_pylist()
        }

        metadata_columns = [
            SPECTRUM_ID_COLUMN,
            "species",
            "peptide_ion_id",
            "precursor_mz",
            "precursor_charge",
        ]
        spectra_file = pq.ParquetFile(self.spectra_path)
        missing_spectra = sorted(set(metadata_columns) - set(spectra_file.schema_arrow.names))
        if missing_spectra:
            raise ValueError(
                f"{self.spectra_path} is missing required pair-benchmark columns: {missing_spectra}"
            )
        metadata = pq.read_table(self.spectra_path, columns=metadata_columns)
        spectrum_ids = [str(value) for value in metadata[SPECTRUM_ID_COLUMN].to_pylist()]
        selected_spectrum_indices = [
            index for index, spectrum_id in enumerate(spectrum_ids)
            if spectrum_id in self._requested_ids
        ]
        available_ids = {spectrum_ids[index] for index in selected_spectrum_indices}
        missing_ids = sorted(self._requested_ids - available_ids)
        if missing_ids:
            raise ValueError(
                f"{self.pairs_path} references {len(missing_ids)} spectra absent from "
                f"{self.spectra_path}; first={missing_ids[0]!r}"
            )
        self.table = metadata.take(pa.array(selected_spectrum_indices, type=pa.int64()))
        self._requested_ids = frozenset(self._requested_ids)
        self.selection = PairBenchmarkSelection(
            source_pairs=self.selection.source_pairs,
            selected_pairs=self.selection.selected_pairs,
            selected_spectra=self.table.num_rows,
            selected_pairs_by_set_species_label=self.selection.selected_pairs_by_set_species_label,
        )
        offset = 0
        self._row_group_offsets = []
        for row_group in range(spectra_file.metadata.num_row_groups):
            self._row_group_offsets.append(offset)
            offset += spectra_file.metadata.row_group(row_group).num_rows

    def __len__(self) -> int:
        return self.selection.selected_spectra

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        parquet = pq.ParquetFile(self.spectra_path)
        columns = [
            "mz_array", "intensity_array", SPECTRUM_ID_COLUMN, "species",
            "peptide_ion_id", "precursor_mz", "precursor_charge",
        ]
        for row_group in range(worker_id, parquet.metadata.num_row_groups, worker_count):
            row_offset = self._row_group_offsets[row_group]
            batch_offset = 0
            for batch in parquet.iter_batches(
                batch_size=self.parquet_batch_size,
                row_groups=[row_group],
                columns=columns,
            ):
                for local_index, row in enumerate(batch.to_pylist()):
                    spectrum_id = str(row[SPECTRUM_ID_COLUMN])
                    if spectrum_id not in self._requested_ids:
                        continue
                    yield {
                        "mz_array": row["mz_array"],
                        "intensity_array": row["intensity_array"],
                        "precursor_mz": row["precursor_mz"],
                        "precursor_charge": row["precursor_charge"],
                        "peptide_id": row["peptide_ion_id"],
                        "partition_id": row["species"],
                        "spectrum_id": spectrum_id,
                        "_embedding_row_index": row_offset + batch_offset + local_index,
                    }
                batch_offset += batch.num_rows


def build_pair_eval_collate(global_args):
    """Build pair-benchmark preprocessing identical to DINO probe inputs."""
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


def load_manifest(manifest_path: str | Path | None) -> dict[str, object] | None:
    """Load an optional static-benchmark manifest for report provenance."""
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Pair benchmark manifest not found: {path}")
    return json.loads(path.read_text())
