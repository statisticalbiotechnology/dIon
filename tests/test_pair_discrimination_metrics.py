import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.embed_eval.pair_data import (
    PairBenchmarkDataset,
    StreamingPairBenchmarkDataset,
)
from src.embed_eval.pair_metrics import (
    evaluate_pair_discrimination,
    pair_discrimination_metrics,
)


def test_perfect_pair_separation_has_perfect_auc_ap_and_fnr():
    metrics = pair_discrimination_metrics(
        torch.tensor([0.10, 0.20, 0.80, 0.90]),
        torch.tensor([True, True, False, False]),
        fdr_levels=[0.01, 0.05, 0.10],
        purity_coverage_levels=[0.5, 1.0],
    )
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["fnr_at_balanced_fdr_0p01"] == 0.0
    assert metrics["threshold_at_balanced_fdr_0p01"] == pytest.approx(0.20)
    assert metrics["purity_at_positive_coverage"][0]["pair_purity"] == 1.0


def test_balanced_pair_fdr_uses_largest_valid_distance_threshold():
    metrics = pair_discrimination_metrics(
        torch.tensor([0.10, 0.20, 0.30, 0.40]),
        torch.tensor([True, False, True, False]),
        fdr_levels=[0.01],
        purity_coverage_levels=[0.5],
    )
    # Threshold 0.10 accepts one true pair and no false pair. At 0.20 the
    # balanced-pair FDR is 1/2, so one of two positives remains missed.
    assert metrics["fnr_at_balanced_fdr_0p01"] == 0.5
    assert metrics["threshold_at_balanced_fdr_0p01"] == pytest.approx(0.10)


def test_pair_evaluation_reports_hard_negative_availability():
    values = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )
    records = [
        {
            "pair_set": "same_charge_10ppm",
            "species": "species",
            "left_spectrum_id": "a0",
            "right_spectrum_id": "a1",
            "label": 1,
        },
        {
            "pair_set": "same_charge_10ppm",
            "species": "species",
            "left_spectrum_id": "a0",
            "right_spectrum_id": "b0",
            "label": 0,
        },
    ]
    report = evaluate_pair_discrimination(
        values,
        ["a0", "a1", "b0", "b1"],
        records,
        {"metric": "cosine", "balanced_pair_fdr_levels": [0.01]},
        availability_by_set_species={
            "same_charge_10ppm": {
                "species": {
                    "hard_negative_anchor_coverage": 0.25,
                    "hard_negative_pair_count": 1,
                }
            }
        },
        device=torch.device("cpu"),
    )
    metrics = report["pair_sets"]["same_charge_10ppm"]["partitions"]["species"]
    assert metrics["roc_auc"] == 1.0
    assert metrics["hard_negative_anchor_coverage"] == 0.25
    assert metrics["hard_negative_pair_count"] == 1.0


def test_pair_dataset_preserves_balanced_selection(tmp_path):
    spectra = pa.table(
        {
            "spectrum_id": ["a0", "a1", "b0", "b1"],
            "species": ["species"] * 4,
            "peptide_ion_id": ["a|z=2", "a|z=2", "b|z=2", "b|z=2"],
            "precursor_mz": [400.0] * 4,
            "precursor_charge": [2] * 4,
            "mz_array": [[100.0], [101.0], [102.0], [103.0]],
            "intensity_array": [[1.0], [1.0], [1.0], [1.0]],
        }
    )
    pairs = pa.table(
        {
            "pair_set": ["same_charge_random"] * 5,
            "species": ["species"] * 5,
            "left_spectrum_id": ["a0"] * 5,
            "right_spectrum_id": ["a1", "b0", "b1", "b0", "b1"],
            "label": [1, 0, 0, 0, 1],
        }
    )
    spectra_path = tmp_path / "spectra.parquet"
    pairs_path = tmp_path / "pairs.parquet"
    pq.write_table(spectra, spectra_path)
    pq.write_table(pairs, pairs_path)
    dataset = PairBenchmarkDataset(
        spectra_path,
        pairs_path,
        seed=0,
        max_pairs_per_partition_per_label=None,
    )
    counts = dataset.selection.selected_pairs_by_set_species_label[
        "same_charge_random"
    ]["species"]
    assert counts == {"0": 2, "1": 2}
    assert dataset.selection.selected_pairs == 4



def test_streaming_pair_dataset_matches_materialized_rows(tmp_path):
    spectra = pa.table(
        {
            "spectrum_id": ["a0", "a1", "b0", "b1"],
            "species": ["species"] * 4,
            "peptide_ion_id": ["a|z=2", "a|z=2", "b|z=2", "b|z=2"],
            "precursor_mz": [400.0] * 4,
            "precursor_charge": [2] * 4,
            "mz_array": [[100.0], [101.0], [102.0], [103.0]],
            "intensity_array": [[1.0], [1.0], [1.0], [1.0]],
        }
    )
    pairs = pa.table(
        {
            "pair_set": ["same_charge_random"] * 4,
            "species": ["species"] * 4,
            "left_spectrum_id": ["a0"] * 4,
            "right_spectrum_id": ["a1", "b0", "b1", "b0"],
            "label": [1, 0, 0, 1],
        }
    )
    spectra_path = tmp_path / "spectra.parquet"
    pairs_path = tmp_path / "pairs.parquet"
    pq.write_table(spectra, spectra_path, row_group_size=2)
    pq.write_table(pairs, pairs_path)
    materialized = PairBenchmarkDataset(
        spectra_path, pairs_path, seed=0, max_pairs_per_partition_per_label=None
    )
    streamed = StreamingPairBenchmarkDataset(
        spectra_path, pairs_path, seed=0, max_pairs_per_partition_per_label=None
    )
    assert streamed.selection == materialized.selection
    assert streamed.pairs.equals(materialized.pairs)
    assert sorted(row["spectrum_id"] for row in streamed) == sorted(materialized.table[
        "spectrum_id"
    ].to_pylist())
