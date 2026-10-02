import pyarrow as pa
import pyarrow.parquet as pq

from src.embed_eval.data import PeptideRetrievalDataset, StreamingPeptideRetrievalDataset


def test_streaming_retrieval_dataset_matches_materialized_rows(tmp_path):
    table = pa.table({
        "spectrum_id": ["a0", "a1", "b0", "b1"],
        "species": ["species"] * 4,
        "peptide_id": ["a", "a", "b", "b"],
        "precursor_mz": [400.0] * 4,
        "precursor_charge": [2] * 4,
        "mz_array": [[100.0], [101.0], [102.0], [103.0]],
        "intensity_array": [[1.0], [1.0], [1.0], [1.0]],
    })
    path = tmp_path / "retrieval.parquet"
    pq.write_table(table, path, row_group_size=2)
    kwargs = dict(
        peptide_id_column="peptide_id",
        partition_column="species",
        seed=0,
        max_peptides_per_partition=None,
        max_spectra_per_peptide=None,
    )
    materialized = PeptideRetrievalDataset(path, **kwargs)
    streamed = StreamingPeptideRetrievalDataset(path, **kwargs)
    assert streamed.selection == materialized.selection
    assert sorted(row["spectrum_id"] for row in streamed) == sorted(materialized.table[
        "spectrum_id"
    ].to_pylist())
