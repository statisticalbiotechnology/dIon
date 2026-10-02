"""Regression coverage for the streaming auxiliary-benchmark Lance converter."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts/data_prep/convert_auxiliary_benchmark_parquet_to_lance.py"
SPEC = importlib.util.spec_from_file_location("auxiliary_lance_converter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class AuxiliaryBenchmarkConverterTests(unittest.TestCase):
    def test_streamed_split_preserves_schema_rows_and_order(self):
        rows = [
            {
                "scan_id": index,
                "mz_array": [100.0 + index, 200.0 + index],
                "intensity_array": [0.1, 1.0],
                "label": index % 2,
            }
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "train.parquet"
            destination = Path(tmpdir) / "train.lance"
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, source)

            count = CONVERTER.convert_split(source, destination, batch_rows=2)
            output = lance.dataset(str(destination)).to_table()

        self.assertEqual(count, len(rows))
        self.assertEqual(output.schema, table.schema)
        self.assertEqual(output.to_pylist(), rows)

    def test_split_summary_records_source_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "split.parquet"
            pq.write_table(pa.Table.from_pylist([{"label": 1}]), source)
            summary = CONVERTER._split_summary(source)
        self.assertEqual(summary["source_rows"], 1)
        self.assertEqual(summary["source_row_groups"], 1)
        self.assertGreater(summary["source_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
