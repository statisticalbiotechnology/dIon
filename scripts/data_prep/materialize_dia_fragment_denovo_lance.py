#!/usr/bin/env python3
"""Materialize DIA-fragment target rows as a standard dIon de novo Lance set.

Each source row is one fixed DIA MS2 scan paired with one DIA-NN target
precursor. The normal de novo path can therefore train or decode a target
sequence under that target precursor, while retaining source scan and
per-peak target-fragment provenance for later component-selection analysis.
The source Parquet is immutable and is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lance
import pyarrow as pa
import pyarrow.parquet as pq


from src.data.unified_tokenizer import PeptideTokenizer


SPLITS = ("train", "val", "test")
DEFAULT_TOKENIZER = PROJECT_ROOT / "configs/tokenizers/pa11.json"
_UNIMOD_RESIDUE_DELTAS = {
    # DIA-NN's fixed cysteine carbamidomethyl annotation. PA1.1 represents
    # the same modification as a residue-attached numeric mass delta.
    "4": ("C", "+57.021"),
}
_UNIMOD_RESIDUE_RE = re.compile(r"(?P<residue>[A-Z])\(UniMod:(?P<id>\d+)\)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema() -> pa.Schema:
    return pa.schema([
        pa.field("peak_file", pa.string()),
        pa.field("scan_id", pa.int64()),
        pa.field("ms_level", pa.uint8()),
        pa.field("precursor_mz", pa.float64()),
        pa.field("precursor_charge", pa.int16()),
        pa.field("precursor_mass", pa.float64()),
        pa.field("mz_array", pa.list_(pa.float32())),
        pa.field("intensity_array", pa.list_(pa.float32())),
        pa.field("title", pa.string()),
        pa.field("seq", pa.string()),
        pa.field("source_modified_sequence", pa.string()),
        pa.field("split", pa.string()),
        pa.field("source_row_index", pa.int64()),
        pa.field("rt", pa.float64()),
        pa.field("isolation_low", pa.float64()),
        pa.field("isolation_high", pa.float64()),
        pa.field("qvalue", pa.float64()),
        pa.field("is_b", pa.list_(pa.bool_())),
        pa.field("is_y", pa.list_(pa.bool_())),
        pa.field("is_fragment", pa.list_(pa.bool_())),
        pa.field("orientation_mask", pa.list_(pa.bool_())),
    ])



def _normalize_dia_modified_sequence(text: str) -> str:
    """Translate only explicitly audited DIA-NN UniMod residue annotations."""
    def replace(match: re.Match[str]) -> str:
        residue = match.group("residue").upper()
        unimod_id = match.group("id")
        expected = _UNIMOD_RESIDUE_DELTAS.get(unimod_id)
        if expected is None:
            raise ValueError(f"unsupported DIA-NN UniMod:{unimod_id}")
        expected_residue, delta = expected
        if residue != expected_residue:
            raise ValueError(
                f"UniMod:{unimod_id} is only audited on {expected_residue}, got {residue}"
            )
        return f"{residue}{delta}"

    normalized = _UNIMOD_RESIDUE_RE.sub(replace, text)
    if "unimod:" in normalized.lower():
        raise ValueError(f"unparsed DIA-NN modification syntax: {text!r}")
    return normalized

def _canonical_sequence(value: Any, tokenizer: PeptideTokenizer) -> str:
    text = str(value or "").strip()
    text = _normalize_dia_modified_sequence(text)
    if not text:
        raise ValueError("empty sequence")
    tokens = tokenizer.preprocess_sequence(text)
    if not tokens:
        raise ValueError("empty token sequence")
    unknown = [token for token in tokens if token not in tokenizer.index]
    if unknown:
        raise ValueError(f"token absent from tokenizer: {unknown!r}")
    return "".join(tokens)


def _rows(batch: pa.RecordBatch, *, split: str, start_index: int, tokenizer: PeptideTokenizer) -> tuple[list[dict[str, Any]], Counter[str]]:
    columns = batch.to_pydict()
    retained, rejected = [], Counter()
    for offset, source_label in enumerate(columns["modified_seq"]):
        try:
            sequence = _canonical_sequence(source_label, tokenizer)
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        scan_id = int(columns["scan_id"][offset])
        precursor_mz = float(columns["precursor_mz"][offset])
        retained.append({
            "peak_file": "dia_fragment_v1",
            "scan_id": scan_id,
            "ms_level": 2,
            "precursor_mz": precursor_mz,
            "precursor_charge": int(columns["precursor_charge"][offset]),
            "precursor_mass": float(columns["precursor_mass"][offset]),
            "mz_array": [float(value) for value in columns["mz_array"][offset]],
            "intensity_array": [float(value) for value in columns["intensity_array"][offset]],
            "title": f"dia_scan={scan_id};target_precursor_mz={precursor_mz:.6f}",
            "seq": sequence,
            "source_modified_sequence": str(source_label),
            "split": split,
            "source_row_index": start_index + offset,
            "rt": float(columns["rt"][offset]),
            "isolation_low": float(columns["isolation_low"][offset]),
            "isolation_high": float(columns["isolation_high"][offset]),
            "qvalue": float(columns["qvalue"][offset]),
            "is_b": list(columns["is_b"][offset]),
            "is_y": list(columns["is_y"][offset]),
            "is_fragment": list(columns["is_fragment"][offset]),
            "orientation_mask": list(columns["orientation_mask"][offset]),
        })
    return retained, rejected

def _write_split(source: Path, destination: Path, *, split: str, tokenizer: PeptideTokenizer, batch_rows: int, max_rows: int | None, allow_unsupported_labels: bool) -> dict[str, Any]:
    parquet = pq.ParquetFile(source)
    mode, source_offset, written = "create", 0, 0
    rejected, scan_ids = Counter(), set()
    for batch in parquet.iter_batches(batch_size=batch_rows):
        if max_rows is not None and source_offset >= max_rows:
            break
        if max_rows is not None:
            batch = batch.slice(0, min(batch.num_rows, max_rows - source_offset))
        rows, batch_rejected = _rows(batch, split=split, start_index=source_offset, tokenizer=tokenizer)
        rejected.update(batch_rejected)
        source_offset += batch.num_rows
        if not rows:
            continue
        table = pa.Table.from_pylist(rows, schema=_schema())
        lance.write_dataset(table, str(destination), mode=mode, max_rows_per_file=100_000, max_rows_per_group=4096)
        mode = "append"
        written += table.num_rows
        scan_ids.update(row["scan_id"] for row in rows)
    rejected_rows = sum(rejected.values())
    if rejected and not allow_unsupported_labels:
        raise ValueError(f"{split} has tokenizer-incompatible labels: {dict(rejected)}")
    if written + rejected_rows != source_offset:
        raise RuntimeError(f"{split}: wrote {written} and rejected {rejected_rows} from {source_offset} source rows")
    return {"source_rows": source_offset, "written_rows": written, "rejected_rows": rejected_rows, "rejected_reasons": dict(rejected), "unique_scan_ids": len(scan_ids), "scan_ids": sorted(scan_ids)}



def _write_combined_test(
    source_paths: dict[str, Path],
    destination: Path,
    *,
    tokenizer: PeptideTokenizer,
    batch_rows: int,
    max_rows: int | None,
    allow_unsupported_labels: bool,
) -> dict[str, Any]:
    """Write every split as one oracle-precursor evaluation Lance dataset."""
    mode, written = "create", 0
    rejected, scan_ids, pair_keys = Counter(), set(), set()
    split_summaries = {}
    for split in SPLITS:
        parquet, source_offset, split_written = pq.ParquetFile(source_paths[split]), 0, 0
        for batch in parquet.iter_batches(batch_size=batch_rows):
            if max_rows is not None and source_offset >= max_rows:
                break
            if max_rows is not None:
                batch = batch.slice(0, min(batch.num_rows, max_rows - source_offset))
            rows, batch_rejected = _rows(
                batch, split=split, start_index=source_offset, tokenizer=tokenizer
            )
            rejected.update(batch_rejected)
            source_offset += batch.num_rows
            if not rows:
                continue
            for row in rows:
                pair_key = (
                    row["scan_id"],
                    row["source_modified_sequence"],
                    row["precursor_charge"],
                    row["precursor_mz"],
                )
                if pair_key in pair_keys:
                    raise RuntimeError(f"Duplicate DIA target pair: {pair_key!r}")
                pair_keys.add(pair_key)
            table = pa.Table.from_pylist(rows, schema=_schema())
            lance.write_dataset(
                table,
                str(destination),
                mode=mode,
                max_rows_per_file=100_000,
                max_rows_per_group=4096,
            )
            mode = "append"
            written += table.num_rows
            split_written += table.num_rows
            scan_ids.update(row["scan_id"] for row in rows)
        split_summaries[split] = {
            "source_rows": source_offset,
            "written_rows": split_written,
        }
    if rejected and not allow_unsupported_labels:
        raise ValueError(f"combined test has tokenizer-incompatible labels: {dict(rejected)}")
    source_rows = sum(summary["source_rows"] for summary in split_summaries.values())
    rejected_rows = sum(rejected.values())
    if written + rejected_rows != source_rows:
        raise RuntimeError(
            f"combined test: wrote {written} and rejected {rejected_rows} from {source_rows} rows"
        )

    return {
        "lance": str(destination),
        "source_rows": source_rows,
        "written_rows": written,
        "unique_target_pairs": len(pair_keys),
        "unique_scan_ids": len(scan_ids),
        "rejected_rows": rejected_rows,
        "rejected_reasons": dict(rejected),
        "split_summaries": split_summaries,
        "caveat": "One physical DIA scan may appear under multiple oracle target precursor queries by design.",
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--allow-unsupported-labels", action="store_true", help="Write only tokenizer-compatible target rows and record every excluded label in the manifest.")
    parser.add_argument(
        "--write-combined-test",
        action="store_true",
        help="Also write all peptide-disjoint source splits as all_splits_test.lance for oracle-precursor evaluation.",
    )
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--max-rows-per-split", type=int, default=None, help="Smoke only; never use for a study dataset.")
    args = parser.parse_args()
    if args.batch_rows < 1:
        raise ValueError("--batch-rows must be positive")
    source_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    source_paths = {split: source_root / f"{split}.parquet" for split in SPLITS}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DIA source split(s): {missing}")
    tokenizer = PeptideTokenizer.from_numeric_mass_delta_manifest(args.tokenizer_manifest)
    output_root.mkdir(parents=True)
    summaries = {}
    for split, source in source_paths.items():
        summaries[split] = _write_split(source, output_root / f"{split}.lance", split=split, tokenizer=tokenizer, batch_rows=args.batch_rows, max_rows=args.max_rows_per_split, allow_unsupported_labels=args.allow_unsupported_labels)
        print(f"{split}: {summaries[split]['written_rows']:,} rows")
    combined_test = None
    if args.write_combined_test:
        combined_test = _write_combined_test(
            source_paths,
            output_root / "all_splits_test.lance",
            tokenizer=tokenizer,
            batch_rows=args.batch_rows,
            max_rows=args.max_rows_per_split,
            allow_unsupported_labels=args.allow_unsupported_labels,
        )
        print(f"all_splits_test: {combined_test['written_rows']:,} rows")
    overlap = {}
    for left in SPLITS:
        for right in SPLITS:
            if left < right:
                overlap[f"{left}__{right}"] = len(set(summaries[left]["scan_ids"]).intersection(summaries[right]["scan_ids"]))
    for summary in summaries.values():
        del summary["scan_ids"]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_files": {split: {"path": str(path), "sha256": _sha256(path)} for split, path in source_paths.items()},
        "tokenizer_manifest": str(args.tokenizer_manifest.resolve()),
        "allow_unsupported_labels": args.allow_unsupported_labels,
        "tokenizer_manifest_sha256": _sha256(args.tokenizer_manifest),
        "dia_nn_unimod_translation": _UNIMOD_RESIDUE_DELTAS,
        "max_rows_per_split": args.max_rows_per_split,
        "splits": summaries,
        "cross_split_scan_id_overlap": overlap,
        "combined_test": combined_test,
        "caveat": "Rows are target-conditioned DIA scan/precursor pairs. Shared scans across peptide-disjoint splits must be handled explicitly for any learned readout claim.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
