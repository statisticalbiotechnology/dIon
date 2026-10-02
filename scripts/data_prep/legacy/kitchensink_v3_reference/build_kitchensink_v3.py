#!/usr/bin/env python3
"""Build the provenance-rich, no-peak-cap General vNext spectrum corpus.

The build is intentionally staged and resumable:

``select``
    Read source metadata only, enforce source quality gates, and retain a
    deterministic, source-balanced candidate pool.  This stage does not copy
    peak arrays.
``resolve``
    Treat MSKBv5 validation/test as frozen.  Remove candidate peptidoforms in
    those holdouts, remove spectra already represented in MSKBv5, strictly
    remove contradictory labels for the same physical spectrum, and assign
    remaining novel labels to sequence-disjoint splits.
``materialize``
    Re-read only selected source spectra and write full, untruncated peak
    arrays to ``train.lance``, ``val.lance`` and ``test.lance``.
``verify``
    Audit physical identity and peptide leakage across final splits.

All mutable artifacts live below an output root.  Raw source datasets are
never modified.  The final Lance schema retains the standard Pairwise columns
plus explicit project/run/scan/source provenance.

This is a General vNext builder, not a backwards-compatible Pairwise builder:
the emitted vocabulary manifest may include bare C and numeric modification
tokens absent from the current MSKB-only tokenizer.  No source modification is
silently erased.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


try:
    import lance
except ImportError as exc:  # pragma: no cover - depends on active environment
    raise SystemExit("This builder requires the `lance` package.") from exc


SCRIPT_VERSION = "2026-07-18.1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "public/config.public.json"

KNOWN_EXT_RE = re.compile(r"\.(?:raw|mzml|mzxml|mgf)$", re.IGNORECASE)
PXD_PREFIX_RE = re.compile(r"^pxd\d+_", re.IGNORECASE)
TRAILING_SCAN_RE = re.compile(r"(?:^|[=:])\s*(?P<scan>\d+)\s*$", re.IGNORECASE)
TITLE_SCAN_RE = re.compile(r"^(?P<run>.+?):scan:(?P<scan>\d+)$", re.IGNORECASE)
PXD_RE = re.compile(r"PXD\d+", re.IGNORECASE)
NUMERIC_DELTA_RE = re.compile(r"[+-]\d+(?:\.\d+)?")
RESIDUE_TOKEN_RE = re.compile(r"(?P<aa>[A-Z])(?P<mods>(?:[+-]\d+(?:\.\d+)?)*)")
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYO")


METADATA_SCHEMA = pa.schema(
    [
        pa.field("candidate_id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("source_path", pa.string()),
        pa.field("source_row", pa.int64()),
        pa.field("source_label", pa.string()),
        pa.field("seq", pa.string()),
        pa.field("peptide_length", pa.int32()),
        pa.field("project_accession", pa.string()),
        pa.field("msv_accession", pa.string()),
        pa.field("run_file", pa.string()),
        pa.field("physical_scan_id", pa.int64()),
        pa.field("base_physical_key", pa.string()),
        pa.field("physical_key", pa.string()),
        pa.field("precursor_mz", pa.float64()),
        pa.field("precursor_charge", pa.int32()),
        pa.field("source_quality", pa.float64()),
        pa.field("quality_metric", pa.string()),
        pa.field("experiment", pa.string()),
        pa.field("species", pa.string()),
        pa.field("enzyme_class", pa.string()),
        pa.field("is_chimeric", pa.bool_()),
        pa.field("priority", pa.uint64()),
        pa.field("source_priority", pa.int32()),
        pa.field("charge_bucket", pa.string()),
        pa.field("final_split", pa.string()),
        pa.field("split_origin", pa.string()),
    ]
)


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("peak_file", pa.string()),
        pa.field("scan_id", pa.int64()),
        pa.field("ms_level", pa.int64()),
        pa.field("precursor_mz", pa.float64()),
        pa.field("precursor_charge", pa.int64()),
        pa.field("mz_array", pa.list_(pa.float32())),
        pa.field("intensity_array", pa.list_(pa.float32())),
        pa.field("seq", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("project_accession", pa.string()),
        pa.field("msv_accession", pa.string()),
        pa.field("run_file", pa.string()),
        pa.field("physical_scan_id", pa.int64()),
        pa.field("source_path", pa.string()),
        pa.field("source_row", pa.int64()),
        pa.field("source_label", pa.string()),
        pa.field("source_quality", pa.float64()),
        pa.field("quality_metric", pa.string()),
        pa.field("experiment", pa.string()),
        pa.field("species", pa.string()),
        pa.field("enzyme_class", pa.string()),
        pa.field("is_chimeric", pa.bool_()),
        pa.field("physical_key", pa.string()),
        pa.field("split_origin", pa.string()),
    ]
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if not config.get("sources"):
        raise ValueError(f"No sources in config: {path}")
    return config


def _expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_pattern in patterns:
        pattern = os.path.expandvars(raw_pattern)
        if "$" in pattern:
            raise ValueError(
                f"Unset environment variable in source path pattern: {raw_pattern}"
            )
        matches = [Path(item) for item in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(Path(pattern))
        else:
            raise FileNotFoundError(f"No paths matched: {pattern}")
    return sorted(set(paths))


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    match = re.search(r"-?\d+", str(value).strip())
    return int(match.group()) if match else None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _truthy(value: Any) -> bool:
    text = _as_text(value).lower()
    return text not in {"", "0", "false", "none", "nan", "no"}


def _normalize_run(value: Any) -> str:
    """Match the run-stem behavior used by the physical-overlap audit."""
    text = Path(_as_text(value)).name
    if text.lower().endswith(".gz"):
        text = text[:-3]
    text = KNOWN_EXT_RE.sub("", text).lower()
    text = PXD_PREFIX_RE.sub("", text)
    return text


def _parse_scan(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = _as_text(value)
    if text.isdigit():
        return int(text)
    match = TRAILING_SCAN_RE.search(text)
    return int(match.group("scan")) if match else None


def _parse_title(value: Any) -> tuple[str, int] | None:
    match = TITLE_SCAN_RE.match(_as_text(value))
    if match is None:
        return None
    return match.group("run"), int(match.group("scan"))


def _physical_keys(run_file: str, scan: int | None, charge: int | None, mz: float | None) -> tuple[str, str]:
    run = _normalize_run(run_file)
    if not run or scan is None or charge is None or mz is None:
        return "", ""
    base = f"{run}|scan={scan}|z={charge}"
    # 0.01 Th is tighter than the 0.02 Th agreement established in the prior
    # MKB2/ProteomeTools audit, while still absorbing ordinary export rounding.
    return base, f"{base}|mz={mz:.2f}"


def _stable_u63(*parts: Any) -> int:
    value = "\x1f".join(_as_text(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _canonicalize_label(value: Any) -> str | None:
    """Canonicalize only known notation changes; never erase an unknown mod."""
    text = _as_text(value)
    if not text:
        return None
    text = text.replace(" ", "").replace("_", "")
    text = re.sub(r"M\(ox\)", "M+15.995", text, flags=re.IGNORECASE)
    text = re.sub(r"M\[Oxidation\]", "M+15.995", text, flags=re.IGNORECASE)
    text = re.sub(r"C\[Carbamidomethyl\]", "C+57.021", text, flags=re.IGNORECASE)
    text = re.sub(r"([NQ])\[Deamidated\]", r"\1+0.984", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Acetyl\]-", "+42.011", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Carbamyl\]-", "+43.006", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Ammonia-loss\]-", "-17.027", text, flags=re.IGNORECASE)
    text = text.upper()
    if any(mark in text for mark in "[](){};"):
        return None

    prefix_match = re.match(r"^(?:[+-]\d+(?:\.\d+)?)*", text)
    assert prefix_match is not None
    pos = prefix_match.end()
    residues = 0
    while pos < len(text):
        match = RESIDUE_TOKEN_RE.match(text, pos)
        if match is None or match.group("aa") not in VALID_RESIDUES:
            return None
        residues += 1
        pos = match.end()
    return text if residues else None


def _peptide_length(seq: str) -> int:
    return sum(1 for token in RESIDUE_TOKEN_RE.finditer(seq) if token.group("aa") in VALID_RESIDUES)


def _charge_bucket(charge: int, limits: dict[str, int]) -> str | None:
    for key in limits:
        if key.endswith("+"):
            if charge >= int(key[:-1]):
                return key
        elif "-" in key:
            lo, hi = (int(value) for value in key.split("-", 1))
            if lo <= charge <= hi:
                return key
        elif charge == int(key):
            return key
    return None


class _BucketTopK:
    """Retain the deterministically lowest-priority examples per charge bucket."""

    def __init__(self, limits: dict[str, int]):
        self.limits = {str(key): int(value) for key, value in limits.items()}
        self.heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
        self.seen_by_bucket: Counter[str] = Counter()

    def add(self, record: dict[str, Any]) -> None:
        bucket = record["charge_bucket"]
        self.seen_by_bucket[bucket] += 1
        limit = self.limits.get(bucket, 0)
        if limit <= 0:
            return
        heap = self.heaps[bucket]
        priority = int(record["priority"])
        item = (-priority, record["candidate_id"], record)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    def records(self) -> list[dict[str, Any]]:
        return [item[2] for heap in self.heaps.values() for item in heap]


def _apply_diversity_caps(records: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply deterministic project/run caps after stratified quota selection.

    ``uncapped_buckets`` is deliberately narrow: it is for source-local, rare
    charge buckets whose configured quotas already bound their total size. It
    prevents abundant lower-charge samples in the same project from consuming
    the entire project cap before scarce high-charge candidates are considered.
    """
    project_cap = int(spec.get("project_cap", 0) or 0)
    run_cap = int(spec.get("run_cap", 0) or 0)
    uncapped_buckets = {str(bucket) for bucket in spec.get("uncapped_buckets", ())}
    projects: Counter[str] = Counter()
    runs: Counter[tuple[str, str]] = Counter()
    retained: list[dict[str, Any]] = []
    dropped_project = 0
    dropped_run = 0
    retained_uncapped = 0
    for record in sorted(records, key=lambda row: (int(row["priority"]), row["candidate_id"])):
        if record["charge_bucket"] in uncapped_buckets:
            retained.append(record)
            retained_uncapped += 1
            continue
        project = record["project_accession"] or record["species"] or record["source_id"]
        run_key = (project, record["run_file"] or record["source_path"])
        if project_cap and projects[project] >= project_cap:
            dropped_project += 1
            continue
        if run_cap and runs[run_key] >= run_cap:
            dropped_run += 1
            continue
        projects[project] += 1
        runs[run_key] += 1
        retained.append(record)
    return retained, {
        "dropped_project_cap": dropped_project,
        "dropped_run_cap": dropped_run,
        "retained_uncapped_bucket": retained_uncapped,
    }


def _empty_metadata_row() -> dict[str, Any]:
    return {field.name: None for field in METADATA_SCHEMA}


def _make_record(
    *,
    spec: dict[str, Any],
    source_path: Path,
    source_row: int,
    source_label: Any,
    project_accession: Any,
    msv_accession: Any,
    run_file: Any,
    physical_scan_id: Any,
    precursor_mz: Any,
    precursor_charge: Any,
    source_quality: Any = None,
    quality_metric: str = "",
    experiment: Any = None,
    species: Any = None,
    enzyme_class: Any = None,
    is_chimeric: Any = None,
) -> tuple[dict[str, Any] | None, str | None]:
    seq = _canonicalize_label(source_label)
    if seq is None:
        return None, "invalid_label"
    mz = _maybe_float(precursor_mz)
    charge = _maybe_int(precursor_charge)
    scan = _parse_scan(physical_scan_id)
    if mz is None or charge is None or charge <= 0:
        return None, "invalid_precursor"
    bucket = _charge_bucket(charge, spec["charge_limits"])
    if bucket is None:
        return None, "outside_charge_plan"
    run = _as_text(run_file)
    base_key, physical_key = _physical_keys(run, scan, charge, mz)
    # Spectrum identity must be available before a candidate can reach resolve.
    if not base_key:
        return None, "missing_physical_identity"
    row = _empty_metadata_row()
    row.update(
        {
            "candidate_id": f"{spec['id']}:{_stable_u63(source_path, source_row, seq):016x}",
            "source_id": spec["id"],
            "source_dataset": spec["kind"],
            "source_path": str(source_path),
            "source_row": int(source_row),
            "source_label": _as_text(source_label),
            "seq": seq,
            "peptide_length": _peptide_length(seq),
            "project_accession": _as_text(project_accession),
            "msv_accession": _as_text(msv_accession),
            "run_file": run,
            "physical_scan_id": scan,
            "base_physical_key": base_key,
            "physical_key": physical_key,
            "precursor_mz": mz,
            "precursor_charge": charge,
            "source_quality": _maybe_float(source_quality),
            "quality_metric": quality_metric,
            "experiment": _as_text(experiment),
            "species": _as_text(species),
            "enzyme_class": _as_text(enzyme_class),
            "is_chimeric": None if is_chimeric is None else bool(is_chimeric),
            "priority": _stable_u63(spec["id"], run, scan, charge, f"{mz:.6f}", seq, spec["seed"]),
            "source_priority": int(spec.get("source_priority", 100)),
            "charge_bucket": bucket,
            "final_split": "",
            "split_origin": "",
        }
    )
    return row, None


def _iter_lance_or_parquet(path: Path, columns: list[str], batch_size: int) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream metadata rows without touching mz/intensity arrays."""
    row_offset = 0
    if path.name.endswith(".lance"):
        dataset = lance.dataset(str(path))
        available = [column for column in columns if column in dataset.schema.names]
        batches = dataset.to_batches(columns=available, batch_size=batch_size)
    elif path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        available = [column for column in columns if column in parquet.schema_arrow.names]
        batches = parquet.iter_batches(columns=available, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported Arrow path: {path}")
    for batch in batches:
        arrays = {name: batch.column(i).to_pylist() for i, name in enumerate(available)}
        for offset in range(batch.num_rows):
            yield row_offset + offset, {name: arrays[name][offset] for name in available}
        row_offset += batch.num_rows


def _iter_mgf_headers(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    in_spectrum = False
    ordinal = -1
    headers: dict[str, str] = {}
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                ordinal += 1
                in_spectrum = True
                headers = {}
            elif line == "END IONS" and in_spectrum:
                yield ordinal, headers
                in_spectrum = False
            elif in_spectrum and "=" in line:
                key, value = line.split("=", 1)
                headers[key.upper()] = value


def _parse_mgf_charge(value: Any) -> int | None:
    return _maybe_int(value)


def _parse_mgf_mz(value: Any) -> float | None:
    return _maybe_float(_as_text(value).split()[0] if _as_text(value) else None)


def _iter_mgf_selected_rows(path: Path, wanted: dict[int, dict[str, Any]]) -> Iterator[tuple[dict[str, Any], dict[str, str], list[float], list[float]]]:
    """Read peak arrays only for selected zero-based MGF ordinal rows."""
    in_spectrum = False
    ordinal = -1
    headers: dict[str, str] = {}
    mz_values: list[float] = []
    intensity_values: list[float] = []
    selected = False
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                ordinal += 1
                in_spectrum = True
                headers = {}
                mz_values = []
                intensity_values = []
                selected = ordinal in wanted
            elif line == "END IONS" and in_spectrum:
                if selected:
                    yield wanted[ordinal], headers, mz_values, intensity_values
                in_spectrum = False
            elif not in_spectrum:
                continue
            elif "=" in line:
                key, value = line.split("=", 1)
                headers[key.upper()] = value
            elif selected and line:
                fields = line.split()
                if len(fields) >= 2:
                    mz = _maybe_float(fields[0])
                    intensity = _maybe_float(fields[1])
                    if mz is not None and intensity is not None:
                        mz_values.append(mz)
                        intensity_values.append(intensity)


def _quality_pass_arrow(spec: dict[str, Any], row: dict[str, Any]) -> tuple[bool, str | None]:
    quality = spec.get("quality", {})
    if quality.get("drop_chimeric") and _truthy(row.get("chimeric")):
        return False, "chimeric"
    if "fdr_max" in quality:
        fdr = _maybe_float(row.get("fdr"))
        if fdr is None or fdr > float(quality["fdr_max"]):
            return False, "fdr"
    if "qvalue_max" in quality:
        qvalue = _maybe_float(row.get("Qvalue"))
        if qvalue is None or qvalue > float(quality["qvalue_max"]):
            return False, "qvalue"
    return True, None


def _select_arrow_source(spec: dict[str, Any], output: Path, batch_size: int) -> dict[str, Any]:
    paths = _expand_paths(spec["path_globs"])
    selector = _BucketTopK(spec["charge_limits"])
    stats: Counter[str] = Counter()
    columns_by_kind = {
        "mkb2_arrow": [
            "modified_sequence", "raw_file", "provenance_scan", "precursor_mass", "precursor_charge",
            "pxd", "msv", "fdr", "score", "chimeric", "experiment_name",
        ],
        "kingdoms_parquet": [
            "modified_sequence", "raw_file", "scan", "precursor_mass", "precursor_charge", "Qvalue",
            "chimeric", "name", "title",
        ],
        "bacteria_arrow": ["seq", "peak_file", "scan_id", "precursor_mz", "precursor_charge", "ms_level"],
        "highcharge_v3_arrow": ["seq", "peak_file", "scan_id", "precursor_mz", "precursor_charge", "ms_level", "project"],
        "usigrabber_arrow": ["seq", "peak_file", "scan_id", "precursor_mz", "precursor_charge", "ms_level"],
    }
    try:
        columns = columns_by_kind[spec["kind"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported Arrow source kind: {spec['kind']}") from exc

    for path in paths:
        desc = f"select {spec['id']} {path.name}"
        iterator = _iter_lance_or_parquet(path, columns, batch_size)
        for row_index, row in tqdm(iterator, desc=desc, unit="row", mininterval=2.0):
            stats["seen"] += 1
            okay, why = _quality_pass_arrow(spec, row)
            if not okay:
                stats[f"quality_reject_{why}"] += 1
                continue
            if spec["kind"] == "mkb2_arrow":
                record, why = _make_record(
                    spec=spec, source_path=path, source_row=row_index,
                    source_label=row.get("modified_sequence"), project_accession=row.get("pxd"),
                    msv_accession=row.get("msv"), run_file=row.get("raw_file"),
                    physical_scan_id=row.get("provenance_scan"), precursor_mz=row.get("precursor_mass"),
                    precursor_charge=row.get("precursor_charge"), source_quality=row.get("fdr"),
                    quality_metric="fdr", experiment=row.get("experiment_name"),
                    is_chimeric=_truthy(row.get("chimeric")),
                )
            elif spec["kind"] == "kingdoms_parquet":
                record, why = _make_record(
                    spec=spec, source_path=path, source_row=row_index,
                    source_label=row.get("modified_sequence"), project_accession="",
                    msv_accession="", run_file=row.get("raw_file"), physical_scan_id=row.get("scan"),
                    precursor_mz=row.get("precursor_mass"), precursor_charge=row.get("precursor_charge"),
                    source_quality=row.get("Qvalue"), quality_metric="qvalue", species=row.get("name"),
                    experiment=row.get("name"), is_chimeric=_truthy(row.get("chimeric")),
                )
            elif spec["kind"] == "bacteria_arrow":
                record, why = _make_record(
                    spec=spec, source_path=path, source_row=row_index,
                    source_label=row.get("seq"), project_accession=spec.get("project_accession", "PXD010000__PXD010613"),
                    msv_accession="", run_file=row.get("peak_file"), physical_scan_id=row.get("scan_id"),
                    precursor_mz=row.get("precursor_mz"), precursor_charge=row.get("precursor_charge"),
                    quality_metric="curated_lance", experiment="bacterial_diversity", is_chimeric=None,
                )
            elif spec["kind"] == "highcharge_v3_arrow":
                project = _as_text(row.get("project"))
                record, why = _make_record(
                    spec=spec, source_path=path, source_row=row_index,
                    source_label=row.get("seq"), project_accession=project,
                    msv_accession="", run_file=row.get("peak_file"), physical_scan_id=row.get("scan_id"),
                    precursor_mz=row.get("precursor_mz"), precursor_charge=row.get("precursor_charge"),
                    quality_metric="curated_lance", experiment=project, is_chimeric=None,
                )
            else:  # usigrabber_arrow
                peak_file = _as_text(row.get("peak_file"))
                project_match = PXD_RE.search(peak_file)
                record, why = _make_record(
                    spec=spec, source_path=path, source_row=row_index,
                    source_label=row.get("seq"), project_accession=project_match.group(0).upper() if project_match else "",
                    msv_accession="", run_file=peak_file, physical_scan_id=row.get("scan_id"),
                    precursor_mz=row.get("precursor_mz"), precursor_charge=row.get("precursor_charge"),
                    quality_metric="HCD_FTMS_gate", experiment="usiGrabber_HCD_FTMS", is_chimeric=None,
                )
            if record is None:
                stats[f"reject_{why}"] += 1
                continue
            stats["eligible"] += 1
            selector.add(record)

    selected, cap_stats = _apply_diversity_caps(selector.records(), spec)
    stats.update(cap_stats)
    stats["selected_before_caps"] = len(selector.records())
    stats["selected_after_caps"] = len(selected)
    _write_metadata(output, selected)
    report = {
        "id": spec["id"], "kind": spec["kind"], "paths": [str(path) for path in paths],
        "stats": dict(stats), "seen_by_charge_bucket": dict(selector.seen_by_bucket),
        "selected_by_charge_bucket": dict(Counter(record["charge_bucket"] for record in selected)),
    }
    return report


def _parse_pt_evidence_file(spec: dict[str, Any], evidence_path: Path) -> tuple[Path, list[dict[str, Any]], Counter[str]]:
    """Select one best PEP label per MGF ordinal before spectrum extraction."""
    mgf_path = evidence_path.parent.parent / f"{evidence_path.stem}.mgf"
    stats: Counter[str] = Counter()
    if not mgf_path.exists():
        stats["missing_mgf"] += 1
        return mgf_path, [], stats
    quality = spec.get("quality", {})
    best: dict[int, dict[str, Any]] = {}
    with evidence_path.open("rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            stats["evidence_rows"] += 1
            pep = _maybe_float(row.get("PEP"))
            if pep is None or pep > float(quality.get("pep_max", 0.01)):
                stats["reject_pep"] += 1
                continue
            if quality.get("drop_reverse") and _truthy(row.get("Reverse")):
                stats["reject_reverse"] += 1
                continue
            if quality.get("drop_contaminant") and _truthy(row.get("Potential contaminant")):
                stats["reject_contaminant"] += 1
                continue
            seq = _canonicalize_label(row.get("Modified sequence"))
            if seq is None:
                stats["reject_label"] += 1
                continue
            charge = _maybe_int(row.get("Charge"))
            mz = _maybe_float(row.get("m/z"))
            if charge is None or mz is None:
                stats["reject_precursor"] += 1
                continue
            bucket = _charge_bucket(charge, spec["charge_limits"])
            if bucket is None:
                stats["outside_charge_plan"] += 1
                continue
            # `MS/MS IDs` are MaxQuant table IDs, not MGF ordinals. The
            # single `MS/MS Scan Number` is the directly verifiable raw scan
            # in the exported MGF and is therefore the conservative choice.
            scan = _maybe_int(row.get("MS/MS Scan Number"))
            if scan is None or scan < 0:
                stats["missing_msms_scan_number"] += 1
                continue
            previous = best.get(scan)
            if previous is None or pep < previous["pep"]:
                best[scan] = {
                    "scan": scan, "seq": seq, "source_label": _as_text(row.get("Modified sequence")),
                    "pep": pep, "charge": charge, "mz": mz,
                    "experiment": _as_text(row.get("Experiment")),
                }
    stats["unique_scan_numbers"] = len(best)
    return mgf_path, list(best.values()), stats


def _hydrate_proteometools(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Fill physical scan/m/z from selected MGF headers before global dedup."""
    by_path: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_path[record["source_path"]][int(record["physical_scan_id"])] = record
    stats: Counter[str] = Counter()
    hydrated: list[dict[str, Any]] = []
    for path_text, wanted in tqdm(sorted(by_path.items()), desc="hydrate ProteomeTools MGF headers", unit="mgf"):
        path = Path(path_text)
        found: set[int] = set()
        for ordinal, headers in _iter_mgf_headers(path):
            header_scan = _parse_scan(headers.get("SCANS"))
            if header_scan is None:
                parsed_title = _parse_title(headers.get("TITLE", ""))
                header_scan = parsed_title[1] if parsed_title is not None else None
            record = wanted.get(header_scan) if header_scan is not None else None
            if record is None:
                continue
            found.add(header_scan)
            title = headers.get("TITLE", "")
            parsed = _parse_title(title)
            scan = _parse_scan(headers.get("SCANS"))
            if scan is None and parsed is not None:
                scan = parsed[1]
            run = parsed[0] if parsed is not None else path.name
            mz = _parse_mgf_mz(headers.get("PEPMASS"))
            charge = _parse_mgf_charge(headers.get("CHARGE"))
            if scan is None or mz is None or charge is None:
                stats["missing_mgf_identity"] += 1
                continue
            if charge != record["precursor_charge"] or abs(mz - record["precursor_mz"]) > 0.05:
                stats["evidence_mgf_precursor_mismatch"] += 1
                continue
            base, physical = _physical_keys(run, scan, charge, mz)
            if not base:
                stats["invalid_mgf_identity"] += 1
                continue
            record["run_file"] = run
            record["source_row"] = ordinal
            record["physical_scan_id"] = scan
            record["base_physical_key"] = base
            record["physical_key"] = physical
            record["priority"] = _stable_u63(
                record["source_id"], run, scan, charge, f"{mz:.6f}", record["seq"], record["candidate_id"]
            )
            hydrated.append(record)
        missing = set(wanted) - found
        stats["missing_selected_scan"] += len(missing)
    stats["hydrated"] = len(hydrated)
    return hydrated, stats


def _select_proteometools_source(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    root = Path(spec["root"])
    evidence_paths = sorted((root / "evidence").glob("*.evid"))
    if not evidence_paths:
        raise FileNotFoundError(f"No evidence files under {root / 'evidence'}")
    selector = _BucketTopK(spec["charge_limits"])
    stats: Counter[str] = Counter()
    for evidence_path in tqdm(evidence_paths, desc=f"select {spec['id']} evidence", unit="file"):
        mgf_path, entries, file_stats = _parse_pt_evidence_file(spec, evidence_path)
        stats.update(file_stats)
        for item in entries:
            # MGF ordinal is populated during hydration. The evidence scan
            # number is an actual raw scan and is used for the join.
            record = _empty_metadata_row()
            record.update(
                {
                    "candidate_id": f"{spec['id']}:{_stable_u63(mgf_path, item['scan'], item['seq']):016x}",
                    "source_id": spec["id"], "source_dataset": spec["kind"],
                    "source_path": str(mgf_path), "source_row": int(item["scan"]),
                    "source_label": item["source_label"], "seq": item["seq"],
                    "peptide_length": _peptide_length(item["seq"]),
                    "project_accession": spec["project_accession"], "msv_accession": "",
                    "run_file": mgf_path.stem, "physical_scan_id": int(item["scan"]),
                    "base_physical_key": f"pending:{mgf_path.name}:scan={item['scan']}",
                    "physical_key": f"pending:{mgf_path.name}:scan={item['scan']}",
                    "precursor_mz": item["mz"], "precursor_charge": item["charge"],
                    "source_quality": item["pep"], "quality_metric": "pep",
                    "experiment": item["experiment"], "species": "Homo_sapiens",
                    "enzyme_class": "ProteomeTools", "is_chimeric": False,
                    "priority": _stable_u63(spec["id"], mgf_path.name, item["scan"], item["seq"], spec["seed"]),
                    "source_priority": int(spec.get("source_priority", 100)),
                    "charge_bucket": _charge_bucket(item["charge"], spec["charge_limits"]),
                    "final_split": "", "split_origin": "",
                }
            )
            stats["eligible"] += 1
            selector.add(record)
    selected, cap_stats = _apply_diversity_caps(selector.records(), spec)
    hydrated, hydrate_stats = _hydrate_proteometools(selected)
    stats.update(cap_stats)
    stats.update(hydrate_stats)
    stats["selected_before_caps"] = len(selector.records())
    stats["selected_after_caps"] = len(selected)
    stats["selected_after_hydration"] = len(hydrated)
    _write_metadata(output, hydrated)
    return {
        "id": spec["id"], "kind": spec["kind"], "root": str(root), "stats": dict(stats),
        "seen_by_charge_bucket": dict(selector.seen_by_bucket),
        "selected_by_charge_bucket": dict(Counter(record["charge_bucket"] for record in hydrated)),
    }


def _select_nonenz_source(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    paths = _expand_paths(spec["path_globs"])
    selector = _BucketTopK(spec["charge_limits"])
    stats: Counter[str] = Counter()
    for path in paths:
        for ordinal, headers in tqdm(_iter_mgf_headers(path), desc=f"select {spec['id']} {path.name}", unit="spectrum", mininterval=2.0):
            stats["seen"] += 1
            title = headers.get("TITLE", "")
            parsed = _parse_title(title)
            run = parsed[0] if parsed else path.name
            scan = parsed[1] if parsed else _parse_scan(headers.get("SCANS"))
            record, why = _make_record(
                spec=spec, source_path=path, source_row=ordinal, source_label=headers.get("SEQ"),
                project_accession="MassIVEKB_nonenzymatic", msv_accession="", run_file=run,
                physical_scan_id=scan, precursor_mz=_parse_mgf_mz(headers.get("PEPMASS")),
                precursor_charge=_parse_mgf_charge(headers.get("CHARGE")), quality_metric="source_curated_mgf",
                experiment="Casanovo_multi_enzyme", enzyme_class="multi_enzyme", is_chimeric=None,
            )
            if record is None:
                stats[f"reject_{why}"] += 1
                continue
            stats["eligible"] += 1
            selector.add(record)
    selected, cap_stats = _apply_diversity_caps(selector.records(), spec)
    stats.update(cap_stats)
    stats["selected_before_caps"] = len(selector.records())
    stats["selected_after_caps"] = len(selected)
    _write_metadata(output, selected)
    return {
        "id": spec["id"], "kind": spec["kind"], "paths": [str(path) for path in paths],
        "stats": dict(stats), "seen_by_charge_bucket": dict(selector.seen_by_bucket),
        "selected_by_charge_bucket": dict(Counter(record["charge_bucket"] for record in selected)),
    }


def _write_metadata(path: Path, records: list[dict[str, Any]]) -> None:
    """Write selection metadata atomically so a refresh never destroys its prior file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=METADATA_SCHEMA)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd", row_group_size=100_000)
    temporary.replace(path)


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _stage_select(args: argparse.Namespace, config: dict[str, Any], output: Path) -> None:
    selection_root = output / "selection_raw"
    selection_root.mkdir(parents=True, exist_ok=True)
    manifest_path = selection_root / "manifest.json"
    if manifest_path.exists():
        manifests = json.loads(manifest_path.read_text())
        manifests["script_version"] = SCRIPT_VERSION
        manifests["config"] = str(args.config)
        manifests.setdefault("sources", {})
    else:
        manifests = {
            "script_version": SCRIPT_VERSION, "config": str(args.config), "sources": {},
        }

    requested_sources = set(args.source or [])
    known_sources = {str(source["id"]) for source in config["sources"]}
    unknown_sources = requested_sources - known_sources
    if unknown_sources:
        raise ValueError(f"Unknown --source value(s): {sorted(unknown_sources)}")

    for source in config["sources"]:
        if requested_sources and source["id"] not in requested_sources:
            continue
        if not source.get("enabled", False):
            manifests["sources"][source["id"]] = {"enabled": False, "rationale": source.get("rationale", "")}
            continue
        spec = dict(source)
        spec["seed"] = int(config["seed"])
        selected_path = selection_root / f"{spec['id']}.parquet"
        if selected_path.exists() and not args.force:
            print(f"[select] reusing {selected_path}")
            manifests["sources"][spec["id"]] = {"status": "reused", "path": str(selected_path)}
            continue
        print(f"[select] {spec['id']} ({spec['kind']})")
        if spec["kind"] in {"mkb2_arrow", "kingdoms_parquet", "bacteria_arrow", "highcharge_v3_arrow", "usigrabber_arrow"}:
            report = _select_arrow_source(spec, selected_path, int(config["batch_size"]))
        elif spec["kind"] == "proteometools":
            report = _select_proteometools_source(spec, selected_path)
        elif spec["kind"] == "nonenz_mgf":
            report = _select_nonenz_source(spec, selected_path)
        else:
            raise ValueError(f"Unsupported enabled source kind: {spec['kind']}")
        report["path"] = str(selected_path)
        report["rationale"] = spec.get("rationale", "")
        manifests["sources"][spec["id"]] = report
        _write_json(selection_root / f"{spec['id']}.summary.json", report)
        _assert_output_budget(output, float(config["max_output_gib"]))
    _write_json(manifest_path, manifests)
    print(f"[select] complete: {selection_root}")


def _iter_seed_records(config: dict[str, Any], split: str, batch_size: int) -> Iterator[dict[str, Any]]:
    path = Path(config["seed_dataset"]["root"]) / f"{split}.lance"
    dataset = lance.dataset(str(path))
    required = ["title", "seq", "precursor_mz", "precursor_charge"]
    if not set(required) <= set(dataset.schema.names):
        raise ValueError(f"MSKBv5 seed schema missing required fields in {path}")
    row_offset = 0
    for batch in dataset.to_batches(columns=required, batch_size=batch_size):
        arrays = {name: batch.column(i).to_pylist() for i, name in enumerate(required)}
        for offset in range(batch.num_rows):
            title = arrays["title"][offset]
            parsed = _parse_title(title)
            if parsed is None:
                raise ValueError(f"Cannot parse MSKBv5 title at {path}:{row_offset + offset}: {title!r}")
            seq = _canonicalize_label(arrays["seq"][offset])
            if seq is None:
                raise ValueError(f"Invalid MSKBv5 seed label at {path}:{row_offset + offset}")
            charge = _maybe_int(arrays["precursor_charge"][offset])
            mz = _maybe_float(arrays["precursor_mz"][offset])
            base, physical = _physical_keys(parsed[0], parsed[1], charge, mz)
            if not physical:
                raise ValueError(f"Missing MSKBv5 physical identity at {path}:{row_offset + offset}")
            yield {
                "seq": seq, "physical_key": physical, "base_physical_key": base,
                "source_path": str(path), "source_row": row_offset + offset,
            }
        row_offset += batch.num_rows


def _split_for_novel_sequence(seq: str, seed: int, split_config: dict[str, Any]) -> str:
    fraction = _stable_u63("general_vnext_split", seed, seq) / float(1 << 63)
    train = float(split_config["novel_train_fraction"])
    val = float(split_config["novel_val_fraction"])
    if fraction < train:
        return "train"
    if fraction < train + val:
        return "val"
    return "test"


def _stage_resolve(args: argparse.Namespace, config: dict[str, Any], output: Path) -> None:
    selection_root = output / "selection_raw"
    source_paths = sorted(path for path in selection_root.glob("*.parquet") if path.name != "final_selection.parquet")
    if not source_paths:
        raise FileNotFoundError(f"No raw selections under {selection_root}; run select first")
    print("[resolve] loading frozen MSKBv5 seed identities and labels")
    seed_labels: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    seed_base_physical: set[str] = set()
    for split in ("train", "val", "test"):
        for record in tqdm(_iter_seed_records(config, split, int(config["batch_size"])), desc=f"seed {split}", unit="spectrum", mininterval=2.0):
            seed_labels[split].add(record["seq"])
            seed_base_physical.add(record["base_physical_key"])

    print("[resolve] loading selected external metadata")
    candidates: list[dict[str, Any]] = []
    for path in tqdm(source_paths, desc="selected metadata", unit="file"):
        candidates.extend(_read_metadata(path))
    stats: Counter[str] = Counter()
    stats["selected_input"] = len(candidates)
    heldout = seed_labels["val"] | seed_labels["test"]
    filtered: list[dict[str, Any]] = []
    for row in candidates:
        if row["seq"] in heldout:
            stats["drop_seed_heldout_label"] += 1
            continue
        if row["base_physical_key"] in seed_base_physical:
            stats["drop_seed_physical_duplicate"] += 1
            continue
        filtered.append(row)
    candidates = filtered

    # Run+scan+charge is the conservative physical identity used by the
    # provenance audit. m/z remains recorded but cannot create a second row.
    by_base_physical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_base_physical[row["base_physical_key"]].append(row)
    resolved: list[dict[str, Any]] = []
    for base_physical_key, group in tqdm(by_base_physical.items(), desc="strict run-scan-charge resolve", unit="spectrum", mininterval=2.0):
        labels = {row["seq"] for row in group}
        if len(labels) != 1:
            stats["drop_conflicting_physical_rows"] += len(group)
            stats["drop_conflicting_physical_spectra"] += 1
            continue
        keeper = min(group, key=lambda row: (int(row["source_priority"]), int(row["priority"]), row["candidate_id"]))
        resolved.append(keeper)
        stats["drop_same_label_physical_duplicates"] += len(group) - 1

    for row in resolved:
        if row["seq"] in seed_labels["train"]:
            row["final_split"] = "train"
            row["split_origin"] = "seed_train_label"
        else:
            row["final_split"] = _split_for_novel_sequence(row["seq"], int(config["seed"]), config["split"])
            row["split_origin"] = "novel_sequence_hash"
    _write_metadata(selection_root / "final_selection.parquet", resolved)
    stats["after_physical_resolve"] = len(resolved)
    summary = {
        "script_version": SCRIPT_VERSION,
        "seed_dataset": config["seed_dataset"],
        "input_selection_files": [str(path) for path in source_paths],
        "counts": dict(stats),
        "final_by_source": dict(Counter(row["source_id"] for row in resolved)),
        "final_by_split": dict(Counter(row["final_split"] for row in resolved)),
        "final_by_charge": dict(sorted(Counter(int(row["precursor_charge"]) for row in resolved).items())),
        "candidate_peptidoforms": len({row["seq"] for row in resolved}),
        "heldout_seed_peptidoforms": len(heldout),
    }
    _write_json(selection_root / "resolve_summary.json", summary)
    seed_vocab_sequences = set().union(*seed_labels.values())
    _write_vocab_manifest(selection_root / "vnext_token_manifest.json", resolved, seed_vocab_sequences)
    _assert_output_budget(output, float(config["max_output_gib"]))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _sequence_tokens(seq: str) -> tuple[str, list[str], list[str]]:
    first_residue = re.search(r"[A-Z]", seq)
    prefix = seq[:first_residue.start()] if first_residue else ""
    nterm_deltas = NUMERIC_DELTA_RE.findall(prefix)
    residue_tokens = [match.group(0) for match in RESIDUE_TOKEN_RE.finditer(seq)]
    return prefix, nterm_deltas, residue_tokens

def _write_vocab_manifest(
    path: Path,
    records: list[dict[str, Any]],
    seed_sequences: set[str],
) -> None:
    residues: Counter[str] = Counter()
    nterm_prefixes: Counter[str] = Counter()
    nterm_deltas: Counter[str] = Counter()
    for record in records:
        nterm_prefix, delta_tokens, residue_tokens = _sequence_tokens(record["seq"])
        if nterm_prefix:
            nterm_prefixes[nterm_prefix] += 1
        nterm_deltas.update(delta_tokens)
        residues.update(residue_tokens)
    for seq in seed_sequences:
        nterm_prefix, delta_tokens, residue_tokens = _sequence_tokens(seq)
        if nterm_prefix:
            nterm_prefixes[nterm_prefix] += 1
        nterm_deltas.update(delta_tokens)
        residues.update(residue_tokens)
    _write_json(
        path,
        {
            "format": "numeric_mass_delta_v1",
            "notes": "Tokens are canonical residue+numeric-delta strings. Bare C is intentionally retained; do not use this manifest with an old MSKB-only tokenizer.",
            "residue_tokens": dict(sorted(residues.items())),
            "nterm_prefix_tokens": dict(sorted(nterm_prefixes.items())),
            "nterm_delta_tokens": dict(sorted(nterm_deltas.items())),
            "candidate_spectra": len(records),
            "seed_unique_peptidoforms": len(seed_sequences),
        },
    )


class _LanceAppender:
    def __init__(self, root: Path, max_output_gib: float):
        self.root = root
        self.max_output_gib = max_output_gib
        self.written: Counter[str] = Counter()
        self.created: set[str] = set()

    def append(self, split: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.root / "lance" / f"{split}.lance"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
        mode = "append" if split in self.created or path.exists() else "create"
        lance.write_dataset(table, str(path), mode=mode, max_rows_per_file=100_000, max_rows_per_group=1024)
        self.created.add(split)
        self.written[split] += len(rows)

    def assert_budget(self) -> None:
        _assert_output_budget(self.root, self.max_output_gib)


def _array_to_f32(value: Any) -> list[float]:
    if value is None:
        return []
    return np.asarray(value, dtype=np.float32).tolist()


def _output_row(
    meta: dict[str, Any],
    *,
    mz_array: Any,
    intensity_array: Any,
    ms_level: Any = 2,
) -> dict[str, Any]:
    return {
        "peak_file": meta["run_file"],
        "scan_id": int(meta["physical_scan_id"]),
        "ms_level": int(_maybe_int(ms_level) or 2),
        "precursor_mz": float(meta["precursor_mz"]),
        "precursor_charge": int(meta["precursor_charge"]),
        "mz_array": _array_to_f32(mz_array),
        "intensity_array": _array_to_f32(intensity_array),
        "seq": meta["seq"],
        "source_dataset": meta["source_dataset"],
        "source_id": meta["source_id"],
        "project_accession": meta["project_accession"],
        "msv_accession": meta["msv_accession"],
        "run_file": meta["run_file"],
        "physical_scan_id": int(meta["physical_scan_id"]),
        "source_path": meta["source_path"],
        "source_row": int(meta["source_row"]),
        "source_label": meta["source_label"],
        "source_quality": meta["source_quality"],
        "quality_metric": meta["quality_metric"],
        "experiment": meta["experiment"],
        "species": meta["species"],
        "enzyme_class": meta["enzyme_class"],
        "is_chimeric": meta["is_chimeric"],
        "physical_key": meta["physical_key"],
        "split_origin": meta["split_origin"],
    }


def _iter_seed_output_rows(config: dict[str, Any], split: str, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    path = Path(config["seed_dataset"]["root"]) / f"{split}.lance"
    dataset = lance.dataset(str(path))
    cols = ["title", "seq", "precursor_mz", "precursor_charge", "mz_array", "intensity_array", "ms_level"]
    row_offset = 0
    for batch in tqdm(dataset.to_batches(columns=cols, batch_size=batch_size), desc=f"materialize seed {split}", unit="batch"):
        arrays = {name: batch.column(i).to_pylist() for i, name in enumerate(cols)}
        rows: list[dict[str, Any]] = []
        for offset in range(batch.num_rows):
            title = arrays["title"][offset]
            parsed = _parse_title(title)
            if parsed is None:
                raise ValueError(f"Unparseable seed title {title!r}")
            seq = _canonicalize_label(arrays["seq"][offset])
            charge = _maybe_int(arrays["precursor_charge"][offset])
            mz = _maybe_float(arrays["precursor_mz"][offset])
            base, physical = _physical_keys(parsed[0], parsed[1], charge, mz)
            meta = _empty_metadata_row()
            meta.update(
                {
                    "source_id": "mskb_v5_seed", "source_dataset": "mskb_v5_seed",
                    "source_path": str(path), "source_row": row_offset + offset,
                    "source_label": _as_text(arrays["seq"][offset]), "seq": seq,
                    "project_accession": "", "msv_accession": "", "run_file": parsed[0],
                    "physical_scan_id": parsed[1], "base_physical_key": base, "physical_key": physical,
                    "precursor_mz": mz, "precursor_charge": charge, "source_quality": None,
                    "quality_metric": "MSKBv5_frozen_seed", "experiment": "MSKBv5",
                    "species": "", "enzyme_class": "mixed_tryptic_nontryptic", "is_chimeric": False,
                    "split_origin": "frozen_mskb_v5",
                }
            )
            rows.append(_output_row(meta, mz_array=arrays["mz_array"][offset], intensity_array=arrays["intensity_array"][offset], ms_level=arrays["ms_level"][offset]))
        row_offset += batch.num_rows
        yield rows


def _materialize_arrow_path(path: Path, wanted: dict[int, dict[str, Any]], appender: _LanceAppender, batch_size: int) -> None:
    if not wanted:
        return
    source_id = next(iter(wanted.values()))["source_id"]
    if path.name.endswith(".lance"):
        dataset = lance.dataset(str(path))
        cols = [name for name in ["mz_array", "intensity_array", "ms_level"] if name in dataset.schema.names]
        row_offset = 0
        batches = dataset.to_batches(columns=cols, batch_size=batch_size)
    elif path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        cols = [name for name in ["mz_array", "intensity_array", "ms_level"] if name in parquet.schema_arrow.names]
        row_offset = 0
        batches = parquet.iter_batches(columns=cols, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported materialization path: {path}")
    for batch in tqdm(batches, desc=f"materialize {source_id} {path.name}", unit="batch", mininterval=2.0):
        arrays = {name: batch.column(i).to_pylist() for i, name in enumerate(cols)}
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for offset in range(batch.num_rows):
            meta = wanted.get(row_offset + offset)
            if meta is None:
                continue
            by_split[meta["final_split"]].append(
                _output_row(
                    meta,
                    mz_array=arrays.get("mz_array", [None] * batch.num_rows)[offset],
                    intensity_array=arrays.get("intensity_array", [None] * batch.num_rows)[offset],
                    ms_level=arrays.get("ms_level", [2] * batch.num_rows)[offset],
                )
            )
        for split, rows in by_split.items():
            appender.append(split, rows)
        row_offset += batch.num_rows


def _materialize_mgf_path(path: Path, wanted: dict[int, dict[str, Any]], appender: _LanceAppender) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for meta, _headers, mz_array, intensity_array in tqdm(_iter_mgf_selected_rows(path, wanted), desc=f"materialize MGF {path.name}", unit="spectrum", mininterval=2.0):
        by_split[meta["final_split"]].append(_output_row(meta, mz_array=mz_array, intensity_array=intensity_array, ms_level=2))
        if len(by_split[meta["final_split"]]) >= 1024:
            appender.append(meta["final_split"], by_split.pop(meta["final_split"]))
    for split, rows in by_split.items():
        appender.append(split, rows)


def _stage_materialize(args: argparse.Namespace, config: dict[str, Any], output: Path) -> None:
    final_path = output / "selection_raw" / "final_selection.parquet"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing {final_path}; run resolve first")
    lance_root = output / "lance"
    if lance_root.exists() and any(lance_root.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to append into existing output {lance_root}; use --force after removing it")
    if args.force and lance_root.exists():
        shutil.rmtree(lance_root)
    appender = _LanceAppender(output, float(config["max_output_gib"]))
    for split in ("train", "val", "test"):
        for rows in _iter_seed_output_rows(config, split, int(config["batch_size"])):
            appender.append(split, rows)
        appender.assert_budget()

    records = _read_metadata(final_path)
    wanted_by_path: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        wanted_by_path[record["source_path"]][int(record["source_row"])] = record
    for path_text, wanted in tqdm(sorted(wanted_by_path.items()), desc="candidate source files", unit="file"):
        path = Path(path_text)
        if path.suffix == ".mgf":
            _materialize_mgf_path(path, wanted, appender)
        else:
            _materialize_arrow_path(path, wanted, appender, int(config["batch_size"]))
        appender.assert_budget()
    summary = {"written_rows": dict(appender.written), "lance_root": str(lance_root)}
    _write_json(output / "materialize_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _stage_verify(args: argparse.Namespace, config: dict[str, Any], output: Path) -> None:
    lance_root = output / "lance"
    split_paths = {split: lance_root / f"{split}.lance" for split in ("train", "val", "test")}
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final Lance split(s): {missing}")

    physical_seen: set[str] = set()
    full_spectrum_seen: set[str] = set()
    run_scan_seen: dict[str, tuple[str, str, int, float, str]] = {}
    seq_by_split: dict[str, set[str]] = {split: set() for split in split_paths}
    summary: dict[str, Any] = {"splits": {}, "intersections": {}}
    global_physical_dups = 0
    global_full_spectrum_dups = 0
    ambiguous_run_scan_rows = 0
    ambiguous_by_source_pair: Counter[str] = Counter()
    ambiguous_examples: list[dict[str, Any]] = []

    def spectrum_fingerprint(mz_array: Any, intensity_array: Any) -> str:
        """Exact spectrum fingerprint over paired float32 m/z and intensity arrays."""
        mz = np.asarray(mz_array, dtype=np.float32)
        intensity = np.asarray(intensity_array, dtype=np.float32)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray([len(mz), len(intensity)], dtype=np.int64).tobytes())
        digest.update(mz.tobytes())
        digest.update(intensity.tobytes())
        return digest.hexdigest()

    columns = [
        "physical_key", "run_file", "physical_scan_id", "seq", "source_id",
        "precursor_charge", "precursor_mz", "mz_array", "intensity_array",
    ]
    verify_batch_size = max(4096, int(config["batch_size"]))
    for split, path in split_paths.items():
        dataset = lance.dataset(str(path))
        source_counts: Counter[str] = Counter()
        charge_counts: Counter[int] = Counter()
        peaks: list[int] = []
        split_physical_dups = 0
        split_full_spectrum_dups = 0
        for batch in tqdm(
            dataset.to_batches(columns=columns, batch_size=verify_batch_size),
            desc=f"verify {split}", unit="batch", mininterval=2.0,
        ):
            values = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
            for idx in range(batch.num_rows):
                physical = values["physical_key"][idx]
                source = values["source_id"][idx]
                charge = int(values["precursor_charge"][idx])
                mz = float(values["precursor_mz"][idx])
                seq = values["seq"][idx]
                base = f"{_normalize_run(values['run_file'][idx])}|scan={int(values['physical_scan_id'][idx])}"
                if physical in physical_seen:
                    global_physical_dups += 1
                    split_physical_dups += 1
                physical_seen.add(physical)

                fingerprint = spectrum_fingerprint(values["mz_array"][idx], values["intensity_array"][idx])
                if fingerprint in full_spectrum_seen:
                    global_full_spectrum_dups += 1
                    split_full_spectrum_dups += 1
                full_spectrum_seen.add(fingerprint)

                previous = run_scan_seen.get(base)
                current = (source, physical, charge, mz, seq)
                if previous is None:
                    run_scan_seen[base] = current
                else:
                    ambiguous_run_scan_rows += 1
                    ambiguous_by_source_pair[f"{previous[0]}|{source}"] += 1
                    if len(ambiguous_examples) < 20:
                        ambiguous_examples.append(
                            {
                                "run_scan": base,
                                "first_source": previous[0],
                                "first_charge": previous[2],
                                "first_mz": previous[3],
                                "second_source": source,
                                "second_charge": charge,
                                "second_mz": mz,
                            }
                        )
                seq_by_split[split].add(seq)
                source_counts[source] += 1
                charge_counts[charge] += 1
                peaks.append(len(values["mz_array"][idx]))
        summary["splits"][split] = {
            "rows": int(dataset.count_rows()),
            "unique_peptidoforms": len(seq_by_split[split]),
            "duplicate_physical_rows_within_or_previous": split_physical_dups,
            "duplicate_exact_spectrum_rows_within_or_previous": split_full_spectrum_dups,
            "source_counts": dict(source_counts),
            "charge_counts": dict(sorted(charge_counts.items())),
            "peak_count": {
                "min": int(np.min(peaks)),
                "median": float(np.median(peaks)),
                "p90": float(np.quantile(peaks, 0.9)),
                "p99": float(np.quantile(peaks, 0.99)),
                "max": int(np.max(peaks)),
            },
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = seq_by_split[left] & seq_by_split[right]
        summary["intersections"][f"{left}_{right}_peptidoforms"] = len(overlap)
        if overlap:
            summary["intersections"][f"{left}_{right}_examples"] = sorted(overlap)[:20]
    summary["global_duplicate_physical_rows"] = global_physical_dups
    summary["global_duplicate_exact_spectrum_rows"] = global_full_spectrum_dups
    # Bare run+scan is a namespace diagnostic, not a physical identity across
    # MGF and raw sources. Physical uniqueness is asserted above by precursor
    # identity and exact paired m/z/intensity-array content.
    summary["global_ambiguous_run_scan_rows"] = ambiguous_run_scan_rows
    summary["ambiguous_run_scan_by_source_pair"] = dict(ambiguous_by_source_pair)
    summary["ambiguous_run_scan_examples"] = ambiguous_examples
    _write_json(output / "verification_summary.json", summary)
    _assert_output_budget(output, float(config["max_output_gib"]))
    if global_physical_dups:
        raise RuntimeError(f"Verification failed: {global_physical_dups} duplicate physical keys")
    if global_full_spectrum_dups:
        raise RuntimeError(
            f"Verification failed: {global_full_spectrum_dups} duplicate exact spectrum fingerprints"
        )
    if any(summary["intersections"][key] for key in ("train_val_peptidoforms", "train_test_peptidoforms", "val_test_peptidoforms")):
        raise RuntimeError("Verification failed: peptidoform overlap across final splits")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _directory_size_gib(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total / (1024 ** 3)


def _assert_output_budget(output: Path, max_gib: float) -> None:
    size_gib = _directory_size_gib(output)
    print(f"[storage] output={output} size={size_gib:.2f} GiB budget={max_gib:.2f} GiB")
    if size_gib > max_gib:
        raise RuntimeError(f"Output budget exceeded: {size_gib:.2f} GiB > {max_gib:.2f} GiB")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "resolve", "materialize", "verify", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Rebuild an already-present stage; materialize removes only OUTPUT/lance.")
    parser.add_argument(
        "--source",
        action="append",
        help="For select only, rebuild one configured source ID. May be passed more than once.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _read_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "build_config.json", config)
    stages = ("select", "resolve", "materialize", "verify") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "select":
            _stage_select(args, config, output)
        elif stage == "resolve":
            _stage_resolve(args, config, output)
        elif stage == "materialize":
            _stage_materialize(args, config, output)
        elif stage == "verify":
            _stage_verify(args, config, output)


if __name__ == "__main__":
    main()
