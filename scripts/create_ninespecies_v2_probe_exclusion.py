#!/usr/bin/env python
"""Create a conservative NineSpecies exclusion manifest for end-AA probe reuse.

The historical end-AA linear probe sampled spectra from ``100_lance.lance``.
Its peptide strings and current NineSpecies V2 strings encode modifications
differently, so the exclusion is performed at peptide-backbone level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lance
import numpy as np
import pyarrow.parquet as pq

from src.embed_eval.sequence_normalization import (
    BACKBONE_NORMALIZATION_VERSION,
    normalize_peptide_backbone,
)


def _precursor_key(value: float, charge: int) -> tuple[int, int]:
    mz_bits = int(np.asarray([value], dtype="<f4").view("<u4")[0])
    return mz_bits, int(charge)


def _spectrum_fingerprint(row: dict[str, object]) -> bytes:
    """Match the probe's original ``100_lance`` spectrum without labels."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([float(row["precursor_mz"])], dtype="<f4").tobytes())
    digest.update(np.asarray([int(row["precursor_charge"])], dtype="<i4").tobytes())
    mz = np.asarray(row["mz_array"], dtype="<f4")
    intensity = np.asarray(row["intensity_array"], dtype="<f4")
    digest.update(np.asarray([mz.size], dtype="<i4").tobytes())
    digest.update(mz.tobytes())
    digest.update(intensity.tobytes())
    return digest.digest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_probe_fingerprints(probe_root: Path) -> tuple[set[bytes], set[tuple[int, int]], int]:
    fingerprints: set[bytes] = set()
    precursor_keys: set[tuple[int, int]] = set()
    count = 0
    columns = ["mz_array", "intensity_array", "precursor_mz", "precursor_charge"]
    for split in ("train", "val", "test"):
        for batch in pq.ParquetFile(probe_root / f"{split}.parquet").iter_batches(
            columns=columns, batch_size=4096
        ):
            for row in batch.to_pylist():
                fingerprints.add(_spectrum_fingerprint(row))
                precursor_keys.add(_precursor_key(row["precursor_mz"], row["precursor_charge"]))
                count += 1
    if count != len(fingerprints):
        raise ValueError("Expected each probe spectrum to have a unique fingerprint.")
    return fingerprints, precursor_keys, count


def create_manifest(args: argparse.Namespace) -> dict[str, object]:
    probe_root = Path(args.probe_root)
    source_path = Path(args.probe_source_lance)
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    fingerprints, precursor_keys, probe_spectra = _load_probe_fingerprints(probe_root)
    matched_backbones: set[str] = set()
    matched_rows = 0
    dataset = lance.dataset(str(source_path))
    columns = [
        "modified_sequence",
        "mz_array",
        "intensity_array",
        "precursor_mz",
        "precursor_charge",
    ]
    for batch in dataset.scanner(columns=columns, batch_size=65_536).to_batches():
        mz_values = batch.column(batch.schema.get_field_index("precursor_mz")).to_pylist()
        charges = batch.column(batch.schema.get_field_index("precursor_charge")).to_pylist()
        mz_column = batch.column(batch.schema.get_field_index("mz_array"))
        intensity_column = batch.column(batch.schema.get_field_index("intensity_array"))
        sequence_column = batch.column(batch.schema.get_field_index("modified_sequence"))
        for index, (precursor_mz, charge) in enumerate(zip(mz_values, charges, strict=True)):
            if _precursor_key(precursor_mz, charge) not in precursor_keys:
                continue
            candidate = {
                "mz_array": mz_column[index].as_py(),
                "intensity_array": intensity_column[index].as_py(),
                "precursor_mz": precursor_mz,
                "precursor_charge": charge,
            }
            if _spectrum_fingerprint(candidate) not in fingerprints:
                continue
            matched_rows += 1
            matched_backbones.add(normalize_peptide_backbone(sequence_column[index].as_py()))
    if matched_rows != probe_spectra:
        raise ValueError(
            f"Matched {matched_rows} probe spectra in {source_path}, expected {probe_spectra}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "builder": Path(__file__).name,
        "normalization_version": BACKBONE_NORMALIZATION_VERSION,
        "probe_root": str(probe_root),
        "probe_source_lance": str(source_path),
        "probe_source_lance_sha256": _sha256(source_path / "_latest.manifest"),
        "probe_spectra": probe_spectra,
        "matched_probe_source_rows": matched_rows,
        "excluded_backbone_count": len(matched_backbones),
        "excluded_backbones": sorted(matched_backbones),
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-root",
        default="/path/to/data/probing_datasets/end_aa_pred",
    )
    parser.add_argument(
        "--probe-source-lance",
        default="/path/to/source.lance",
    )
    parser.add_argument(
        "--output-path",
        default=(
            "/path/to/data/probing_datasets/"
            "ninespecies_v2_paper_test/probe_excluded_backbones.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(create_manifest(parse_args()), indent=2, sort_keys=True))
