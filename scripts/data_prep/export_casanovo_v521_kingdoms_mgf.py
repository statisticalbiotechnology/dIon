#!/usr/bin/env python3
"""Export explicit charge-supported Kingdoms de novo rows as annotated MGF.

Every source row remains addressable through ``supported_to_full_index``. The
Casanovo v5.2.1 default preserves its strict charge-1--4 contract, while other
released decoders can request their own explicit charge domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import lance
import numpy as np
from tqdm.auto import tqdm

REQUIRED_COLUMNS = (
    "peak_file",
    "scan_id",
    "precursor_mz",
    "precursor_charge",
    "mz_array",
    "intensity_array",
    "seq",
    "species",
    "source_row_index",
)
DEFAULT_SOURCE = Path(
    "/path/to/data/denovo_kingdoms/run_disjoint_v1/"
    "run_disjoint_v1_species_cap100k/test.lance"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lance", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-charge", type=int, default=1)
    parser.add_argument("--max-charge", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--model-label",
        default="casanovo_v5_2_1",
        help="Recorded in the manifest; defaults to the original Casanovo export.",
    )
    parser.add_argument(
        "--output-mgf-name",
        default="test_charge1to4_for_casanovo_v5_2_1.mgf",
        help="Basename for the emitted MGF within --output-root.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Bounded smoke-only export. Omit for the canonical complete export.",
    )
    return parser.parse_args()


def ensure_valid_record(
    *,
    full_index: int,
    precursor_mz: object,
    charge: object,
    sequence: object,
    mz_values: object,
    intensity_values: object,
) -> tuple[float, int, str, list[float], list[float]]:
    try:
        precursor = float(precursor_mz)
        charge_int = int(charge)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid precursor metadata at source row {full_index}") from exc
    if not math.isfinite(precursor) or precursor <= 0.0:
        raise ValueError(f"Invalid precursor_mz at source row {full_index}: {precursor_mz!r}")
    if charge_int < 1:
        raise ValueError(f"Invalid precursor_charge at source row {full_index}: {charge!r}")
    if not isinstance(sequence, str) or not sequence or any(char.isspace() for char in sequence):
        raise ValueError(f"Invalid SEQ value at source row {full_index}: {sequence!r}")
    mzs = list(mz_values or [])
    intensities = list(intensity_values or [])
    if len(mzs) != len(intensities) or not mzs:
        raise ValueError(f"Invalid peak arrays at source row {full_index}")
    try:
        mzs = [float(value) for value in mzs]
        intensities = [float(value) for value in intensities]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Non-numeric peak values at source row {full_index}") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in mzs):
        raise ValueError(f"Invalid m/z peak at source row {full_index}")
    if any(not math.isfinite(value) or value < 0.0 for value in intensities):
        raise ValueError(f"Invalid intensity peak at source row {full_index}")
    return precursor, charge_int, sequence, mzs, intensities


def write_mgf_record(
    handle, *, full_index: int, precursor_mz: float, charge: int, sequence: str,
    mzs: Iterable[float], intensities: Iterable[float], scan_id: object,
) -> None:
    # PEPMASS is precursor m/z, not neutral precursor mass. Casanovo's own
    # official MGF also uses SEQ= and a charge suffixed with '+'.
    handle.write("BEGIN IONS\n")
    handle.write(f"TITLE=kingdoms_species_cap100k:test:index={full_index}\n")
    handle.write(f"SCANS={scan_id}\n")
    handle.write(f"PEPMASS={precursor_mz:.10f}\n")
    handle.write(f"CHARGE={charge}+\n")
    handle.write(f"SEQ={sequence}\n")
    for mz, intensity in zip(mzs, intensities, strict=True):
        handle.write(f"{mz:.6f} {intensity:.6f}\n")
    handle.write("END IONS\n\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.min_charge < 1 or args.max_charge < args.min_charge:
        raise ValueError("Charge bounds must be positive and ordered.")
    if args.model_label == "casanovo_v5_2_1" and (args.min_charge != 1 or args.max_charge != 4):
        raise ValueError("The canonical Casanovo v5.2.1 export must retain exactly charges 1--4.")
    mgf_name = Path(args.output_mgf_name)
    if mgf_name.name != args.output_mgf_name or mgf_name.suffix.lower() != ".mgf":
        raise ValueError("--output-mgf-name must be an .mgf basename.")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive.")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {args.output_root}")

    dataset = lance.dataset(args.source_lance)
    missing = set(REQUIRED_COLUMNS).difference(dataset.schema.names)
    if missing:
        raise ValueError(f"Source Lance is missing required columns: {sorted(missing)}")
    total_rows = dataset.count_rows()
    output_parent = args.output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temp_root = output_parent / f".{args.output_root.name}.tmp-{os.getpid()}"
    temp_root.mkdir()
    mgf_path = temp_root / mgf_name
    mapping_path = temp_root / "supported_to_full_test_index.npy"
    try:
        retained_indices: list[int] = []
        total_by_charge: Counter[int] = Counter()
        retained_by_charge: Counter[int] = Counter()
        retained = 0
        with mgf_path.open("w", encoding="ascii", newline="\n") as handle, tqdm(
            total=total_rows,
            desc="Export Casanovo v5.2.1 Kingdoms MGF",
            unit="spectrum",
        ) as progress:
            scanner = dataset.scanner(columns=list(REQUIRED_COLUMNS), batch_size=args.batch_size)
            full_index = 0
            for batch in scanner.to_batches():
                columns = {name: batch.column(name).to_pylist() for name in REQUIRED_COLUMNS}
                for offset in range(batch.num_rows):
                    precursor, charge, sequence, mzs, intensities = ensure_valid_record(
                        full_index=full_index,
                        precursor_mz=columns["precursor_mz"][offset],
                        charge=columns["precursor_charge"][offset],
                        sequence=columns["seq"][offset],
                        mz_values=columns["mz_array"][offset],
                        intensity_values=columns["intensity_array"][offset],
                    )
                    total_by_charge[charge] += 1
                    if args.min_charge <= charge <= args.max_charge:
                        write_mgf_record(
                            handle,
                            full_index=full_index,
                            precursor_mz=precursor,
                            charge=charge,
                            sequence=sequence,
                            mzs=mzs,
                            intensities=intensities,
                            scan_id=columns["scan_id"][offset],
                        )
                        retained_indices.append(full_index)
                        retained_by_charge[charge] += 1
                        retained += 1
                        if args.max_records is not None and retained >= args.max_records:
                            break
                    full_index += 1
                progress.update(batch.num_rows)
                if args.max_records is not None and retained >= args.max_records:
                    break
        np.save(mapping_path, np.asarray(retained_indices, dtype=np.int64), allow_pickle=False)
        manifest = {
            "dataset": "denovo_kingdoms_run_disjoint_v1_species_cap100k",
            "purpose": f"{args.model_label} supported-input inference export",
            "source_lance": str(args.source_lance.resolve()),
            "source_lance_rows": total_rows,
            "source_rows_scanned": full_index + (1 if args.max_records is not None and retained else 0),
            "canonical_complete_export": args.max_records is None,
            "max_records": args.max_records,
            "charge_policy": {
                "supported_inference_charges": list(range(args.min_charge, args.max_charge + 1)),
                "unsupported_full_denominator_policy": "rows outside the explicit decoder charge domain remain addressable in the canonical full denominator",
            },
            "rows_by_charge_scanned": {str(key): value for key, value in sorted(total_by_charge.items())},
            "supported_rows_by_charge": {str(key): value for key, value in sorted(retained_by_charge.items())},
            "supported_inference_rows": retained,
            "unsupported_rows_in_scanned_scope": sum(total_by_charge.values()) - retained,
            "mgf": mgf_path.name,
            "mgf_sha256": sha256(mgf_path),
            "supported_to_full_test_index": mapping_path.name,
            "supported_to_full_test_index_sha256": sha256(mapping_path),
            "mgf_contract": {
                "title": "kingdoms_species_cap100k:test:index=<aggregate source row ordinal>",
                "pepmass": "Lance precursor_mz in m/z units",
                "charge": "Lance precursor_charge rendered as <z>+",
                "sequence": "Lance seq rendered as SEQ=<numeric-mass peptidoform>",
                "peaks": "Lance mz_array and intensity_array rendered as two-column MGF peak lines",
            },
        }
        (temp_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temp_root.rename(args.output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    print(f"Wrote {retained:,} supported Casanovo spectra: {args.output_root / mgf_path.name}")
    print(f"Wrote manifest: {args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
