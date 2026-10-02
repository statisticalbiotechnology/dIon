#!/usr/bin/env python3
"""Filter an MGF by precursor charge while preserving retained records verbatim."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

CHARGE_PATTERN = re.compile(rb"(?m)^CHARGE=(\d+)\+?\r?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mgf", type=Path, required=True)
    parser.add_argument("--output-mgf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-charge", type=int, default=1)
    parser.add_argument("--max-charge", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.min_charge < 1 or args.max_charge < args.min_charge:
        raise ValueError("Invalid charge range")
    if not args.input_mgf.is_file():
        raise FileNotFoundError(args.input_mgf)
    if args.output_mgf.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite an existing output or manifest")

    args.output_mgf.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    source_counts: Counter[int] = Counter()
    retained_counts: Counter[int] = Counter()
    source_records = retained_records = 0
    block: list[bytes] = []

    with args.input_mgf.open("rb") as source, args.output_mgf.open("xb") as output:
        progress = tqdm(desc="Filter MGF by precursor charge", unit="spectrum")
        for line in source:
            if line.rstrip(b"\r\n") == b"BEGIN IONS":
                if block:
                    raise ValueError("Nested BEGIN IONS record")
                block = [line]
                continue
            if not block:
                continue
            block.append(line)
            if line.rstrip(b"\r\n") != b"END IONS":
                continue
            payload = b"".join(block)
            match = CHARGE_PATTERN.search(payload)
            if match is None:
                raise ValueError(f"Record {source_records} has no integer CHARGE")
            charge = int(match.group(1))
            source_counts[charge] += 1
            source_records += 1
            if args.min_charge <= charge <= args.max_charge:
                output.write(payload)
                retained_counts[charge] += 1
                retained_records += 1
            block = []
            progress.update()
        progress.close()
    if block:
        raise ValueError("Input ends inside an MGF record")

    manifest = {
        "artifact": "byte_preserving_charge_filtered_mgf",
        "input_mgf": str(args.input_mgf.resolve()),
        "input_sha256": sha256(args.input_mgf),
        "output_mgf": str(args.output_mgf.resolve()),
        "output_sha256": sha256(args.output_mgf),
        "min_charge_inclusive": args.min_charge,
        "max_charge_inclusive": args.max_charge,
        "source_records": source_records,
        "retained_records": retained_records,
        "excluded_records": source_records - retained_records,
        "source_charge_counts": dict(sorted(source_counts.items())),
        "retained_charge_counts": dict(sorted(retained_counts.items())),
        "record_transform": "none; retained BEGIN IONS blocks are copied byte-for-byte",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {retained_records:,}/{source_records:,} records: {args.output_mgf}")
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
