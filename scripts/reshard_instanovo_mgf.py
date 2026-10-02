#!/usr/bin/env python3
"""Deterministically split canonical InstaNovo MGF shards without reordering spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BEGIN = b"BEGIN IONS"
END = b"END IONS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards-per-input", type=int, default=4)
    return parser.parse_args()


def split_sizes(total: int, parts: int) -> list[int]:
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def main() -> None:
    args = parse_args()
    if args.shards_per_input < 1:
        raise SystemExit("--shards-per-input must be positive")
    source_manifest = json.loads((args.input_dir / "manifest.json").read_text())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_shards: list[dict[str, object]] = []
    output_index = 0
    for source in source_manifest["shards"]:
        source_path = args.input_dir / source["mgf"]
        counts = split_sizes(int(source["record_count"]), args.shards_per_input)
        boundaries, total = [], 0
        for count in counts:
            total += count
            boundaries.append(total)
        handles = []
        for count in counts:
            filename = f"part-{output_index + len(handles):05d}.mgf"
            handles.append((filename, (args.output_dir / filename).open("wb"), count, hashlib.sha256()))

        target = spectra = 0
        active = False
        try:
            with source_path.open("rb") as source_handle:
                for line in source_handle:
                    if line.rstrip(b"\r\n") == BEGIN:
                        if active:
                            raise ValueError(f"Nested BEGIN IONS in {source_path}")
                        active = True
                    if target >= len(handles):
                        if line.strip():
                            raise ValueError(f"Unexpected data after final spectrum in {source_path}")
                        continue
                    _, output_handle, _, digest = handles[target]
                    output_handle.write(line)
                    digest.update(line)
                    if line.rstrip(b"\r\n") == END:
                        if not active:
                            raise ValueError(f"END IONS without BEGIN IONS in {source_path}")
                        active = False
                        spectra += 1
                        if spectra == boundaries[target]:
                            target += 1
            if active or spectra != int(source["record_count"]) or target != len(handles):
                raise ValueError(f"Unexpected spectrum count in {source_path}: {spectra}")
        finally:
            for _, output_handle, _, _ in handles:
                output_handle.close()

        start = int(source["global_start"])
        for filename, _, count, digest in handles:
            output_shards.append({"index": output_index, "mgf": filename, "global_start": start,
                                  "global_stop_exclusive": start + count, "record_count": count,
                                  "sha256": digest.hexdigest()})
            start += count
            output_index += 1

    if sum(int(shard["record_count"]) for shard in output_shards) != int(source_manifest["source_records"]):
        raise RuntimeError("Output record count does not match source manifest")
    manifest = {"schema_version": source_manifest["schema_version"],
                "source_mgf": source_manifest.get("source_mgf"),
                "source_records": source_manifest["source_records"],
                "source_sha256": source_manifest.get("source_sha256"),
                "canonical_complete_export": source_manifest.get("canonical_complete_export"),
                "resharded_from": str(args.input_dir), "shards_per_input": args.shards_per_input,
                "shard_count": len(output_shards), "shards": output_shards}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(output_shards)} shards with {manifest['source_records']} spectra.")


if __name__ == "__main__":
    main()
