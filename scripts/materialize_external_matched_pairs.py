"""Filter and rebalance static pairs after external preprocessing.

Run under dIon-env after external inference. Every retained pair references
two externally embedded spectra, and each species/protocol remains 1:1
positive:negative. The output is the sole pair corpus for both model families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embed_eval.pair_data import stable_rank


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spectra", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--external-embeddings", type=Path, required=True)
    parser.add_argument("--output-spectra", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spectra = pq.read_table(args.source_spectra)
    pairs = pq.read_table(args.source_pairs)
    with np.load(args.external_embeddings, allow_pickle=False) as artifact:
        if "spectrum_id" not in artifact.files:
            raise ValueError("External pair artifact must preserve spectrum_id.")
        valid_ids = {str(value) for value in artifact["spectrum_id"].tolist()}
    source_ids = [str(value) for value in spectra["spectrum_id"].to_pylist()]
    kept_spectrum_indices = [
        index for index, spectrum_id in enumerate(source_ids) if spectrum_id in valid_ids
    ]
    if len(kept_spectrum_indices) != len(valid_ids):
        missing = valid_ids - set(source_ids)
        raise ValueError(f"External artifact has spectra absent from source: {next(iter(missing))!r}")
    matched_spectra = spectra.take(pa.array(kept_spectrum_indices, type=pa.int64()))

    pair_records = pairs.to_pylist()
    available_pairs = [
        record for record in pair_records
        if str(record["left_spectrum_id"]) in valid_ids
        and str(record["right_spectrum_id"]) in valid_ids
    ]
    grouped: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in available_pairs:
        grouped[(str(record["pair_set"]), str(record["species"]))][int(record["label"])].append(record)
    kept_records = []
    availability: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    dropped_groups = []
    for (pair_set, species), by_label in sorted(grouped.items()):
        if set(by_label) != {0, 1}:
            dropped_groups.append(f"{pair_set}/{species}")
            continue
        ordered = {
            label: sorted(
                records,
                key=lambda record: stable_rank(
                    args.seed,
                    pair_set,
                    species,
                    label,
                    record["pair_id"],
                ),
            )
            for label, records in by_label.items()
        }
        balanced = min(len(ordered[0]), len(ordered[1]))
        if balanced == 0:
            dropped_groups.append(f"{pair_set}/{species}")
            continue
        kept_records.extend(ordered[0][:balanced])
        kept_records.extend(ordered[1][:balanced])
        availability[pair_set][species] = {
            "positive_pair_count": balanced,
            "negative_pair_count": balanced,
            "pair_count": 2 * balanced,
        }
    if not kept_records:
        raise ValueError("No balanced pairs remain after external preprocessing.")
    matched_pairs = pa.Table.from_pylist(kept_records, schema=pairs.schema)
    for output_path in (args.output_spectra, args.output_pairs, args.output_manifest):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(matched_spectra, args.output_spectra)
    pq.write_table(matched_pairs, args.output_pairs)
    args.output_manifest.write_text(json.dumps({
        "source_spectra": str(args.source_spectra.resolve()),
        "source_spectra_sha256": _sha256(args.source_spectra),
        "source_pairs": str(args.source_pairs.resolve()),
        "source_pairs_sha256": _sha256(args.source_pairs),
        "external_embeddings": str(args.external_embeddings.resolve()),
        "external_embeddings_sha256": _sha256(args.external_embeddings),
        "seed": args.seed,
        "source_spectra_count": spectra.num_rows,
        "external_retained_spectra_count": matched_spectra.num_rows,
        "source_pair_count": pairs.num_rows,
        "pairs_with_both_external_spectra": len(available_pairs),
        "balanced_pair_count": matched_pairs.num_rows,
        "pair_sets": availability,
        "dropped_pair_set_species": dropped_groups,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {matched_spectra.num_rows:,} externally retained pair spectra: {args.output_spectra}")
    print(f"Wrote {matched_pairs.num_rows:,} balanced matched pairs: {args.output_pairs}")
    print(f"Wrote manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
