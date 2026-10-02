"""Materialize the oxidized-methionine benchmark.

Binary task: given an MS2 spectrum of a **methionine-containing** peptide, is
that methionine oxidized (M+15.995)? Scored by AUROC.

The negative class is deliberately restricted to unoxidized *M-containing*
peptides rather than all peptides. Against all peptides the task degenerates
into "does this peptide contain M at all", which is a composition question
answerable from the precursor mass and amino-acid statistics. Restricted this
way, the model must find a +15.995 shift in the fragment ions -- a local,
fragment-level competence, which is the axis the Casanovo-suite glycosylation
and phosphorylation tasks probe and which the dIon benchmark suite
otherwise lacks.

Labels are free: the bacterial v3 label policy writes oxidation as
``M+15.995`` in ``seq`` and ignores every other modification, so no search is
needed. The corpus is PRIDE PXD010613, the held-out project of the dIon
bacterial corpora, so no spectrum here was seen during pretraining. Splits are
species-disjoint, matching the chimericity benchmark built from the same
project.

Two guards against shortcuts, both materialized as columns:

* ``backbone`` (the sequence with M unmodified) is stored instead of ``seq``.
  Storing the labelled sequence would hand over the answer.
* ``matched_backbone`` marks spectra whose backbone is observed *both*
  oxidized and unoxidized within its split. On that subset the sequence is
  controlled by construction: the model cannot succeed by recognising which
  peptides tend to be oxidized, only by reading the modification. Report the
  matched-subset AUROC alongside the full-set number.

Usage:
    python scripts/materialize_oxidized_met_benchmark.py \
        --lance <pooled test.lance> --output-root <out>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OXIDATION = "M+15.995"
SPLIT_MAP = {
    "caulobacter": "train",
    "efaecalis": "val",
    "akkermansia": "val",
    "halanaerobium": "test",
}
COLUMNS = ["peak_file", "scan_id", "mz_array", "intensity_array",
           "precursor_mz", "precursor_charge", "seq"]


def species_of(run: str) -> str:
    if run.startswith("Ha_"):
        return "halanaerobium"
    if run.startswith("YJ_Cc_"):
        return "caulobacter"
    if run.startswith("Alverdy_Efae"):
        return "efaecalis"
    if "muciniphila" in run:
        return "akkermansia"
    raise ValueError(run)


def backbone_of(seq: str) -> str:
    """Sequence with the oxidation mark removed -- never the label."""
    return seq.replace(OXIDATION, "M")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    import lance

    rows_in = lance.dataset(str(args.lance)).scanner(columns=COLUMNS).to_table().to_pylist()
    print(f"lance rows: {len(rows_in)}")

    rows = []
    for row in rows_in:
        seq = row["seq"]
        if not seq or "M" not in seq:
            continue                      # universe: methionine-containing only
        backbone = backbone_of(seq)
        rows.append({
            "peak_file": row["peak_file"].removesuffix(".mgf"),
            "scan_id": row["scan_id"],
            "species": species_of(row["peak_file"].removesuffix(".mgf")),
            "mz_array": row["mz_array"],
            "intensity_array": row["intensity_array"],
            "precursor_mz": row["precursor_mz"],
            "precursor_charge": row["precursor_charge"],
            "backbone": backbone,
            "label": int(OXIDATION in seq),
            "n_methionine": backbone.count("M"),
            "n_oxidized": seq.count(OXIDATION),
        })
    print(f"methionine-containing spectra: {len(rows)}")

    by_split = defaultdict(list)
    for row in rows:
        by_split[SPLIT_MAP[row["species"]]].append(row)

    # matched_backbone is per-split: the sequence control only holds if both
    # forms are present in the same split the model is scored on.
    for split, items in by_split.items():
        seen = defaultdict(set)
        for row in items:
            seen[row["backbone"]].add(row["label"])
        both = {b for b, labels in seen.items() if len(labels) == 2}
        for row in items:
            row["matched_backbone"] = int(row["backbone"] in both)

    backbones = {s: {r["backbone"] for r in items} for s, items in by_split.items()}
    overlap = {
        f"{a}_vs_{b}": len(backbones[a] & backbones[b])
        for a in backbones for b in backbones if a < b
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "builder": "scripts/materialize_oxidized_met_benchmark.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": "PXD010613 (held-out project of the dIon bacterial corpora)",
        "task": {
            "question": "Is the methionine in this peptide oxidized?",
            "universe": "spectra whose peptide contains at least one M",
            "label_source": "v3 label policy writes oxidation as M+15.995 in seq; "
                            "no search required",
            "metric": "AUROC, reported on the full set and on matched_backbone==1",
            "why_restricted": "Against all peptides the task collapses to 'does the "
                              "peptide contain M', answerable from composition/mass.",
            "shortcut_guards": [
                "seq is not stored; only the unmodified backbone.",
                "matched_backbone==1 marks spectra whose backbone appears BOTH "
                "oxidized and unoxidized within the same split, controlling sequence.",
            ],
        },
        "split_policy": {"disjoint_by": "species", "map": SPLIT_MAP,
                         "backbone_overlap_between_splits": overlap},
        "splits": {},
    }
    for split, items in sorted(by_split.items()):
        labels = np.array([i["label"] for i in items])
        matched = np.array([i["matched_backbone"] for i in items])
        pq.write_table(pa.Table.from_pylist(items), args.output_root / f"{split}.parquet")
        manifest["splits"][split] = {
            "rows": len(items),
            "oxidized": int(labels.sum()),
            "oxidized_fraction": round(float(labels.mean()), 4),
            "matched_backbone_rows": int(matched.sum()),
            "matched_backbone_oxidized_fraction": (
                round(float(labels[matched == 1].mean()), 4) if matched.any() else None
            ),
            "unique_backbones": len({i["backbone"] for i in items}),
            "species": sorted({i["species"] for i in items}),
            "methionines_per_peptide": dict(sorted(Counter(i["n_methionine"] for i in items).items())),
        }
        print(f"{split:6s} {len(items):>7} rows  ox={labels.mean():.3f}  "
              f"matched={int(matched.sum()):>7} ({matched.mean():.1%})  "
              f"species={sorted({i['species'] for i in items})}")
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("cross-split backbone overlap:", overlap)
    print(f"wrote {args.output_root}")


if __name__ == "__main__":
    main()
