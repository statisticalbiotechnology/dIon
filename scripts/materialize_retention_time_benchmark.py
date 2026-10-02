"""Materialize the retention-time benchmark.

Predict when a peptide eluted, from its MS2 spectrum. Unlike the other
benchmarks in this suite there is no external precedent to match: the Casanovo
Foundation paper has no RT task (its four are spectrum quality, chimericity,
phosphorylation, glycosylation), so the design is ours to choose. It is chosen
as follows.

**Target is normalized, not raw minutes.** Raw RT carries per-run offset and
gradient scale that the model cannot see and therefore cannot predict; it would
be irreducible noise in the target. ``nrt`` is the position within the run's
own gradient (robust 0.5/99.5 percentiles -> [0, 1]), the field's usual move
(Prosit predicts indexed RT; DeepLC predicts native units plus a per-run
calibration). ``nrt_aligned`` additionally maps each run onto its species'
reference run by isotonic regression over shared peptidoforms, which absorbs
nonlinear gradient-shape differences. Measured on this corpus, alignment
tightens the replicate spread from 2.24 to 1.90 min, so ``nrt_aligned`` is the
primary target. ``rt_seconds`` is kept so any other scale (iRT included) can be
recomputed.

**Alignment cannot leak across splits.** Splits are species-disjoint and
alignment is within-species, so a run is only ever calibrated against runs in
its own split.

**The noise floor is measured, not assumed.** The same peptidoform observed in
different runs does not land at the same normalized RT; the spread of that is
the best any model can do. It is recorded in the manifest and is what sets
``sigma`` for the soft-label head (see src/soft_ordinal.py): the label's
fuzziness is the measurement's own uncertainty rather than a tuned
hyperparameter. Note the spread is heavy-tailed -- MAE 1.9 min but Delta-t95
near 10 min -- and the tail is most likely misassigned PSMs rather than
chromatography, so report both statistics and do not treat Delta-t95 as a
chromatographic bound.

Usage:
    python scripts/materialize_retention_time_benchmark.py \
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

SPLIT_MAP = {
    "caulobacter": "train",
    "efaecalis": "val",
    "akkermansia": "val",
    "halanaerobium": "test",
}
COLUMNS = ["peak_file", "scan_id", "mz_array", "intensity_array",
           "precursor_mz", "precursor_charge", "seq", "ms2_rt_seconds"]
MIN_ANCHORS = 50


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


def replicate_spread(rows, key):
    """|value - peptidoform median| over peptidoforms seen in >= 2 runs."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["species"], row["seq"])].append((row["peak_file"], row[key]))
    deviations = []
    for observations in grouped.values():
        if len({run for run, _ in observations}) < 2:
            continue
        values = np.array([v for _, v in observations], dtype=float)
        deviations.append(np.abs(values - np.median(values)))
    if not deviations:
        return None
    pooled = np.concatenate(deviations)
    return {
        "peptidoforms": len(deviations),
        "mae": round(float(pooled.mean()), 5),
        "delta_t95": round(float(np.percentile(pooled, 95)), 5),
    }


def align_within_species(rows) -> None:
    """Isotonic map of each run onto its species' reference run."""
    from sklearn.isotonic import IsotonicRegression

    for species in sorted({r["species"] for r in rows}):
        group = [r for r in rows if r["species"] == species]
        counts = Counter(r["peak_file"] for r in group)
        reference = counts.most_common(1)[0][0]
        medians = defaultdict(lambda: defaultdict(list))
        for row in group:
            medians[row["peak_file"]][row["seq"]].append(row["nrt"])
        medians = {run: {seq: float(np.median(v)) for seq, v in d.items()}
                   for run, d in medians.items()}
        for run in counts:
            subset = [r for r in group if r["peak_file"] == run]
            shared = [s for s in medians[run] if s in medians[reference]]
            if run == reference or len(shared) < MIN_ANCHORS:
                for row in subset:
                    row["nrt_aligned"] = row["nrt"]
                    row["aligned"] = int(run == reference)
                continue
            x = np.array([medians[run][s] for s in shared])
            y = np.array([medians[reference][s] for s in shared])
            model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
            predicted = model.predict([r["nrt"] for r in subset])
            for row, value in zip(subset, predicted, strict=True):
                row["nrt_aligned"] = float(value)
                row["aligned"] = 1
        print(f"  {species}: reference={reference}, {len(counts)} runs aligned")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=64)
    args = parser.parse_args()

    import lance

    table = lance.dataset(str(args.lance)).scanner(columns=COLUMNS).to_table().to_pylist()
    rows = []
    for row in table:
        if not row["seq"] or row["ms2_rt_seconds"] is None:
            continue
        run = row["peak_file"].removesuffix(".mgf")
        rows.append({
            "peak_file": run,
            "scan_id": row["scan_id"],
            "species": species_of(run),
            "mz_array": row["mz_array"],
            "intensity_array": row["intensity_array"],
            "precursor_mz": row["precursor_mz"],
            "precursor_charge": row["precursor_charge"],
            "seq": row["seq"],
            "rt_seconds": float(row["ms2_rt_seconds"]),
        })
    print(f"spectra with seq and RT: {len(rows)}")

    # Per-run gradient window -> nrt in [0, 1].
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["peak_file"]].append(row["rt_seconds"])
    windows = {run: tuple(np.percentile(v, [0.5, 99.5])) for run, v in by_run.items()}
    for row in rows:
        low, high = windows[row["peak_file"]]
        row["nrt"] = float(np.clip((row["rt_seconds"] - low) / (high - low), 0.0, 1.0))
    gradients = {run: round((high - low) / 60, 2) for run, (low, high) in windows.items()}

    print("aligning runs within species:")
    align_within_species(rows)
    for row in rows:
        row["nrt_aligned"] = float(np.clip(row["nrt_aligned"], 0.0, 1.0))

    floor_raw = replicate_spread(rows, "nrt")
    floor_aligned = replicate_spread(rows, "nrt_aligned")
    print(f"noise floor nrt        : {floor_raw}")
    print(f"noise floor nrt_aligned: {floor_aligned}")

    # Species-disjoint, then drop peptidoforms spanning splits so the split is
    # exactly peptide-disjoint (RT is a property of the peptide, so a shared
    # peptidoform would be a memorizable answer).
    by_split = defaultdict(list)
    for row in rows:
        by_split[SPLIT_MAP[row["species"]]].append(row)
    seqs = {split: {r["seq"] for r in items} for split, items in by_split.items()}
    shared = set()
    for a in seqs:
        for b in seqs:
            if a < b:
                shared |= seqs[a] & seqs[b]
    print(f"peptidoforms spanning splits: {len(shared)} (dropped)")
    for split in by_split:
        by_split[split] = [r for r in by_split[split] if r["seq"] not in shared]

    sigma = floor_aligned["mae"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "builder": "scripts/materialize_retention_time_benchmark.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": "PXD010613 (held-out project of the dIon bacterial corpora)",
        "task": {
            "question": "At what point in the gradient did this peptide elute?",
            "no_external_precedent": "The Casanovo Foundation paper has no RT task; "
                                     "this design is ours, not a reproduction.",
            "primary_target": "nrt_aligned",
            "targets": {
                "rt_seconds": "raw, kept so any other scale (e.g. iRT) is recomputable",
                "nrt": "position in the run's own gradient, robust 0.5/99.5 percentiles",
                "nrt_aligned": "nrt after isotonic alignment to the species reference run",
            },
            "why_not_raw_minutes": "Per-run offset and gradient scale are invisible to "
                                   "the model, so they would be irreducible target noise.",
            "metrics": ["MAE", "delta_t95", "pearson", "spearman"],
            "caveat": "The replicate spread is heavy-tailed (MAE ~1.9 min, delta_t95 "
                      "~10 min); the tail is most likely misassigned PSMs rather than "
                      "chromatography. Report both, and do not read delta_t95 as a "
                      "chromatographic bound.",
        },
        "noise_floor": {"nrt": floor_raw, "nrt_aligned": floor_aligned,
                        "units": "fraction of gradient; x95 min for approximate minutes"},
        "soft_label_spec": {
            "low": 0.0, "high": 1.0, "n_bins": args.n_bins, "sigma": sigma,
            "rationale": "sigma is the measured replicate spread, not a tuned "
                         "hyperparameter; see src/soft_ordinal.py",
        },
        "split_policy": {
            "disjoint_by": "species, then peptidoform",
            "map": SPLIT_MAP,
            "peptidoforms_dropped_for_overlap": len(shared),
        },
        "gradient_span_minutes": gradients,
        "splits": {},
    }
    for split, items in sorted(by_split.items()):
        target = np.array([r["nrt_aligned"] for r in items])
        pq.write_table(pa.Table.from_pylist(items), args.output_root / f"{split}.parquet")
        manifest["splits"][split] = {
            "rows": len(items),
            "unique_peptidoforms": len({r["seq"] for r in items}),
            "species": sorted({r["species"] for r in items}),
            "runs": sorted({r["peak_file"] for r in items}),
            "target_mean": round(float(target.mean()), 4),
            "target_std": round(float(target.std()), 4),
            "noise_floor": replicate_spread(items, "nrt_aligned"),
        }
        print(f"{split:6s} {len(items):>7} rows  peptidoforms="
              f"{len({r['seq'] for r in items}):>7}  species={sorted({r['species'] for r in items})}")
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_root}")


if __name__ == "__main__":
    main()
