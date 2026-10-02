"""Materialize a chimericity benchmark from FragPipe wide-window searches.

Protocol follows the Casanovo Foundation chimericity task (arXiv 2505.10848):
a wide-window search that may assign several peptides to one spectrum, a
spectrum labelled chimeric when more than one distinct peptide passes the
PSM-level FDR threshold, unannotated spectra excluded, scored by AUROC.

This is NOT a reproduction. Their corpus is Carafe-derived human/mouse/yeast
and neither the paper nor the Carafe preprint publishes an accession for it,
so an exact rebuild is impossible; report this as a same-protocol benchmark
on an independent, named corpus.

Two corpora are supported, and both are disjoint from dIon pretraining:

* ``--mzml-root`` (primary): PRIDE PXD024584, the HYE three-proteome mixture
  (HeLa + yeast + E. coli, Q Exactive Plus DDA). Because three organisms are
  present, a second and much stricter label is available: a spectrum whose
  co-identified peptides come from *different* organisms is chimeric beyond
  argument -- it cannot be within-proteome homology or search ambiguity.
  Splits are mixture-disjoint (A/B/C differ in H:Y:E ratio).
* ``--lance`` (secondary): PRIDE PXD010613, four bacterial/archaeal species,
  the held-out project of the dIon bacterial corpora. Species-disjoint
  splits; useful as an out-of-corpus test set.

``ms1_window_purity`` is carried as auxiliary metadata, never the target: it
measures *potential* co-isolation from the survey scan, a different quantity
than realized co-identification (measured AUROC of one predicting the other:
0.70, so they are related but far from interchangeable).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.ms1_envelope import C13_C12_DIFF

ORGANISM_TAG = re.compile(r"\b(HUMAN|YEAST|ECOLI)__")
HYE_SPLIT_MAP = {"A": "test", "B": "val", "C": "train"}
HYE_RUN_MIXTURE = {
    "1_1": "A", "1_9": "A",
    "2_1": "B", "2_2": "B",
    "3_1": "C", "3_3": "C", "3_5": "C", "3_6": "C",
}
BACTERIA_SPLIT_MAP = {
    "caulobacter": "train", "efaecalis": "val",
    "akkermansia": "val", "halanaerobium": "test",
}


def read_psm_labels(psm_path: Path, max_qvalue: float):
    """(run, scan) -> (distinct peptides, organisms seen)."""
    peptides: dict[tuple[str, int], set[str]] = defaultdict(set)
    organisms: dict[tuple[str, int], set[str]] = defaultdict(set)
    with psm_path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parts = row["Spectrum"].split(".")
            key = (".".join(parts[:-3]), int(parts[-3]))
            qvalue = row.get("Qvalue")
            if qvalue not in (None, "") and float(qvalue) > max_qvalue:
                continue
            peptides[key].add(row["Peptide"])
            proteins = f"{row.get('Protein', '')} {row.get('Mapped Proteins', '')}"
            organisms[key] |= set(ORGANISM_TAG.findall(proteins))
    return peptides, organisms


def collect_labels(search_root: Path, max_qvalue: float):
    """Merge every psm.tsv under a FragPipe work root."""
    peptides: dict[tuple[str, int], set[str]] = defaultdict(set)
    organisms: dict[tuple[str, int], set[str]] = defaultdict(set)
    found = sorted(search_root.rglob("psm.tsv"))
    if not found:
        raise SystemExit(f"No psm.tsv under {search_root}")
    for psm in found:
        pep, org = read_psm_labels(psm, max_qvalue)
        for key, value in pep.items():
            peptides[key] |= value
        for key, value in org.items():
            organisms[key] |= value
    print(f"read {len(found)} psm.tsv -> {len(peptides)} annotated spectra")
    return peptides, organisms


def window_purity(mz, intensity, peak_index, charge, envelope_span, tolerance_ppm):
    """Envelope intensity / window intensity; None when not computable."""
    if peak_index is None or peak_index < 0 or mz is None or len(mz) < 2:
        return None
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    total = intensity.sum()
    if total <= 0 or not charge:
        return None
    spacing = C13_C12_DIFF / int(charge)
    steps = np.round((mz - mz[peak_index]) / spacing)
    envelope = np.abs(mz - (mz[peak_index] + steps * spacing)) <= mz * tolerance_ppm * 1e-6
    envelope &= np.abs(steps) <= envelope_span
    return float(intensity[envelope].sum() / total)


def iter_mzml_ms2(path: Path):
    """(scan, mz array, intensity array) for every centroided MS2 scan."""
    from pyteomics import mzml

    with mzml.read(str(path)) as reader:
        for spectrum in reader:
            if spectrum.get("ms level") != 2:
                continue
            scan_id = None
            match = re.search(r"scan=(\d+)", spectrum.get("id", ""))
            if match:
                scan_id = int(match.group(1))
            if scan_id is None:
                continue
            yield scan_id, spectrum["m/z array"], spectrum["intensity array"]


def build_hye_rows(args, peptides, organisms):
    """Spectra straight from the HYE mzML, MS1 windows from the sidecars."""
    sidecars = {}
    for parquet in sorted(Path(args.ms1_sidecars).glob("*.parquet")):
        table = pq.read_table(parquet).to_pylist()
        sidecars[parquet.stem] = {r["scan_id"]: r for r in table}
        print(f"  sidecar {parquet.stem}: {len(table)} scans")

    rows = []
    for run, mixture in sorted(HYE_RUN_MIXTURE.items()):
        path = Path(args.mzml_root) / f"{run}.mzML"
        if not path.exists():
            print(f"  WARNING missing {path}")
            continue
        side = sidecars.get(run, {})
        kept = 0
        for scan_id, mz, intensity in iter_mzml_ms2(path):
            key = (run, scan_id)
            found = peptides.get(key)
            if not found:
                continue
            meta = side.get(scan_id, {})
            charge = meta.get("precursor_charge")
            orgs = organisms.get(key, set())
            rows.append({
                "peak_file": run,
                "scan_id": scan_id,
                "partition": mixture,
                "mz_array": [float(v) for v in mz],
                "intensity_array": [float(v) for v in intensity],
                "precursor_mz": meta.get("precursor_mz"),
                "precursor_charge": charge,
                "label": int(len(found) > 1),
                "label_cross_species": int(len(found) > 1 and len(orgs) > 1),
                "n_peptides": len(found),
                "n_organisms": len(orgs),
                "ms1_window_purity": window_purity(
                    meta.get("ms1_mz_array"), meta.get("ms1_intensity_array"),
                    meta.get("ms1_precursor_peak_index"), charge,
                    args.envelope_span, args.tolerance_ppm),
            })
            kept += 1
        print(f"  {run} ({mixture}): {kept} labelled spectra")
    return rows, HYE_SPLIT_MAP, "partition"


def build_lance_rows(args, peptides, organisms):
    """Spectra from the pooled MS1 Lance dataset (bacterial corpus)."""
    import lance

    def species_of(run: str) -> str:
        if run.startswith("Ha_"): return "halanaerobium"
        if run.startswith("YJ_Cc_"): return "caulobacter"
        if run.startswith("Alverdy_Efae"): return "efaecalis"
        if "muciniphila" in run: return "akkermansia"
        raise ValueError(run)

    columns = ["peak_file", "scan_id", "mz_array", "intensity_array", "precursor_mz",
               "precursor_charge", "seq", "ms1_mz_array", "ms1_intensity_array",
               "ms1_precursor_peak_index", "ms1_peak_count", "has_ms1"]
    table = lance.dataset(str(args.lance)).scanner(columns=columns).to_table().to_pylist()
    rows = []
    for row in table:
        run = row["peak_file"].removesuffix(".mgf")
        found = peptides.get((run, row["scan_id"]))
        if not found:
            continue
        rows.append({
            "peak_file": run,
            "scan_id": row["scan_id"],
            "partition": species_of(run),
            "mz_array": row["mz_array"],
            "intensity_array": row["intensity_array"],
            "precursor_mz": row["precursor_mz"],
            "precursor_charge": row["precursor_charge"],
            "seq": row["seq"],
            "label": int(len(found) > 1),
            "label_cross_species": None,   # single-organism runs: undefined
            "n_peptides": len(found),
            "n_organisms": None,
            "ms1_window_purity": window_purity(
                row["ms1_mz_array"], row["ms1_intensity_array"],
                row["ms1_precursor_peak_index"], row["precursor_charge"],
                args.envelope_span, args.tolerance_ppm),
        })
    return rows, BACTERIA_SPLIT_MAP, "partition"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mzml-root", type=Path, help="HYE corpus: mzML directory")
    parser.add_argument("--ms1-sidecars", type=Path, help="HYE corpus: MS1 window sidecars")
    parser.add_argument("--lance", type=Path, help="bacterial corpus: pooled Lance split")
    parser.add_argument("--corpus-name", default=None)
    parser.add_argument("--max-qvalue", type=float, default=0.01)
    parser.add_argument("--envelope-span", type=int, default=4)
    parser.add_argument("--tolerance-ppm", type=float, default=15.0)
    args = parser.parse_args()
    if bool(args.mzml_root) == bool(args.lance):
        raise SystemExit("Pass exactly one of --mzml-root (HYE) or --lance (bacteria).")

    peptides, organisms = collect_labels(args.search_root, args.max_qvalue)
    if args.mzml_root:
        rows, split_map, key = build_hye_rows(args, peptides, organisms)
        corpus = args.corpus_name or "PXD024584_HYE"
    else:
        rows, split_map, key = build_lance_rows(args, peptides, organisms)
        corpus = args.corpus_name or "PXD010613_bacteria"

    args.output_root.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for row in rows:
        by_split[split_map[row[key]]].append(row)

    manifest = {
        "builder": "scripts/materialize_chimericity_benchmark.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "protocol": {
            "reference": "Casanovo Foundation chimericity task (arXiv 2505.10848)",
            "reproduction": False,
            "search": "FragPipe 23.0 / MSFragger 4.2, WWA workflow (+-20 Da precursor "
                      "window, deisotope, up to 5 peptides per spectrum)",
            "label_rule": f"chimeric = >1 distinct peptide at PSM q <= {args.max_qvalue}",
            "excluded": "spectra with no wide-window peptide assignment",
            "metric": "AUROC",
            "deviations": [
                "The reference corpus (Carafe-derived human/mouse/yeast) has no "
                "published accession in the paper or the Carafe preprint; this is a "
                "same-protocol benchmark on a named, independent corpus.",
                "label_cross_species (peptides from different organisms) is an "
                "addition, definable only in a multi-species mixture.",
                "ms1_window_purity is auxiliary metadata, not a target.",
            ],
        },
        "split_policy": {"disjoint_by": key, "map": split_map},
        "splits": {},
    }
    for split, items in sorted(by_split.items()):
        labels = np.array([i["label"] for i in items])
        cross = [i["label_cross_species"] for i in items if i["label_cross_species"] is not None]
        purity = np.array([i["ms1_window_purity"] for i in items
                           if i["ms1_window_purity"] is not None], dtype=float)
        pq.write_table(pa.Table.from_pylist(items), args.output_root / f"{split}.parquet")
        manifest["splits"][split] = {
            "rows": len(items),
            "chimeric": int(labels.sum()),
            "chimeric_fraction": round(float(labels.mean()), 4),
            "cross_species_fraction": round(float(np.mean(cross)), 4) if cross else None,
            "partitions": sorted({i[key] for i in items}),
            "runs": sorted({i["peak_file"] for i in items}),
            "peptides_per_spectrum": dict(sorted(Counter(i["n_peptides"] for i in items).items())),
            "median_ms1_window_purity": round(float(np.median(purity)), 4) if purity.size else None,
        }
        print(f"{split:6s} {len(items):>7} rows  chimeric={labels.mean():.3f}  "
              f"partitions={sorted({i[key] for i in items})}")
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_root}")


if __name__ == "__main__":
    main()
