#!/usr/bin/env python
"""Add MSGF+ oxidation labels to the regenerated bacterial Lance splits.

The v1 datasets retain the authoritative MGF labels but discard modifications
stored separately in mzIdentML.  This rebuild preserves split membership and
arrays, replacing only sequences for which a rank-1 MSGF+ PSM contains an
explicit oxidation annotation.  Bare C remains bare C.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import lance
import pyarrow as pa
from tqdm import tqdm


OXIDATION = 15.99491463
OXIDATION_TOL = 1e-4
WRITE_BATCH_SIZE = 10_000


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--replace", action="store_true")
    return p.parse_args()


def local(tag):
    return tag.rsplit("}", 1)[-1]


def scan_id(value):
    m = re.search(r"(?:^|\s)scan=(\d+)(?:\s|$)", value or "")
    if not m:
        m = re.search(r"(?:^|\s)scan(?:s)?=(\d+)(?:\s|$)", value or "")
    return int(m.group(1)) if m else None


def modseq(peptide):
    seq_node = next((x for x in peptide if local(x.tag) == "PeptideSequence"), None)
    if seq_node is None or not seq_node.text:
        return None
    seq = list(seq_node.text)
    mods = []
    for node in peptide:
        if local(node.tag) != "Modification":
            continue
        try:
            delta = float(node.attrib["monoisotopicMassDelta"])
            loc = int(node.attrib["location"])
        except (KeyError, ValueError):
            continue
        if abs(delta - OXIDATION) <= OXIDATION_TOL:
            mods.append((loc, delta))
    for loc, _ in sorted(mods, reverse=True):
        if not 1 <= loc <= len(seq):
            raise ValueError(f"Invalid modification location {loc} in {seq_node.text}")
        if seq[loc - 1] != "M":
            raise ValueError(
                f"MSGF+ oxidation is not on M at {loc}: {seq_node.text}"
            )
        seq.insert(loc, "+15.995")
    return "".join(seq)


def read_mzid(path):
    peptides = {}
    results = {}
    current_scan = None
    in_peptide = False
    with gzip.open(path, "rb") as fh:
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            kind = local(elem.tag)
            if event == "start":
                if kind == "Peptide":
                    in_peptide = True
                elif kind == "SpectrumIdentificationResult":
                    current_scan = scan_id(elem.attrib.get("spectrumID", ""))
                continue
            if kind == "Peptide":
                peptides[elem.attrib["id"]] = modseq(elem)
                in_peptide = False
            elif kind == "SpectrumIdentificationItem":
                if (
                    current_scan is not None
                    and elem.attrib.get("rank") == "1"
                    and elem.attrib.get("passThreshold", "true").lower() == "true"
                ):
                    ref = elem.attrib.get("peptide_ref")
                    if ref:
                        results[current_scan] = ref
            elif kind == "SpectrumIdentificationResult":
                current_scan = None
            if not in_peptide:
                elem.clear()
    return {sid: peptides[ref] for sid, ref in results.items() if peptides.get(ref)}


def modification_map(source_root):
    mapping = {}
    files = sorted(source_root.rglob("*_msgfplus.mzid.gz"))
    if not files:
        raise FileNotFoundError(f"No MSGF+ mzIdentML files under {source_root}")
    for path in tqdm(files, desc="parse MSGF+", unit="file"):
        run = path.name.removesuffix("_msgfplus.mzid.gz")
        mapping[run] = read_mzid(path)
    return mapping


def run_key(peak_file):
    return Path(peak_file).name.removesuffix(Path(peak_file).suffix)


def rewrite_split(src, dst, mapping, stats):
    ds = lance.dataset(str(src))
    mode = "create"
    buffered = []
    for batch in tqdm(ds.scanner(batch_size=4096).to_batches(), desc=src.name, unit="batch"):
        rows = batch.to_pylist()
        for row in rows:
            stats["rows"] += 1
            run = run_key(row["peak_file"])
            new_seq = mapping.get(run, {}).get(int(row["scan_id"]))
            if new_seq is not None:
                if new_seq != row["seq"]:
                    stats["updated"] += 1
                row["seq"] = new_seq
            else:
                stats["unmatched"] += 1
            if "+15.995" in row["seq"]:
                stats[f"{src.name}.oxidized_M_rows"] += 1
            stats[f"{src.name}.unmodified_M_rows"] += int(
                "M" in row["seq"] and "+15.995" not in row["seq"]
            )
            buffered.append(row)
            if len(buffered) >= WRITE_BATCH_SIZE:
                table = pa.Table.from_pylist(buffered, schema=ds.schema)
                lance.write_dataset(table, str(dst), mode=mode)
                mode = "append"
                buffered.clear()
    if buffered:
        lance.write_dataset(pa.Table.from_pylist(buffered, schema=ds.schema), str(dst), mode=mode)


def main():
    a = args()
    if a.output_root.exists():
        if not a.replace:
            raise FileExistsError(f"Output exists: {a.output_root}; use --replace")
        shutil.rmtree(a.output_root)
    a.output_root.mkdir(parents=True)
    mapping = modification_map(a.source_root)
    stats = Counter()
    for split in ("train.lance", "val.lance", "test.lance"):
        rewrite_split(a.input_root / split, a.output_root / split, mapping, stats)
    (a.output_root / "manifest.json").write_text(
        json.dumps({"source": str(a.input_root), "stats": stats, "token": "M+15.995"}, indent=2) + "\n"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
