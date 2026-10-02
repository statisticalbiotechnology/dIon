#!/usr/bin/env python3
"""Validate and atomically stitch global-ID InstaNovo shard CSVs."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--shard-manifest",type=Path,required=True);p.add_argument("--shard-output-root",type=Path,required=True);p.add_argument("--output-csv",type=Path,required=True);p.add_argument("--report",type=Path,required=True);a=p.parse_args();m=json.loads(a.shard_manifest.read_text()); expected=m["source_records"]
 if a.output_csv.exists() or a.report.exists():raise FileExistsError("Refusing to overwrite stitched artifact.")
 a.output_csv.parent.mkdir(parents=True,exist_ok=True);partial=a.output_csv.with_suffix(a.output_csv.suffix+".partial");seen=bytearray(expected);header=None;emitted=0
 try:
  with partial.open("w",newline="") as dst:
   w=None
   for s in m["shards"]:
    root=a.shard_output_root/f"part-{s['index']:05d}";meta=json.loads((root/"metadata.json").read_text());path=root/"predictions.csv"
    if meta["global_start"]!=s["global_start"] or meta["record_count"]!=s["record_count"]:raise ValueError(f"Invalid metadata: {root}")
    with path.open(newline="") as src:
     r=csv.DictReader(src)
     if not r.fieldnames:raise ValueError(f"Missing CSV header: {path}")
     if header is None:header=r.fieldnames;w=csv.DictWriter(dst,fieldnames=header,lineterminator="\n");w.writeheader()
     elif r.fieldnames!=header:raise ValueError(f"CSV schema mismatch: {path}")
     local=set()
     for row in r:
      i=int(row["prediction_id"])
      if not s["global_start"]<=i<s["global_stop_exclusive"] or i in local or seen[i]:raise ValueError(f"Duplicate/out-of-range prediction ID {i}")
      local.add(i);seen[i]=1;w.writerow(row);emitted+=1
    if meta["emitted_predictions"]!=len(local):raise ValueError(f"Count mismatch: {root}")
  missing=[i for i,v in enumerate(seen) if not v];h=hashlib.sha256()
  with partial.open("rb") as f:
   for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
  partial.rename(a.output_csv);report={"schema_version":1,"shard_manifest":str(a.shard_manifest.resolve()),"expected_records":expected,"emitted_predictions":emitted,"missing_prediction_count":len(missing),"missing_prediction_indices":missing,"predictions_csv":str(a.output_csv.resolve()),"predictions_csv_sha256":h.hexdigest()};a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 except Exception:partial.unlink(missing_ok=True);raise
 print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()
