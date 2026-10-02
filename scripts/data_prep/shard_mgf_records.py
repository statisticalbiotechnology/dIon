#!/usr/bin/env python3
"""Byte-preservingly split an annotated MGF into deterministic record shards."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
from tqdm.auto import tqdm

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--source-mgf",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--shard-count",type=int,default=16); p.add_argument("--expected-records",type=int,default=4_926_232); p.add_argument("--max-records",type=int,help="Bounded smoke-only prefix; omit for the canonical full export."); a=p.parse_args()
 if a.max_records is not None: a.expected_records=a.max_records
 if a.shard_count<2 or a.expected_records<a.shard_count: raise ValueError("Invalid shard count or expected record count.")
 if not a.source_mgf.is_file() or a.output_root.exists(): raise FileExistsError("Source missing or output root already exists.")
 counts=[a.expected_records//a.shard_count+(i<a.expected_records%a.shard_count) for i in range(a.shard_count)]
 tmp=a.output_root.parent/f".{a.output_root.name}.tmp-{os.getpid()}"; tmp.mkdir(parents=True); names=[f"part-{i:05d}.mgf" for i in range(a.shard_count)]; source_hash=hashlib.sha256(); hashes=[hashlib.sha256() for _ in names]; hs=[]
 try:
  hs=[(tmp/n).open("wb") for n in names]; shard=record=0; last_shard=0; in_record=False; limits=[]; total=0
  for count in counts: total+=count; limits.append(total)
  with a.source_mgf.open("rb") as src, tqdm(total=a.source_mgf.stat().st_size,unit="B",unit_scale=True,desc="Shard InstaNovo Kingdoms MGF") as bar:
   for line in src:
    source_hash.update(line); bar.update(len(line)); marker=line.strip()
    if marker==b"BEGIN IONS":
     if in_record: raise ValueError(f"Nested MGF record before ordinal {record}.")
     if shard>=len(hs): raise ValueError("Source has more records than expected.")
     in_record=True
    if not in_record:
     if marker: raise ValueError(f"Content outside MGF record before ordinal {record}.")
     # Preserve inter-record whitespace in the preceding shard so concatenating
     # shards recreates the source MGF byte-for-byte, including boundaries.
     target = last_shard
     hs[target].write(line); hashes[target].update(line)
     continue
    hs[shard].write(line); hashes[shard].update(line)
    if marker==b"END IONS":
     in_record=False; last_shard=shard; record+=1
     if record==limits[shard]: shard+=1
     if record==a.expected_records: break
  if in_record or record!=a.expected_records or shard!=a.shard_count: raise ValueError(f"Expected {a.expected_records} complete records; found {record}.")
  for h in hs:h.close()
  start=0; specs=[]
  for i,(n,c,h) in enumerate(zip(names,counts,hashes,strict=True)):
   specs.append({"index":i,"mgf":n,"global_start":start,"record_count":c,"global_stop_exclusive":start+c,"sha256":h.hexdigest()});start+=c
  (tmp/"manifest.json").write_text(json.dumps({"schema_version":1,"source_mgf":str(a.source_mgf.resolve()),"source_sha256":source_hash.hexdigest(),"source_records":record,"shard_count":a.shard_count,"max_records":a.max_records,"canonical_complete_export":a.max_records is None,"shards":specs},indent=2,sort_keys=True)+"\n")
  tmp.rename(a.output_root)
 except Exception:
  for h in hs:
   if not h.closed:h.close()
  shutil.rmtree(tmp,ignore_errors=True);raise
 print(f"Wrote {record:,} byte-preserved MGF records across {a.shard_count} shards: {a.output_root}")
if __name__=="__main__":main()
