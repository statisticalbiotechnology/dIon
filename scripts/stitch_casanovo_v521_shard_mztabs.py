#!/usr/bin/env python3
"""Validate and atomically stitch Casanovo v5.2.1 Kingdoms shard mzTabs."""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

INDEX = re.compile(r"(?:^|:)index=(\d+)$")

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard-manifest',type=Path,required=True)
    p.add_argument('--supported-to-full-index',type=Path,required=True)
    p.add_argument('--shard-output-root',type=Path,required=True)
    p.add_argument('--output-mztab',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.output_mztab.exists() or a.report.exists(): raise FileExistsError('Refusing to overwrite stitched artifact.')
    manifest=json.loads(a.shard_manifest.read_text())
    full=np.load(a.supported_to_full_index,allow_pickle=False).astype(np.int64,copy=False)
    if len(full)!=manifest['source_records'] or len(np.unique(full))!=len(full): raise ValueError('Invalid supported-to-full index mapping.')
    supported=np.zeros(int(full.max())+1,dtype=bool); supported[full]=True; seen=np.zeros_like(supported); header=None; emitted=0
    shard_metadata=[]
    for spec in manifest['shards']:
      root=a.shard_output_root/f"part-{spec['index']:05d}"; meta=json.loads((root/'metadata.json').read_text())
      if meta['local_start']!=spec['global_start'] or meta['record_count']!=spec['record_count']: raise ValueError(f'Invalid metadata: {root}')
      shard_metadata.append((spec, root, meta, root/'predictions.mztab'))
    expected_emitted=sum(int(meta['emitted_psms']) for _,_,meta,_ in shard_metadata)
    a.output_mztab.parent.mkdir(parents=True,exist_ok=True); partial=a.output_mztab.with_suffix(a.output_mztab.suffix+'.partial')
    try:
      with partial.open('w',newline='') as dst, tqdm(total=expected_emitted, unit='PSM', desc='Stitch Casanovo mzTabs') as progress:
       writer=csv.writer(dst,delimiter='\t',lineterminator='\n')
       for spec, root, meta, path in shard_metadata:
        n=0; psh=None
        for row in csv.reader(path.open(),delimiter='\t'):
         if not row: continue
         if row[0]=='MTD':
          if header is None: writer.writerow(row)
         elif row[0]=='PSH':
          if psh is None: psh=row
          elif row!=psh: raise ValueError(f'Multiple incompatible PSH rows: {path}')
          if header is None: writer.writerow(row); header=row
          elif row != header: raise ValueError(f'PSH schema mismatch: {path}')
         elif row[0]=='PSM':
          if psh is None: raise ValueError(f'PSM before PSH: {path}')
          values=dict(zip(psh[1:],row[1:],strict=True)); match=INDEX.search(values['spectra_ref'])
          if match is None: raise ValueError(f'Unparseable spectra_ref in {path}: {values["spectra_ref"]!r}')
          local_index=int(match.group(1))
          if local_index >= spec['record_count']:
           raise ValueError(f'Out-of-range shard-local source index {local_index} in {path}')
          supported_index = int(spec['global_start']) + local_index
          canonical_index = int(full[supported_index])
          if seen[canonical_index]:
           raise ValueError(f'Duplicate canonical source index {canonical_index}')
          seen[canonical_index]=True
          spectra_ref_column = psh[1:].index('spectra_ref') + 1
          row[spectra_ref_column] = INDEX.sub(lambda match: (':' if match.group(0).startswith(':') else '') + f'index={canonical_index}', values['spectra_ref'])
          writer.writerow(row); n+=1; emitted+=1
        if n!=meta['emitted_psms']: raise ValueError(f'PSM count mismatch: {root}')
        progress.update(n)
      digest=hashlib.sha256(partial.read_bytes()).hexdigest(); partial.rename(a.output_mztab)
      report={'schema_version':1,'shard_manifest':str(a.shard_manifest.resolve()),'supported_input_count':int(len(full)),'full_denominator_count':int(full.max()+1),'emitted_psms':emitted,'missing_supported_prediction_count':int(len(full)-emitted),'unsupported_charge_count':int(len(supported)-len(full)),'mztab':str(a.output_mztab.resolve()),'mztab_sha256':digest}
      a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True))
    except Exception:
      partial.unlink(missing_ok=True); raise
if __name__=='__main__': main()
