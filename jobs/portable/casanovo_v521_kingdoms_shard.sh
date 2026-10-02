#!/usr/bin/env bash
# Decode one byte-preserved Casanovo Kingdoms shard on a CPU or GPU node.
set -euo pipefail
[[ "${ALLOW_LOCKED_TEST:-0}" == 1 ]] || { echo "Set ALLOW_LOCKED_TEST=1." >&2; exit 2; }

ROOT=${DION_ROOT:-/path/to/dIon}
ENV=${CASANOVO_V521_ENV:-/path/to/conda-envs/casanovo_5_2_1}
PY=${CASANOVO_V521_PY:-$ENV/bin/python}
BIN=$ENV/bin/casanovo
CKPT=${CASANOVO_V521_CKPT:-${HOME}/.cache/casanovo/casanovo_orbitrap_v5-2-0.ckpt}
STOCK_CONFIG=$ENV/lib/python3.10/site-packages/casanovo/config.yaml
SHARDS=${CASANOVO_KINGDOMS_SHARDS:-/path/to/data/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/casanovo_v5_2_1_charge1to4/shards_16}
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
BASE=${RESULTS_BASE:-/path/to/results/denovo_eval/casanovo_v5_2_1/kingdoms_species_cap100k_full_denominator_sharded}
OUT=$BASE/shards/part-$(printf '%05d' "$TASK")

read -r NAME START COUNT <<< "$("$PY" - "$SHARDS/manifest.json" "$TASK" <<'PYINNER'
import json,sys
s=json.load(open(sys.argv[1]))['shards'][int(sys.argv[2])]
print(s['mgf'],s['global_start'],s['record_count'])
PYINNER
)"
MGF=$SHARDS/$NAME
[[ -x $BIN && -x $PY && -f $CKPT && -f $STOCK_CONFIG && -f $MGF ]] || { echo "Missing Casanovo executable, checkpoint, config, or shard." >&2; exit 2; }
if [[ -e $OUT ]]; then
  [[ -s $OUT/predictions.mztab && -s $OUT/metadata.json ]] && exit 0
  echo "Refusing incomplete existing output: $OUT" >&2
  exit 2
fi

mkdir -p "$OUT"
CONFIG=${SLURM_TMPDIR:-/tmp}/casanovo_v521_shard_${SLURM_JOB_ID:-manual}_${TASK}.yaml
trap 'rm -f "$CONFIG"' EXIT
sed "s/^predict_batch_size: .*/predict_batch_size: ${CASANOVO_BATCH_SIZE:-2048}/" "$STOCK_CONFIG" > "$CONFIG"
cd "$ROOT"
"$BIN" sequence --model "$CKPT" --config "$CONFIG" --output_dir "$OUT" --output_root predictions --force_overwrite "$MGF"
MZTAB=$OUT/predictions.mztab
[[ -s $MZTAB ]] || { echo "Casanovo produced no mzTab: $MZTAB" >&2; exit 1; }
"$PY" - "$MZTAB" "$START" "$COUNT" "$OUT/metadata.json.partial" <<'PYINNER'
import csv,json,sys
path,start,count,out=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
n=sum(1 for r in csv.reader(open(path),delimiter='\t') if r and r[0]=='PSM')
if n > count: raise SystemExit(f'mzTab has {n} PSMs for a {count}-record shard')
json.dump({'local_start':start,'record_count':count,'emitted_psms':n},open(out,'w'),indent=2)
open(out,'a').write('\n')
PYINNER
mv "$OUT/metadata.json.partial" "$OUT/metadata.json"
