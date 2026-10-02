#!/usr/bin/env bash
set -euo pipefail
[[ "${ALLOW_LOCKED_TEST:-0}" == 1 ]] || { echo "Set ALLOW_LOCKED_TEST=1." >&2; exit 2; }
ROOT=${DION_ROOT:-/path/to/dIon}; ENV=${INSTANOVO_V122_ENV:-/path/to/conda-envs/instanovo_1_2_2}; PY=${CASANOVO_V521_PY:-/path/to/conda-envs/casanovo_5_2_1/bin/python}; BIN=$ENV/bin/instanovo; CKPT=${INSTANOVO_V120_CKPT:-/path/to/work/checkpoints/instanovo_v1_2_0/instanovo-v1.2.0.ckpt}; SHARDS=${INSTANOVO_KINGDOMS_SHARDS:-/path/to/data/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/instanovo_v1_2_2_charge1to10/shards_16}; TASK=${SLURM_ARRAY_TASK_ID:?array task required}; OUT=${RESULTS_BASE:-/path/to/results/denovo_eval/instanovo_v1_2_2/kingdoms_species_cap100k_full_charge_sharded}/shards/part-$(printf "%05d" "$TASK"); FP16=${INSTANOVO_FP16:-true}; FORCE_CPU=${INSTANOVO_FORCE_CPU:-false}
read -r NAME START COUNT <<< "$("$PY" - "$SHARDS/manifest.json" "$TASK" <<'PYINNER'
import json,sys
s=json.load(open(sys.argv[1]))["shards"][int(sys.argv[2])];print(s["mgf"],s["global_start"],s["record_count"])
PYINNER
)"
MGF=$SHARDS/$NAME; [[ -x $BIN && -x $PY && -f $CKPT && -f $MGF ]] || { echo "Missing executable, checkpoint, or shard." >&2; exit 2; }
if [[ -e $OUT ]]; then [[ -f $OUT/metadata.json && -s $OUT/predictions.csv ]] && exit 0; echo "Refusing incomplete existing output: $OUT" >&2;exit 2;fi
mkdir -p "$OUT"; PART=$OUT/predictions.csv.partial;export TMPDIR=${TMPDIR_OVERRIDE:-/tmp};cd "$ROOT"
$BIN transformer predict --data-path "$MGF" --output-path "$PART" --instanovo-model "$CKPT" --denovo num_beams=1 use_knapsack=false batch_size=${INSTANOVO_BATCH_SIZE:-512} num_workers=${INSTANOVO_NUM_WORKERS:-2} fp16="$FP16" force_cpu="$FORCE_CPU"
"$PY" - "$PART" "$START" "$COUNT" "$OUT/metadata.json.partial" <<'PYINNER'
import csv,json,sys
p,start,count,out=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4];rows=[];seen=set()
with open(p,newline="") as f:
 r=csv.DictReader(f);fields=r.fieldnames
 if not fields or not {"prediction_id","predictions","log_probs"}<=set(fields):raise SystemExit("Invalid CSV schema")
 for row in r:
  i=int(row["prediction_id"])
  if not 0<=i<count or i in seen:raise SystemExit(f"Invalid local prediction ID {i}")
  seen.add(i);row["prediction_id"]=str(start+i);rows.append(row)
with open(p,"w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
json.dump({"global_start":start,"record_count":count,"emitted_predictions":len(rows)},open(out,"w"),indent=2);open(out,"a").write("\n")
PYINNER
mv "$PART" "$OUT/predictions.csv";mv "$OUT/metadata.json.partial" "$OUT/metadata.json"
