# InstaNovo v1.2.2 Kingdoms Sharded GPU Hand-Off

## Purpose

Run the released InstaNovo baseline on the full-charge, 100k-per-species
Kingdoms held-out de novo cohort without loading the complete 4.9M-spectrum
MGF into one process. The upstream InstaNovo MGF loader materializes its input
as a Python list, so the canonical 18 GB MGF exceeds the memory available to a
single normal GPU job. The dataset is therefore split into 16 deterministic,
byte-preserving MGF shards and decoded independently.

This is a locked held-out test. Do not alter the spectra, labels, decoding
configuration, or shard order.

## Canonical Inputs

The completed shard directory and manifest will be at:

```text
${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/instanovo_v1_2_2_charge1to10/shards_16/
```

It contains:

```text
manifest.json
part-00000.mgf
...
part-00015.mgf
```

`manifest.json` is authoritative. It records each shard's MGF filename,
global record range, record count, and SHA-256. The 16 shards together contain
exactly `4,926,232` MGF records, in the original canonical order. All charges
in the full Kingdoms cohort are included for InstaNovo.

The stock supervised checkpoint is:

```text
${CHECKPOINT_ROOT}/instanovo_v1_2_0/instanovo-v1.2.0.ckpt
```

If running from a workstation that cannot access these shared paths, copy the
entire `shards_16/` directory unchanged, including `manifest.json`, plus the
checkpoint. Do not concatenate, rewrite, or peak-filter the MGF files.

## Isolated InstaNovo v1.2.2 Installation

Use a fresh environment; do not modify dIon-env or any existing Casanovo or
InstaNovo environment. The pinned source revision is the v1.2.2 release:
`5a2dd39d602132486662a070354b54412f6e21c7`.

```bash
conda create -y -n instanovo_1_2_2 python=3.11
conda activate instanovo_1_2_2

git clone https://github.com/instadeepai/InstaNovo.git instanovo_1_2_2_source
cd instanovo_1_2_2_source
git checkout 5a2dd39d602132486662a070354b54412f6e21c7

python -m pip install --upgrade pip
# Choose the extra matching the workstation CUDA runtime. This is CUDA 12.6:
python -m pip install -e '.[cu126]'

python - <<'PY'
import instanovo, torch
print('InstaNovo:', instanovo.__version__)
print('Source:', instanovo.__file__)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
git rev-parse HEAD
```

For CUDA 12.4, replace `.[cu126]` with `.[cu124]`. If the workstation already
has a compatible PyTorch CUDA install, retain it and use `python -m pip install
-e . --no-deps` only after confirming its torch version is compatible.

The model checkpoint can alternatively be obtained from the official
InstaNovo release page, but use the exact `instanovo-v1.2.0.ckpt` checkpoint
above for this comparison.

## Fixed Inference Configuration

Run genuine greedy decoding to match the prespecified baseline:

```text
num_beams=1
use_knapsack=false
fp16=true
batch_size=512
num_workers=2
```

Do not turn knapsack back on, increase beam count, refine predictions, or use
a different checkpoint. `log_probs` is retained as the scalar confidence used
for precision--coverage ranking.

## GPU Smoke: 32 Real Spectra and Stitching

Run this on a GPU before the full array. Set paths for the local workstation:

```bash
export SHARDS=${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/instanovo_v1_2_2_charge1to10/shards_16
export CKPT=${CHECKPOINT_ROOT}/instanovo_v1_2_0/instanovo-v1.2.0.ckpt
export SMOKE=$PWD/instanovo_kingdoms_smoke
```

The dIon stitcher requires shard CSVs whose `prediction_id` values are
global, plus per-shard metadata. The portable runner performs that conversion
and validation. The cleanest smoke is therefore to use a checkout of dIon
and invoke the runner twice, overriding only environment-local paths:

```bash
git clone <REPOSITORY_URL> dIon
cd dIon

export ALLOW_LOCKED_TEST=1
export PROJECT_ROOT="$PWD"
export INSTANOVO_V122_ENV="$CONDA_PREFIX"
export INSTANOVO_V120_CKPT="$CKPT"
export INSTANOVO_BATCH_SIZE=512
export INSTANOVO_NUM_WORKERS=2
```

This must produce both of the following for each shard:

```text
$SMOKE/shards/part-00000/predictions.csv
$SMOKE/shards/part-00000/metadata.json
```

For the smoke, use a two-shard manifest created from a small prefix, not the
16-way production manifest. Then stitch exactly those two real outputs. The following creates a byte-preserved 32-record
prefix and its corresponding two-part manifest, then runs the two shard tasks:

```bash
python -u scripts/data_prep/shard_mgf_records.py \
  --source-mgf "$SHARDS/part-00000.mgf" \
  --output-root "$SMOKE/input_shards" \
  --shard-count 2 \
  --expected-records 307890 \
  --max-records 32

export INSTANOVO_KINGDOMS_SHARDS="$SMOKE/input_shards"
export RESULTS_BASE="$SMOKE/run"
for i in 0 1; do
  SLURM_ARRAY_TASK_ID="$i" bash jobs/portable/instanovo_v122_kingdoms_shard.sh
done

# dIon-env is not required for decoding, but it is required for the
# project-controlled stitcher and evaluator.
conda activate dIon-env
python -u scripts/stitch_instanovo_shard_predictions.py \
  --shard-manifest "$SMOKE/input_shards/manifest.json" \
  --shard-output-root "$SMOKE/run/shards" \
  --output-csv "$SMOKE/stitched_predictions.csv" \
  --report "$SMOKE/stitch_manifest.json"
```

The smoke passes only when the stitch report states:

```text
expected_records: 32
missing_prediction_count: 0
emitted_predictions: 32
```

The stitched `prediction_id` column must be exactly the contiguous range
`0..31`. This verifies that local InstaNovo ordinals are correctly offset to
the canonical global MGF order.

## Full 16-Shard Production Run

After the GPU smoke succeeds, return to the full `shards_16` directory and run
one process per shard. The provided Slurm array is configured for one GPU,
four CPUs, 32 GB host RAM, and 90 minutes per shard:

```bash
cd ${WORK_ROOT}/dIon
export ALLOW_LOCKED_TEST=1
sbatch --array=0-15%16 jobs/ClusterA/instanovo_v1_2_2_kingdoms_sharded_locked_test_array.sbatch
```

For another scheduler, call the portable runner once per array index with the
same exports used in the GPU smoke. A shard is complete only when both
`predictions.csv` and `metadata.json` exist. The runner intentionally refuses
to reuse a partial output directory, preventing a failed shard from being
mistaken for a valid result.

## dIon-env Stitching and Return Location

Install dIon-env from the dIon checkout separately. Do not install
InstaNovo into dIon-env. Activate it only for the project-controlled
stitching and evaluation tools:

```bash
cd /path/to/dIon
conda env create -f environment.yml
conda activate dIon-env

export SHARDS=${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/instanovo_v1_2_2_charge1to10/shards_16
export RUN=${RESULTS_ROOT}/denovo_eval/instanovo_v1_2_2/kingdoms_species_cap100k_full_charge_sharded

python -u scripts/stitch_instanovo_shard_predictions.py \
  --shard-manifest "$SHARDS/manifest.json" \
  --shard-output-root "$RUN/shards" \
  --output-csv "$RUN/predictions.csv" \
  --report "$RUN/stitch_manifest.json"
```

Send back the complete output directory, preserving every shard CSV and
metadata file, to:

```text
${RESULTS_ROOT}/denovo_eval/instanovo_v1_2_2/kingdoms_species_cap100k_full_charge_sharded/
```

At minimum it must contain:

```text
shards/part-00000/predictions.csv
shards/part-00000/metadata.json
...
shards/part-00015/predictions.csv
shards/part-00015/metadata.json
predictions.csv
stitch_manifest.json
```

Do not score or import into the results ledger until the stitch report verifies
all `4,926,232` expected records, zero missing global IDs, and a matching
prediction CSV hash.
