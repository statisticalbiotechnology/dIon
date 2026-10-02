# Casanovo v5.2.1 Kingdoms Sharded Test Hand-Off

## Scope

This package runs the stock Casanovo v5.2.1 checkpoint on the canonical
Kingdoms species-capped held-out test. Casanovo supports charges 1--4 only.
The shard input has exactly `4,912,728` supported records; the final scoring
denominator remains all `4,926,232` canonical Kingdoms rows, with the `13,504`
charge-5/6 rows treated as no-prediction peptide errors.

The original monolithic run completed its GPU inference batches but never
reached Casanovo's all-at-once mzTab writer. Sharding bounds that in-memory
post-inference stage. Each shard is self-contained and produces one mzTab.

## ClusterB CPU Recovery: 128-Way Parser-Safe Layout

The initial ClusterB CPU attempt showed that the generic byte-range MGF splitter could leave a leading blank line at a shard boundary. Casanovo rejects that form. The completed ClusterB run therefore used a **128-way parser-safe subdivision**, created with `scripts/data_prep/shard_mgf_records.py`, which splits only between complete `BEGIN IONS` to `END IONS` records.

For the completed ClusterB run, use these artifacts:

```text
# Authoritative manifest for the completed 128-way run.
.../casanovo_v5_2_1_charge1to4/shards_128_valid/manifest.json

# Transferred output tree.
$RUN/shards/part-00000/ through $RUN/shards/part-00127/
```

The supported-record count remains `4,912,728`; this is a layout recovery, not a dataset change. When stitching the transferred ClusterB outputs, pass `shards_128_valid/manifest.json` to the stitcher. Do **not** use the older `shards_16/manifest.json` with these 128-way outputs.

## Required Files To Copy

Copy these unchanged to ClusterB. Preserve relative filenames inside each
directory.

```text
# Input shards and authoritative local ordinal manifest
${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/casanovo_v5_2_1_charge1to4/shards_16/

# Required to validate mzTab TITLE ordinals and restore the full denominator
${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/casanovo_v5_2_1_charge1to4/supported_to_full_test_index.npy

# Optional but recommended provenance record
${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/casanovo_v5_2_1_charge1to4/manifest.json

# Stock released checkpoint
${HOME}/.cache/casanovo/casanovo_orbitrap_v5-2-0.ckpt
```

Use a dIon checkout that includes these project files:

```text
jobs/portable/casanovo_v521_kingdoms_shard.sh
scripts/stitch_casanovo_v521_shard_mztabs.py
scripts/data_prep/shard_mgf_records.py
scripts/evaluate_released_casanovo_mztab.py
jobs/ClusterB/casanovo_v521_kingdoms_sharded_locked_test_array.sbatch
```

The input `shards_16/manifest.json` is authoritative: it lists every MGF,
its local ordinal range, record count, and SHA-256. Do not rewrite or
preprocess the shard MGFs. They are byte-preserved splits of the raw,
charge-1--4 MGF; Casanovo applies its own preprocessing exactly once.

## Isolated Casanovo Installation

Do not modify dIon-env. Make a separate environment:

```bash
conda create -y -p /path/to/conda-envs/casanovo_5_2_1 python=3.10
conda activate /path/to/conda-envs/casanovo_5_2_1
python -m pip install --upgrade pip
python -m pip install 'git+https://github.com/Noble-Lab/casanovo.git@v5.2.1'

python - <<'PY'
import casanovo, torch
print('Casanovo:', casanovo.__file__)
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
PY
```

The reference decoder is Casanovo v5.2.1 code with the released
`casanovo_orbitrap_v5-2-0.ckpt` checkpoint. Do not change checkpoint,
beam count, or Casanovo preprocessing defaults.

## Scheduler-Neutral Shard Invocation

The portable runner uses `SLURM_ARRAY_TASK_ID` to select a shard. It can be
launched on either a GPU or a CPU node. GPU is substantially faster; a CPU
array is valid if ClusterB has abundant CPU capacity. Select the shard count
or concurrency at ClusterB based on a small throughput smoke, but retain the
existing 16 input shards and their order.

```bash
cd /path/to/dIon
export ALLOW_LOCKED_TEST=1
export PROJECT_ROOT="$PWD"
export CASANOVO_V521_ENV=/path/to/conda-envs/casanovo_5_2_1
export CASANOVO_V521_PY="$CASANOVO_V521_ENV/bin/python"
export CASANOVO_V521_CKPT=/path/to/casanovo_orbitrap_v5-2-0.ckpt
export CASANOVO_KINGDOMS_SHARDS=/path/to/shards_16
export RESULTS_BASE=/path/to/results/casanovo_v5_2_1/kingdoms_species_cap100k_full_denominator_sharded
export CASANOVO_BATCH_SIZE=2048

SLURM_ARRAY_TASK_ID=0 bash jobs/portable/casanovo_v521_kingdoms_shard.sh
```

A successful shard creates:

```text
$RESULTS_BASE/shards/part-00000/predictions.mztab
$RESULTS_BASE/shards/part-00000/metadata.json
```

The runner refuses an incomplete pre-existing shard directory. It is therefore
safe to rerun a failed array element only after removing that element's output
directory, never the complete results tree.

The provided ClusterB GPU wrapper is optional:

```bash
export PROJECT_ROOT=/path/to/dIon
export CASANOVO_V521_ENV=/path/to/conda-envs/casanovo_5_2_1
sbatch --array=0-15%16 jobs/ClusterB/casanovo_v521_kingdoms_sharded_locked_test_array.sbatch
```

For CPU arrays, use the same portable runner from the site-specific CPU sbatch
wrapper; do not add a GPU request. The runner and stitcher are independent of
the scheduler wrapper.

## Stitching On A CPU Node

After all 16 shard directories are complete, stitch on a CPU node using
dIon-env. InstaNovo is not required for this step.

```bash
cd /path/to/dIon
conda env create -f environment.yml  # omit if dIon-env already exists
conda activate dIon-env

export CAS_ROOT=/path/to/casanovo_v5_2_1_charge1to4
export RUN=/path/to/results/casanovo_v5_2_1/kingdoms_species_cap100k_full_denominator_sharded

python -u scripts/stitch_casanovo_v521_shard_mztabs.py \
  --shard-manifest "$CAS_ROOT/shards_128_valid/manifest.json" \
  --supported-to-full-index "$CAS_ROOT/supported_to_full_test_index.npy" \
  --shard-output-root "$RUN/shards" \
  --output-mztab "$RUN/predictions.mztab" \
  --report "$RUN/stitch_manifest.json"
```

Casanovo writes `spectra_ref` ordinals local to each shard, even though the
input MGF TITLE preserves a canonical ordinal. The stitcher translates every
shard-local ordinal through the manifest's supported-input offset and
`supported_to_full_test_index.npy` before writing the stitched mzTab. It rejects
schema changes, duplicate canonical source ordinals, out-of-range local IDs,
mismatched shard metadata, and a pre-existing stitched output. It allows
Casanovo's own post-filter omissions and records those explicitly in
`stitch_manifest.json`.

Score only after the stitched report succeeds. The evaluator resolves mzTab
`TITLE` ordinals through `supported_to_full_test_index.npy`, then restores the
full `4,926,232`-spectrum denominator:

```bash
python -u scripts/evaluate_released_casanovo_mztab.py \
  --mgf "$CAS_ROOT/test_charge1to4_for_casanovo_v5_2_1.mgf" \
  --mztab "$RUN/predictions.mztab" \
  --checkpoint /path/to/casanovo_orbitrap_v5-2-0.ckpt \
  --allow-missing-predictions \
  --full-denominator-count 4926232 \
  --supported-to-full-index "$CAS_ROOT/supported_to_full_test_index.npy" \
  --output "$RUN/metrics.json" \
  --precision-coverage-output "$RUN/precision_coverage.csv"
```

## Return Package

Return the complete directory:

```text
${RESULTS_ROOT}/denovo_eval/casanovo_v5_2_1/kingdoms_species_cap100k_full_denominator_sharded/
```

For the completed ClusterB 128-way recovery, it must include all `shards/part-00000` through `shards/part-00127`, each
with `predictions.mztab` and `metadata.json`, plus top-level
`predictions.mztab` and `stitch_manifest.json`. Do not import into the results
ledger until the stitch report validates all shards.
