# Stock Casanovo v5.2.1 De Novo Baseline

Use a dedicated environment. Do not install Casanovo or its dependencies into
dIon-env or a mutable Casanovo checkout.

## Verified Environment

The isolated environment is:

```text
${WORK_ROOT}/conda-envs/casanovo_5_2_1
```

It contains Casanovo v5.2.1 from upstream commit
`40f4e9753fd804302d90dea3032dd43f0dbc34bf`. The released stock checkpoint is:

```text
${HOME}/.cache/casanovo/casanovo_orbitrap_v5-2-0.ckpt
```

The checkpoint tokenizer declares `max_charge=4`.

## Primary Official Test Cohort

Use the byte-preserving charge-restricted derivative of the released MGF:

```text
${DATA_ROOT}/casanovo_official/v5_mskb_final/mgf_data/mskb_final/charge1to4_raw/official_test.mgf
```

This is the official 200,000-spectrum MSKB-final test filtered only to retained
charges 2--4: 196,979 spectra (103,352 charge-2, 74,677 charge-3, 18,950
charge-4). Retained `BEGIN IONS` blocks are copied byte-for-byte from the
official MGF; no peaks or metadata are preprocessed before Casanovo runs.
The original full official MGF remains unchanged at:

```text
${DATA_ROOT}/casanovo_official/v5_mskb_final/mgf_data/mskb_final/mskb_final.test.mgf
```

The 3,021 charge-5 spectra are unsupported by the stock v5 checkpoint. They are
not peptide errors in the primary charge-restricted result.

The validated MSKB result is written to the distinct `mskb_final_charge_lt5_raw_mgf` result directory.

## Decode And Score

Run Casanovo without its built-in `--evaluate` option: the released MGF labels use
legacy numeric mass notation, whereas Casanovo v5.2.1 built-in evaluation expects
ProForma. Decode first, then score using dIon's PA1.1 tokenizer offline.

```bash
export CASANOVO_V521_ENV=${WORK_ROOT}/conda-envs/casanovo_5_2_1
export CASANOVO_V521_PY="$CASANOVO_V521_ENV/bin/python"
export CASANOVO_V521_CKPT=${HOME}/.cache/casanovo/casanovo_orbitrap_v5-2-0.ckpt
export MGF=${DATA_ROOT}/casanovo_official/v5_mskb_final/mgf_data/mskb_final/charge1to4_raw/official_test.mgf
export OUT=/path/to/casanovo_v5_2_1_mskb_final_charge_lt5_raw_mgf

"$CASANOVO_V521_ENV/bin/casanovo" sequence --model "$CASANOVO_V521_CKPT" --output_dir "$OUT" --output_root predictions --force_overwrite "$MGF"

"$CASANOVO_V521_PY" -u scripts/evaluate_released_casanovo_mztab.py --mgf "$MGF" --mztab "$OUT/predictions.mztab" --checkpoint "$CASANOVO_V521_CKPT" --output "$OUT/metrics.json" --precision-coverage-output "$OUT/precision_coverage.csv"
```

The scorer uses dIon's PA1.1 tokenizer and in-repository matcher for both
legacy MGF labels and mzTab ProForma predictions. The checkpoint argument is
provenance and selects the ProForma mzTab field; it does not supply matching
masses. The scorer writes confidence-ranked peptide matches and reports
precision--coverage AUC plus full-coverage peptide precision.

## Unlabeled Yeast Run

The stock Casanovo checkpoint can also serve as the supervised reference for
the unlabeled yeast-run experiment. On ClusterB, the isolated CPU installation
and released checkpoint used for the Kingdoms run are:

```text
${WORK_ROOT}/conda_envs/casanovo_5_2_1_cpu
${CHECKPOINT_ROOT}/casanovo_v5_2_1/casanovo_orbitrap_v5-2-0.ckpt
```

Use Casanovo v5.2.1 with this checkpoint and its stock decoding/preprocessing
settings; the checkpoint supports precursor charges 1--4. Keep unsupported
charges explicit in the input accounting rather than silently treating them as
failed peptide calls. Do not pass `--evaluate` for this unlabeled run.

For a large MGF, reuse the parser-safe, complete-record sharder. First count
the charge-supported MGF records, then supply that exact count to
`--expected-records`; select shard count and CPU-array concurrency after a
small timing smoke. The source MGF is not edited.

```bash
PY=${WORK_ROOT}/conda_envs/casanovo_5_2_1_cpu/bin/python
MGF=/path/to/yeast_charge1to4.mgf
SHARDS=/path/to/yeast_casanovo_shards
N=$(rg -c '^BEGIN IONS\r?$' "$MGF")
"$PY" scripts/data_prep/shard_mgf_records.py \
  --source-mgf "$MGF" --output-root "$SHARDS" \
  --shard-count 128 --expected-records "$N"
```

The existing `jobs/portable/casanovo_v521_kingdoms_shard.sh` can decode these
shards on CPU without changing Casanovo code. Its legacy
`CASANOVO_KINGDOMS_SHARDS` variable accepts any shard directory with the
sharder's `manifest.json`. Set a distinct results directory and smoke one
shard before submitting a CPU array; do not submit the Kingdoms GPU wrapper.

```bash
export PROJECT_ROOT=/path/to/dIon
export CASANOVO_V521_ENV=${WORK_ROOT}/conda_envs/casanovo_5_2_1_cpu
export CASANOVO_V521_CKPT=${CHECKPOINT_ROOT}/casanovo_v5_2_1/casanovo_orbitrap_v5-2-0.ckpt
export CASANOVO_KINGDOMS_SHARDS="$SHARDS"
export RESULTS_BASE=/path/to/yeast_casanovo_results
export ALLOW_LOCKED_TEST=1
SLURM_ARRAY_TASK_ID=0 bash "$PROJECT_ROOT/jobs/portable/casanovo_v521_kingdoms_shard.sh"
```

Each completed shard has `predictions.mztab` and `metadata.json`. Keep the
shard manifest and an identifier mapping for the original yeast spectra when
combining results. The Kingdoms stitcher and labeled-spectrum evaluator are
**not** yeast scorers: peptide-to-proteome mapping, modification treatment,
and the denominator for the fraction of calls mapping to yeast proteins must
be fixed separately and applied identically to dIon and Casanovo.

## Kingdoms Species-Capped Test

The canonical cross-species test is not dIon-de-novo-labeled-v1 (DNLv1). It is the deterministic
100,000-per-species cap at:

```text
${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/test.lance
```

Casanovo v5.2.1 supports precursor charges 1--4. Export exactly those rows with
`export_casanovo_v521_kingdoms_mgf.py`; it writes Casanovo-compatible `PEPMASS`,
`CHARGE`, `SEQ`, and two-column peak lines, plus an ordinal mapping to the full
aggregate test. Do not let the CLI silently filter high-charge inputs.

```bash
$PY -u scripts/data_prep/export_casanovo_v521_kingdoms_mgf.py \
  --source-lance ${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/test.lance \
  --output-root ${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/casanovo_v5_2_1_charge1to4
```

The later offline scorer must use `supported_to_full_test_index.npy` to restore
all charge-5+ rows as no-prediction peptide errors in the full 4,926,232-row
denominator. The exporter manifest is the authority for the exact counts.

## Locked-Test Production Array

After the Kingdoms MGF export has completed, submit the stock v5.2.1
held-out cohorts separately, because their runtime scales differ substantially:

```bash
sbatch --export=ALL,ALLOW_LOCKED_TEST=1 \
  jobs/ClusterA/casanovo_v521_mskb_final_locked_test.sbatch

sbatch --export=ALL,ALLOW_LOCKED_TEST=1 \
  jobs/ClusterA/casanovo_v521_kingdoms_locked_test.sbatch
```

The MSKB job has a 1-hour limit for the 196,979-spectrum official charge-<5
cohort. The Kingdoms job has a 16-hour limit for all 4,912,728 charge-1--4
inputs, scored over the canonical 4,926,232-row denominator with the 13,504
unsupported charge-5/6 rows appended as explicit no-prediction peptide errors.
Results are staged in `${RESULTS_ROOT}/denovo_eval/casanovo_v5_2_1/`,
outside the Git worktree.
The job uses `predict_batch_size: 2048`, derived at runtime from the stock v5.2.1
configuration, with `n_beams: 1` unchanged. On an A100 80 GiB benchmark this
was only approximately 2% faster end-to-end than the stock 1024 batch while
leaving materially more headroom than 4096 on ordinary 40 GiB nodes. Override
only for a separately benchmarked environment with `CASANOVO_BATCH_SIZE`.

The job refuses pre-existing output directories and invalid mzTabs. Inputs Casanovo skips after its own peak filtering are appended as explicit zero-confidence no-prediction errors rather than silently removed.

