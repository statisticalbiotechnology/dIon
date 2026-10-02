# ClusterA Job Specifications

These are historical job specifications for the completed epoch-100 19-PXD SQA
ablation. They are retained as immutable result provenance, not as current
production submission recipes.

| Script | Git revision | Slurm job | Role |
|---|---|---|---|
| `sqa_19pxd_epoch100_ablation_v1.sbatch` | `ca3161e` | `17425290` | Original eight-condition screen; retain only v1 rows with loss/AUROC-optimum agreement. |
| `sqa_19pxd_epoch100_ablation_v2_corrected.sbatch` | `f65cc10` | `17428454` | Five reruns selected directly by validation AUROC. |

The historical source configs are frozen under
`results/sqa/casanovo_19pxd_auc_selected__20260908T160857Z/configs/v1/` and
`results/sqa/casanovo_19pxd_auc_selected__20260908T160857Z/configs/v2/`.
Per-model Slurm task IDs and stdout source paths are in that result record's
`runs.csv`.


## Shell Safety Invariant

Do not use `set -u` in ClusterA job scripts. The Mambaforge module wrapper
reads shell variables that can be absent in a clean Slurm environment, and
nounset has previously terminated entire arrays before Python started. Use
`set -eo pipefail` only:

```bash
set -eo pipefail
cd /path/to/dIon
mkdir -p slurm_outs
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate dIon-env
```

Keep `set -e` and `pipefail` enabled from the start. Historical scripts may
still contain `set -u`; do not use them as templates for new submissions.

## Representation Validation Array


`representation_validation_array.sbatch` launches registry-defined offline
retrieval plus pair evaluation. It does not hard-code model paths: add a future
candidate to `configs/evaluation/representation_benchmark_registry.yaml`, count
cells, then submit the corresponding array.

```bash
PY=/path/to/conda-envs/dIon-env/bin/python
N="$($PY scripts/run_representation_benchmark.py --list-cells \
  --cohort charge2to4 --split validation --model-set all | wc -l)"
sbatch --array="0-$((N - 1))" jobs/ClusterA/representation_validation_array.sbatch
```

The default cohort is validation-only GLEAMS-matched charge 2--4. The script
rejects locked test unless both `SPLIT=locked_test` and `ALLOW_LOCKED_TEST=1`
are set; current matched locked-test artifacts are deliberately marked planned
in the registry.

## Paper Validation Cohorts

The following current scripts are validation-only. They set `test_on_end=0` and
are not substitutes for a later locked-test job.

| Script | Array | Purpose |
|---|---:|---|
| `representation_full_charge_validation_array.sbatch` | dynamically 33 | Full-charge raw DINO and selected metric-learning retrieval/pair cells. |
| `representation_hybrid300_uncapped_validation_array.sbatch` | 0-5 | Hybrid-300-last uncapped inference ablation, conditioned/null x charge-2--4 validation corpora. |
| `binned_full_charge_validation_array.sbatch` | 0-2 | Full-charge uncapped binned spectral-angle reference. |
| `casanovo_v4_full_charge_validation_array.sbatch` | 0-2 | Full-charge released Casanovo v4 peak-mean retrieval/pair reference. |
| `sqa_hybrid300last_validation_array.sbatch` | 0-3 | Hybrid-300-last conditioned/null, frozen/full SQA. |
| `auxiliary_hybrid300last_validation_array.sbatch` | 0-7 | Hybrid-300-last HYE chimericity and oxidized-Met candidates. |
| `retention_time_hybrid300last_validation_array.sbatch` | 0-3 | Hybrid-300-last ordinal and scalar frozen RT probes. |
| `metric_learning_bacterial_1pct_array.sbatch` | 0-2 | Canonical 1% bacterial SupCon cohort: scratch plus selected Hybrid-300-last full/frozen runs. Uses the fixed 100-batch validation monitor and an exhaustive `test.lance` traversal. |
| `metric_learning_bacterial_10pct_hybrid300last_array.sbatch` | 0-2 | Canonical 10% bacterial SupCon rerun with the selected Hybrid-300-last initialization; scratch plus full/frozen DINO variants, fixed validation monitor, exhaustive `test.lance` traversal. |
| `metric_learning_label_efficiency_validation_array.sbatch` | 0-17 | Full-charge (`COHORT=primary`) or GLEAMS-matched charge-2--4 validation retrieval/pair evaluation for all validation-selected 1% and rerun 10% SupCon checkpoints. |
| `metric_learning_label_efficiency_locked_test_array.sbatch` | 0-17 | Guarded held-out retrieval/pair evaluation of that complete prespecified label-efficiency cohort; results stage outside the repository. |
| `casanovo_v4_downstream_validation_cache_array.sbatch` | 0-3 | Train/validation-only Casanovo v4 feature cache. |
| `casanovo_v4_downstream_validation_array.sbatch` | 0-4 | Frozen Casanovo v4 heads after the cache array succeeds. |
| `instanovo_fm_representation_array.sbatch` | 0-2 | Charge-2--4 released InstaNovo-FM validation or guarded held-out retrieval/pair. |
| `instanovo_fm_full_charge_validation_array.sbatch` | 0-2 | Full-charge released InstaNovo-FM validation retrieval/pair. |
| `instanovo_fm_downstream_validation_cache_array.sbatch` | 0-3 | Train/validation-only InstaNovo-FM feature cache. |
| `instanovo_fm_downstream_validation_array.sbatch` | 0-4 | Frozen InstaNovo-FM heads after its cache array succeeds. |
| `denovo_mskb_final_encoderld_array.sbatch` | 0-8 | Official MSKB-final de novo cohort; 0-4 are high priority. |

Submit the representation cohort with its dynamic cell count:

```bash
PY=/path/to/conda-envs/dIon-env/bin/python
N="$($PY scripts/run_representation_benchmark.py --list-cells \
  --cohort primary --split validation --model-set all | wc -l)"
sbatch --array="0-$((N - 1))" jobs/ClusterA/representation_full_charge_validation_array.sbatch
```

The Casanovo cache array must complete before its downstream-head array is
submitted. The cache deliberately contains `train.pt` and `val.pt` only;
held-out test embeddings are materialized only for a separately authorized
final test.

The complete prespecified 1%/10% SupCon label-efficiency cohort has 18 cells
(six validation-loss-selected checkpoints times three corpora). Submit both
validation cohorts and the guarded held-out cohort separately:

```bash
sbatch --array=0-17 jobs/ClusterA/metric_learning_label_efficiency_validation_array.sbatch
sbatch --array=0-17 --export=ALL,COHORT=charge2to4 \
  jobs/ClusterA/metric_learning_label_efficiency_validation_array.sbatch
sbatch --array=0-17 --export=ALL,ALLOW_LOCKED_TEST=1 \
  jobs/ClusterA/metric_learning_label_efficiency_locked_test_array.sbatch
```

All output stages under `/path/to/results/`.
The held-out launcher has no repository output path and requires its explicit
guard even though its committed manifest contains all six prespecified models.

Representation arrays stage generated JSON/results under
`/path/to/results/` by default. Copy only
completed, reviewed result packages into `results/` before committing.


## Locked-Test Launchers

All locked-test launchers require `ALLOW_LOCKED_TEST=1` and stage outputs under
`/path/to/results/`.

- `representation_selected_locked_test_array.sbatch`: selected dIon registry
  candidates from a `status: frozen` selection manifest.
- `binned_locked_test_array.sbatch`: fixed uncapped full-charge binned baseline.
- `casanovo_v4_representation_array.sbatch`: fixed charge-2--4 Casanovo v4 baseline.
- `casanovo_v4_full_charge_locked_test_array.sbatch`: fixed full-charge Casanovo v4 baseline.
- `gleams_v03_locked_test_array.sbatch`: fixed charge-2--4 GLEAMS baseline.
- `casanovo_v4_downstream_locked_test_cache_array.sbatch`: materializes test-only
  Casanovo cache files, after final selection.
- `instanovo_fm_representation_array.sbatch`: fixed charge-2--4 InstaNovo-FM
  baseline with `BENCHMARK_SPLIT=heldout_test,ALLOW_LOCKED_TEST=1`.
- `instanovo_fm_full_charge_locked_test_array.sbatch`: fixed full-charge
  InstaNovo-FM baseline.
- `instanovo_fm_downstream_locked_test_cache_array.sbatch`: materializes
  test-only InstaNovo-FM cache files after final selection.
- `downstream_selected_locked_test_array.sbatch`: selected SQA/auxiliary/RT head
  checkpoints from that same frozen manifest.

For frozen external downstream heads, use `configs/evaluation/locked_test_external_downstream.json`.

Use `configs/evaluation/locked_test_selection.template.json` as the starting
selection record. Copy it to a dated, committed JSON file, fill checkpoint paths
and validation evidence, then change only its `status` to `frozen`.
