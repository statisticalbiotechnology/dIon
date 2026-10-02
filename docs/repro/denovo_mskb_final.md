# Standalone Official MSKB-final De Novo Benchmark

This is the clean supervised de novo fine-tuning corpus for the Casanovo
`mskb_final` release. It avoids a custom mixed training corpus while retaining
the released official test split exactly.

## Environment Setup

Run materialization and training from a checkout of this exact dIon commit.
The canonical setup is maintained in [README.md](../../README.md#setup):

```bash
git clone <repository-url> dIon
cd dIon
mamba env create -f environment.yml
conda activate dIon-env
```

`conda env create -f environment.yml` is the supported fallback when Mamba is
unavailable. On a managed cluster, load the site Conda/Mamba module first, then
activate `dIon-env`; do not install dependencies into the system Python.

## Source And Split Policy

- Source: `${DATA_ROOT}/casanovo_official/v5_mskb_final/lance_no_peak_cap`.
- Official `test.lance` is copied byte-for-byte into the derived corpus. It is
  never used for validation selection, filtering, or cleanup.
- `val.lance` is a deterministic seed-42 carve-out from official `train.lance`,
  targeting 10,000 spectra before whole-peptidoform rounding.
- `train.lance` contains every remaining official-train spectrum.
- Identity is canonical Casanovo numeric-mass-delta peptidoform, without charge.
  Thus all spectra for an identity move together and train/validation/test
  peptidoform overlap is zero.
- The source labels themselves are retained unchanged in Lance. The canonical
  form is only the membership/audit key.

The derived root is:

```text
${DATA_ROOT}/denovo_mskb_final/lance_peptidoform_val10k_seed42
```

Its `manifest.json` records row/identity counts, zero-overlap audit, the exact
validation identity list hash, and source/copied-test tree hashes.

## Casanovo V5 Charge-Restricted Test Derivative

The canonical official `test.lance` remains the unmodified 200,000-spectrum
test set. For a single shared comparison domain with Casanovo v5-series models,
which support precursor charges 1--4, use only this separately materialized
test-only derivative:

```text
${DATA_ROOT}/denovo_mskb_final/lance_peptidoform_val10k_seed42/
test_charge_lt5_for_casanovo_v_gt_5_0/test.lance
```

Its adjacent `manifest.json` has the explicit label
`charge <5 for Casanovo v>5.0`. It retains 196,979 spectra (charges 2--4)
and excludes 3,021 charge-5 spectra. It preserves every retained source row,
column, sequence notation, and relative row order; it does not modify or
replace the official test artifact. The materializer is:

```text
scripts/data_prep/materialize_denovo_charge_restricted_test.py
```


## Rebuild

Run under `dIon-env` only when intentionally replacing the immutable derived
artifact:

```bash
python -u scripts/data_prep/materialize_mskb_final_denovo_split.py \
  --source-root ${DATA_ROOT}/casanovo_official/v5_mskb_final/lance_no_peak_cap \
  --output-root ${DATA_ROOT}/denovo_mskb_final/lance_peptidoform_val10k_seed42 \
  --seed 42 \
  --validation-target-spectra 10000 \
  --batch-size 8192
```

## Training Recipes

All use the established encoder-larger-deeper architecture, DINO-compatible
peak preprocessing, Casanovo decoder, `n_beams: 1`, and 100 validation batches.
In every variant, the de novo decoder receives the real precursor inputs for
training and decoding; `conditioned` versus `null` only controls encoder
precursor conditioning.

### Run Matrix

1. One full scratch baseline: conditioned encoder, random initialization.
2. For **each** DINO checkpoint, run both full-encoder fine-tunes:
   `conditioned` and `null` encoder conditioning.

The current checkpoints are the hybrid 100-epoch and hybrid 300-epoch models,
so this required set contains five jobs. New DINO checkpoints are expected: for
each newly selected checkpoint, append at least the two required full-encoder
runs (`conditioned` and `null`) before considering its frozen controls. Do not
replace or reinterpret the existing checkpoint rows.


| Index | Variant | Master config | Initialization | Encoder update |
| ---: | --- | --- | --- | --- |
| 0 | Scratch conditioned | `master_denovo_mskb_final_encoderld_scratch.yaml` | random | full |
| 1 | Hybrid-100 conditioned | `master_denovo_dion_hybrid_mskb_final.yaml` | hybrid 100-epoch | full |
| 2 | Hybrid-300 conditioned | `master_denovo_dion_hybrid_mskb_final.yaml` | hybrid 300-epoch | full |
| 3 | Hybrid-100 null | `master_denovo_dion_hybrid_null_mskb_final.yaml` | hybrid 100-epoch | full |
| 4 | Hybrid-300 null | `master_denovo_dion_hybrid_null_mskb_final.yaml` | hybrid 300-epoch | full |

### Frozen Controls

The currently queued frozen control is the 300-epoch hybrid checkpoint with
precursor-conditioned input. The pretrained encoder is held in `eval()` under
`torch.no_grad()` and excluded from the optimizer; the fresh Casanovo decoder
still trains. Its separate array is:

```bash
sbatch jobs/ClusterB/denovo_mskb_final_hybrid300_frozen_16gpu_array.sbatch
```

The hybrid-100/null frozen variants are not implemented in this protocol.

### Validation And Final Test Contract

The official test contains 200,000 spectra and is intentionally expensive to
decode. Keep `n_beams: 1` for every run; multi-beam decoding is not part of
this comparison and makes full-test evaluation unnecessarily costly.

Select checkpoints and make any early-stop decision from `val.lance` only.
Do not use official test metrics for model selection. After selecting a run and
checkpoint, perform one final test with **all** official-test batches enabled:
`limit_test_batches: 1.0` means the complete test set in this codebase. Never
pass a smaller `--limit_test_batches` value, `--limit_test_batches 0.x`, or a
debug subset for the reported result. Record the selected checkpoint and the
full-test setting in the resulting manifest/W&B run.

### Historical Checkpoint Selection Audit

Before treating prior MSKB-final full-test scores as validation-peptide-precision-selected results, inspect their checkpoint callback state and W&B history. The initial Cluster B recipes did not set `checkpoint_monitor`, so they defaulted to `denovo_tf_val_loss_epoch` (minimum). Determine whether the loss-selected checkpoint coincides with each run's `denovo_tf_val_pep_prec` peak; otherwise rerun only the final full-test evaluation from a peptide-precision-selected retained checkpoint, or retrain if that checkpoint was not retained.

### Portable Array Launcher

Use the scheduler-neutral launcher `jobs/portable/denovo_mskb_final_encoderld_array.sh` after activating `dIon-env`. A site-specific scheduler wrapper should submit indices `0-8` and invoke:

```bash
bash jobs/portable/denovo_mskb_final_encoderld_array.sh "$SLURM_ARRAY_TASK_ID"
```

For a non-Slurm launcher, pass the index directly. The main array indices `0-4` are the
primary run set, while local-view comparisons occupy `5-8`; the frozen control
uses its separate one-element array above. Set
`EXTRA_ARGS` only for launch topology, for example
`--num_devices 4 --num_nodes 1 --strategy ddp`; do not place optimizer or
schedule overrides there.

The masters are deliberately one-GPU neutral (`num_devices: 1`, `num_nodes: 1`)
and can be overridden by the target cluster launcher. The reference recipe was
run on 16 GPUs, but a smaller allocation must retain every optimization setting
unchanged: in particular, keep the per-GPU batch size at 100 and do **not**
scale learning rate, warmup, or the step schedule by world size. Those horizons
were originally selected to allow full convergence on the larger dIon-de-novo-labeled-v1 (DNLv1)
corpus. Keep them unchanged at launch for comparability; however, if the
standalone MSKB-final validation monitor has clearly converged earlier, it is
acceptable to stop the run early rather than alter its optimizer schedule. Their
downstream configs are strict copies of the dIon-de-novo-labeled-v1 (DNLv1) recipe with only the
corpus-specific paths and DNLv1-only external validation sets removed.

Example one-GPU conditioned 100-epoch initialization:

```bash
python -u -m src.main \
  --config configs/master_denovo_dion_hybrid_mskb_final.yaml \
  --encoder_weights /path/to/hybrid_100_epoch/last.ckpt \
  --accelerator gpu \
  --num_devices 1 --num_nodes 1 \
  --batch_size 100 \
  --max_steps 149200 \
  --limit_val_batches 100 \
  --log_wandb 1
```

Use the `denovo-mskb-final` W&B project and
`denovo_mskb_final_encoderld_ablation` run group for this comparison.

## dIon-de-novo-labeled-v1 (DNLv1) Reference

The existing dIon-de-novo-labeled-v1 (DNLv1) recipes provide the mixed-corpus
comparison and remain separate from this standalone benchmark:

- Data: `${DATA_ROOT}/denovo_dnlv1/lance`
- Scratch master: `configs/master_denovo_dnlv1_encoderld_scratch.yaml`
- Hybrid conditioned master: `configs/master_denovo_dion_hybrid_dnlv1.yaml`
- Hybrid null master: `configs/master_denovo_dion_hybrid_null_dnlv1.yaml`
- Downstream recipes: `configs/downstream/denovo_dnlv1_dion_hybrid.yaml`
  and `configs/downstream/denovo_dnlv1_dion_hybrid_null.yaml`
- Original ClusterA array example: `jobs/ClusterA/denovo_dnlv1_encoderld_16gpu_array.sbatch`
- Build/provenance record: `docs/datasets/dnlv1_build.md`.
- Materialized provenance and cleanup manifests:
  `${DATA_ROOT}/denovo_dnlv1/build_summary_pre_cleanup.json`,
  `cleanup_config.json`, and `cleanup_summary.json`. The retained
  `lance_pre_cleanup/` root supports a full pre/post-cleanup audit.
- Reproduction/audit scripts: `scripts/data_prep/materialize_dnlv1.py`
  and `scripts/data_prep/audit_dnlv1.py`; the preserved V3 cleanup
  implementation is under `scripts/data_prep/legacy/kitchensink_v3_reference/`.

dIon-de-novo-labeled-v1 (DNLv1) is a distinct mixed-corpus experiment. Do not combine its
validation/test numbers with the standalone official MSKB-final benchmark.

### Cluster B DNLv1 Transfer Array

`jobs/ClusterB/denovo_dnlv1_encoderld_16gpu_array.sbatch` runs only
the high-value three-way comparison: larger-deeper scratch, pairwise scratch,
and conditioned hybrid-300. It stages both DNLv1 and MSKB-final locally on every
allocated node. Each run selects a single checkpoint by **maximum native DNLv1
`denovo_tf_val_pep_prec`**, evaluates that checkpoint on native DNLv1 validation
(200 batches) and the full native DNLv1 test, then resumes the same W&B run ID to
evaluate MSKB-final validation (100 batches) and its full held-out test. The
MSKB metrics are transfer-only and cannot influence DNLv1 checkpoint selection. The DNLv1 train split has 5,275,467 cleaned rows; with 16 GPUs and batch size 100, 80 epochs is 263,840 optimizer updates. The established 6,250-step warmup is retained; only the post-warmup cosine decay is extended through that full 80-epoch budget.

Submit from the frozen run worktree:

```bash
sbatch jobs/ClusterB/denovo_dnlv1_encoderld_16gpu_array.sbatch
```

The matching DNLv1 frozen hybrid-300 conditioned control uses the same native-plus-transfer evaluation contract:

```bash
sbatch jobs/ClusterB/denovo_dnlv1_hybrid300_frozen_16gpu_array.sbatch
```
