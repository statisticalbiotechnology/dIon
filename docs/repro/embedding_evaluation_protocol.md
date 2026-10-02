# Canonical Embedding Evaluation Protocol

This is the required protocol for every serious MS2 embedding checkpoint and
baseline. It separates online diagnostics, model selection, and locked tests. Do not compare numbers across different corpus manifests, row counts,
peak caps, embedding readouts, or distance metrics.

Benchmark definitions, split construction, and limitations live in
[docs/datasets/dino_ms2_embedding_benchmark_card.md](../datasets/dino_ms2_embedding_benchmark_card.md).
Reported benchmark values are provided in the accompanying publication.
Evaluation commands may generate local artifacts under `results/`; generated
artifacts are not distributed with this source repository.

## Rules

1. Use the checkpoint training encoder configuration, use_mass=1, use_charge=1,
   embedding_readout=backbone, and the training max_peaks. Record max_peaks in
   the leaderboard. The current hybrid used 200.
2. Never enable training augmentations, distractor mixing, masking, or crops at
   evaluation. Evaluators embed clean static-artifact rows.
3. Do not select models on locked test results. Run full validation for each
   serious candidate; run locked tests only after freezing finalists. If a
   test is accessed early, mark it as exploratory exposure and exclude it
   from subsequent model selection.
4. Store immutable result records in a chosen output directory, never only
   beside a checkpoint or in W&B. Canonical YAMLs still use unique report_tag
   values so evaluator reports cannot overwrite each other before ingestion.
5. Use cosine distance for DINO-vs-DINO and DINO-vs-external comparisons.
   Per-species values remain in JSON; leaderboard values are macro averages.
6. Evaluate external models only through a row-aligned artifact pipeline. The
   external model's released preprocessing may reject rows; materialize the
   retained retrieval/pair corpus and run every dIon comparator on those
   exact retained rows. Record the external model's input contract and every
   mismatch with a dIon pretrained encoder in `models.csv`.

## Evaluation Tiers

| Tier | When | Corpora | Purpose | Leaderboard use |
|---|---|---|---|---|
| Online callback | During training | compact 9spec_v2_val_cb, bact_val_cb, kingdoms_val_cb | trajectory and collapse diagnostics | never final |
| Full validation | Every serious completed candidate | NineSpecies V2 validation, bacterial validation, Kingdoms validation | model selection | validation table |
| Locked test | Frozen finalists and fixed baselines only | NineSpecies V2 test, bacterial PXD010613 peptide-disjoint test, Kingdoms test | final results | test table |
| External reference | Once per frozen corpus/version | model-compatible, post-preprocessing matched subset | supervised context only | separate reference table |

The callback suite is
configs/probing_tasks/embedding_probes_paper_validation_suite.yaml; it is not
a substitute for either offline tier.

## External References

External supervised references use the same dIon retrieval and static-pair
metrics but retain their released preprocessing and representation definitions.
They are not used for selecting SSL checkpoints.

| Reference | Representation | Scope | Procedure |
|---|---|---|---|
| GLEAMS v0.3 | released 32-D supervised embedding including precursor metadata and reference spectra | charges 2--5 | [GLEAMS external runner](../repro/gleams_external_reference.md) |
| Casanovo Foundation v4.0.0 | mean of released supervised de novo encoder's valid 512-D peak embeddings | all rows accepted by Casanovo preprocessing | [Casanovo v4 foundation runner](../repro/casanovo_v4_foundation_reference.md) |

For an external retrieval evaluation: export the selected source rows, embed in
the isolated external environment, materialize the retained rows, then use
`evaluate_external_embeddings.py`. For fixed pairs, export every endpoint,
embed, rebalance only among pairs whose two endpoints remain, then use
`evaluate_external_pairs.py`. Store the raw external artifact, external embedding manifest, retained-corpus
manifest, and reports under the same generated experiment directory.


### Metric Sensitivity

The primary cross-model representation metric is cosine. Before finalizing comparisons, evaluate Euclidean retrieval and static-pair metrics for raw
dIon backbone embeddings as a sensitivity analysis, especially for DINO
representations whose norm may carry useful structure. Do not compare a
GLEAMS Euclidean result against cosine-only dIon results; use the same
retained rows and metric for every model in that comparison.

Also evaluate DINO prototype-head assignment similarity as a separate,
explicitly head-level analysis. Suitable candidate distances are cosine over
pre-softmax logits and Jensen-Shannon divergence over prototype probabilities.
DINO's training cross-entropy is teacher--student assignment matching, not a
valid peptide-pair similarity metric, so do not report pairwise cross-entropy
on raw embeddings or arbitrary candidate pairs.

Run inference-time peak-cap sensitivity as a separate analysis
for frozen DINO encoders: evaluate uncapped or higher-cap clean peak lists even
when the encoder was trained with a capped list. Older exploratory experiments
suggest this can improve downstream similarity metrics. This must remain a
separately labelled protocol with its effective peak policy recorded; it does
not replace the matched-preprocessing primary comparison without a deliberate
protocol decision.


## Generated Results Layout

The commands may write auditable evaluation artifacts under a local `results/` directory. This generated directory is not tracked or distributed. W&B
is useful for live trajectories, but it is not the source of final numbers:
run retention, display names, and mutable summaries are not sufficient
provenance for reported results.

Do not encode every experimental attribute in a directory name. In particular,
do not create one path component for a full checkpoint filename: it is long,
fragile, and makes comparisons across models unnecessarily hard. The path gives
the human-scannable experiment boundary; tables carry structured provenance.

```
results/
  <task>/
    <experiment_slug>__<UTC timestamp>/
      README.md
      manifest.json
      datasets.csv
      models.csv
      runs.csv
      metrics.csv
      reports/
        <dataset_id>/
          <model_id>/
            <report_kind>.json
```

Examples of `<task>` are `representation`, `sqa`, `denovo`,
`metric_learning`, and `dense_probe`. Use a task name for the scientific
prediction/evaluation task, not the pretraining objective. For example, a
hybrid encoder evaluated on SQA belongs under `results/sqa/`, while an offline
retrieval/pair benchmark belongs under `results/representation/`.

`<experiment_slug>` describes the frozen experimental question and should be
short but specific, such as `casanovo_19pxd_auc_selected` or
`bacterial_metric_learning_10pct`. The UTC suffix is the creation time in
`YYYYMMDDTHHMMSSZ` form. It distinguishes materially separate executions; it
is not a substitute for source revision or dataset identity.

### Required Files

| File | Role |
|---|---|
| `README.md` | Human summary: question, selection policy, status (`exploratory`, `validation`, or `locked_test`), and known limitations. |
| `manifest.json` | Immutable experiment-level provenance: schema version, creation UTC, Git commit, command(s), random seed(s), selection rule, software/environment notes, and the resolved runtime configuration after every CLI override. It must record all data-limiting settings, including `limit_train_batches`, `limit_val_batches`, `limit_test_batches`, `max_steps`, `epochs`, dataset subsets, and callback frequency. |
| `datasets.csv` | One row per dataset/split artifact, including dataset ID, split, local path, source accession/version, manifest hash where available, row count, and filtering. |
| `models.csv` | One row per evaluated representation/model. It records model ID, initialization type, encoder config, pretraining config, full pretrained checkpoint path and hash, full fine-tuned/downstream checkpoint path and hash when applicable, decoder/head information, and precursor-conditioning mode. |
| `runs.csv` | One row per concrete training/evaluation execution: model ID, W&B entity/project/run ID, SLURM job ID, command/config paths, seed, start/end UTC, and selection status. |
| `metrics.csv` | The canonical machine-readable long-form metric ledger. One row is one metric value for one model, dataset, split, and evaluation protocol. |
| `reports/` | Unmodified JSON evaluator output, W&B export, or task-specific raw report required to audit a ledger row. These artifacts are referenced by relative path from `metrics.csv`. |

`README.md` through `metrics.csv` are mandatory for any final table candidate.
Submitted Sbatch/shell scripts live in the repository-level `jobs/<cluster>/`
directory. A batch-generated result must reference each exact repository-relative
job path, source Git revision, and content hash from `manifest.json` and
`runs.csv`; otherwise the manifest must state that it was an interactive or
external execution and preserve its exact command. A task that does not use W&B
or Slurm still writes `runs.csv`, leaving irrelevant fields empty rather than
silently omitting execution provenance.

### Stable IDs and Checkpoints

`dataset_id` and `model_id` are short stable identifiers used in all CSVs, for
example `casanovo19pxd_balanced_charge1_10` and
`hybrid_e219_finetuned`. They are not source paths. Full paths and a content
hash belong in `datasets.csv`/`models.csv`.

Every model row must distinguish these checkpoint roles explicitly:

| Field | Meaning |
|---|---|
| `pretrained_encoder_checkpoint` | SSL checkpoint used to initialize the encoder; empty for scratch/random initialization. |
| `finetuned_checkpoint` | Resulting downstream checkpoint; empty for frozen representation-only evaluation. |
| `downstream_head_checkpoint` | Optional separately stored task head/decoder checkpoint. |
| `representation_source_checkpoint` | Exact checkpoint whose representation was evaluated. This is the pretrained checkpoint for frozen models and normally the fine-tuned checkpoint for downstream models. |

Record a SHA-256 (or documented equivalent content hash) beside every
non-empty checkpoint path. A display name alone is never sufficient.

For every model initialized from a pretrained encoder, `models.csv` must also
contain these fields:

```text
pretraining_conditions_exact_match,pretraining_reference_checkpoint,
pretraining_reference_config,pretraining_conditioning_deviations
```

`pretraining_conditions_exact_match` is a required boolean attestation that
all representation-relevant input conditions match the pretrained encoder's
training conditions: peak filtering/range/intensity transformation, peak cap,
precursor conditioning mode and null-token semantics, charge/mass inputs, and
any other encoder-visible preprocessing. `true` requires the reference
checkpoint and resolved reference config. `false` requires a concise,
structured `pretraining_conditioning_deviations` value naming every mismatch;
it must never be left ambiguous. For scratch models these fields are empty
except for an optional `not_applicable` value in the boolean column.

### `metrics.csv` Contract

Use tidy/long rows, never one wide CSV per model. Required columns are:

```text
task,experiment_id,model_id,dataset_id,split_id,protocol_id,
metric_name,metric_value,higher_is_better,n_examples,seed,
report_path,selection_status
```

Additional task-specific columns are allowed, for example `threshold`,
`charge_filter`, `label_budget`, `precursor_conditioning`, `n_beams`, or
`embedding_readout`. Keep them explicit columns rather than encoding them in
the metric name or directory path. `protocol_id` must identify the exact
manifest/configuration, such as `same_charge_10ppm_pairs_v1` or
`casanovo19pxd_sqa_balanced_charge1_10_v1`.

`n_examples` is the actual number of examples evaluated for that metric, after
every loader or batch limit. It is not the nominal split size. Record the
number of batches and every effective limit in `manifest.json`; when a report
aggregates differently (for example macro over species or pairs), add explicit
task-specific count columns such as `n_batches`, `n_queries`, `n_pairs`, or
`n_species`.

`selection_status` is mandatory and one of:

```text
exploratory | validation | locked_test | external_reference
```

This prevents a test number accessed during development from quietly entering
a final table as if it were untouched.

### Writer Policy

New evaluation/training entry points should accept a result-root or
experiment-directory argument and write this layout directly. They must not
infer the experiment identity from an output checkpoint basename. A result
writer should:

1. Create a new experiment directory rather than overwrite a previous result.
2. Write `manifest.json`, `datasets.csv`, and `models.csv` before evaluation.
   The manifest contains the fully resolved configuration, including CLI
   overrides and all `limit_*_batches` values. Do not record only a YAML path.
3. Store every submitted job specification in repository-level
   `jobs/<cluster>/` before or immediately after submission. Record its
   repository-relative path, source Git revision, content hash, and Slurm job
   ID in both the script header and the result manifest/runs ledger; a direct
   interactive command is stored as an executable `.sh` file instead.
4. Preserve raw task reports under `reports/<dataset_id>/<model_id>/`.
5. Append or atomically rewrite the normalized `metrics.csv` only after a raw
   report exists and is referenced by `report_path`.
6. Fail if an existing experiment manifest disagrees on schema, dataset,
   checkpoint identity, or selection status.

An evaluation with any non-default `limit_val_batches` or `limit_test_batches`
is a development/screening result, not a full validation or locked-test
result. Its `protocol_id` and `selection_status` must make that explicit. Full-set values require the intended complete manifest unless the published
protocol explicitly defines a fixed subset.

For a multi-checkpoint sweep, keep all models in one experiment directory when
the dataset manifests, task definition, and selection protocol are identical.
Create a new experiment directory when any of those scientific conditions
changes. This yields compact result tables without erasing the distinction
between, for example, an SQA monitor-selection rerun and a new SQA dataset.

The existing
`results/sqa/casanovo_19pxd_balanced_charge1_10/hybrid_epoch100_ablation_auroc_selected/`
artifact is a useful precursor but does not yet conform to this complete
contract. Preserve it unchanged; migrate it only through a documented,
lossless conversion once the common writer is implemented.

## Required Full Validation Suite

Run both retrieval and static pair discrimination for each corpus:

| Corpus | Retrieval config | Pair config | Output tag |
|---|---|---|---|
| NineSpecies V2 | ninespecies_v2_paper_validation_retrieval.yaml | ninespecies_v2_paper_validation_pairs.yaml | 9spec_v2_val |
| Bacterial PXD010000 | bacterial_paper_validation_retrieval.yaml | bacterial_paper_validation_pairs.yaml | bact_val |
| Kingdoms | kingdoms_paper_validation_retrieval.yaml | kingdoms_paper_validation_pairs.yaml | kingdoms_val |

## Locked Test Suite

| Corpus | Retrieval config | Pair config | Output tag |
|---|---|---|---|
| NineSpecies V2 | ninespecies_v2_paper_test_retrieval.yaml | ninespecies_v2_paper_test_pairs.yaml | 9spec_v2_test |
| Bacterial PXD010613 peptide-disjoint | bacterial_paper_test_retrieval.yaml | bacterial_paper_test_pairs.yaml | bact_test |
| Kingdoms | kingdoms_paper_test_retrieval.yaml | kingdoms_paper_test_pairs.yaml | kingdoms_test |

The bacterial and Kingdoms test configs use complete static artifacts. Kingdoms
test is a large end-of-project offline job; do not substitute an undocumented
cap. If it exceeds memory, improve streaming rather than silently resample it.

## Standard DINO Invocation

Use the same master/pretraining configuration used by the checkpoint. Example
for the current hybrid:

~~~bash
CKPT=${CHECKPOINT_ROOT}/important/DINO/bacteria/dinov2_hybrid_distractor_null_local_maxpeaks200_bs128_64gpu_cluster_b_epoch100/last.ckpt
MASTER=configs/master_dion_hybrid_distractor_null_local.yaml
PRETRAIN=configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml

CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_embeddings.py \
  --config "$MASTER" --pretrain_config "$PRETRAIN" \
  --encoder_weights "$CKPT" --pretraining_task dion \
  --probing_config configs/probing_tasks/ninespecies_v2_paper_validation_retrieval.yaml \
  --accelerator gpu --use_mass 1 --use_charge 1 --max_peaks 200

CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_pair_discrimination.py \
  --config "$MASTER" --pretrain_config "$PRETRAIN" \
  --encoder_weights "$CKPT" --pretraining_task dion \
  --probing_config configs/probing_tasks/ninespecies_v2_paper_validation_pairs.yaml \
  --accelerator gpu --use_mass 1 --use_charge 1 --max_peaks 200
~~~

Repeat with every retrieval/pair YAML in the tier. This writes, for example,
last.9spec_v2_val.embedding_eval.json and last.9spec_v2_val.pair_eval.json
beside last.ckpt.

## Required Baselines

| Baseline | Validation | Locked test | Notes |
|---|---|---|---|
| Precursor metadata only | once per benchmark version | once per benchmark version | --embedding_baseline precursor_metadata; shortcut availability, not model quality |
| Binned spectral cosine / spectral angle | once per benchmark version | once per benchmark version | Fixed 1,024-D peak-only binned vector; no precursor metadata or checkpoint |
| Architecture-matched random initialization | once per architecture/input setting | optional after final protocol freeze | calibration only |
| Plain DINO reference | full | full after finalist freeze | no-iBOT/no-KoLeo baseline |
| Candidate DINO variants | full | finalists only | record max_peaks and checkpoint identity |
| GLEAMS | matched corpus only | matched corpus only | supervised reference, separate from primary SSL board |

For metadata-only, omit --encoder_weights and add
--embedding_baseline precursor_metadata. It does not read peak lists.

### Binned Spectral Cosine / Spectral Angle

Use `configs/master_binned_spectrum_embedding.yaml` with
`--embedding_baseline binned_spectrum`. This baseline has 1,024 uniform bins
over m/z `[0, 2500]` (bin width `2.44140625 Da`). Canonical input preprocessing
keeps the top 200 default-filtered peaks, applies per-spectrum min-max intensity
scaling, sums intensities within each bin, then divides the binned vector by its
maximum bin intensity. Cosine evaluation L2-normalizes that fixed vector only
when computing similarity. It has no precursor m/z, mass, charge, or other metadata
features. Retrieval and pair metrics use cosine. Normalized spectral angle
`1 - (2/pi) * arccos(clamp(cosine, -1, 1))` is recorded as an equivalent
monotonic interpretation; it is not recomputed because rankings, mAP/Hit@k/MRR,
ROC-AUC, and AP are identical.

**Locked-test policy (2026-09-11):** do not impose a learned-model
`max_peaks` cap on the binned spectral baseline. The final binned
cosine/spectral-angle test reference must bin every peak surviving its native
filtering and intensity preprocessing, with the effective peak policy recorded
in the result manifest. The current charge-2--4 validation references used
`max_peaks=200` for expedience and are exploratory development comparisons;
they must not be substituted for the final uncapped baseline.

```bash
MASTER=configs/master_binned_spectrum_embedding.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_embeddings.py \
  --config "$MASTER" --embedding_baseline binned_spectrum \
  --probing_config configs/probing_tasks/ninespecies_v2_paper_validation_retrieval.yaml \
  --accelerator gpu --use_mass 0 --use_charge 0 --max_peaks 200

CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_pair_discrimination.py \
  --config "$MASTER" --embedding_baseline binned_spectrum \
  --probing_config configs/probing_tasks/ninespecies_v2_paper_validation_pairs.yaml \
  --accelerator gpu --use_mass 0 --use_charge 0 --max_peaks 200
```

Repeat for all canonical validation and locked-test retrieval/pair manifests.

## Metrics To Record

Primary metrics are static same_charge_10ppm pair discrimination:

- ROC-AUC: probability that a same-peptide pair is ranked above a mass/charge
  matched different-peptide pair.
- Average Precision: precision-recall ranking quality on the balanced pair set.
- FNR at balanced-pair FDR 5%: secondary strict operating point; lower is
  better. It is not search-engine FDR.

Secondary retrieval diagnostic: broad cosine mAP. It helps monitor global
organization but is precursor-metadata-sensitive and cannot alone support a
fragment-understanding claim. Retain all other and per-species metrics in JSON.

## GLEAMS Reference Protocol

Runner/environment details and the exact external-artifact procedure live in
[`docs/repro/gleams_external_reference.md`](../repro/gleams_external_reference.md).
Use that document rather than the dirty external clone's helper scripts.

GLEAMS has already run on compact callback subsets and on the older
gleams_matched NineSpecies corpus. Those results are not canonical validation
or locked-test rows.

For a canonical comparison:

1. Export the exact canonical retrieval or pair input in dIon-env with the
   appropriate export_external script.
2. Encode it in gleams39 with embed_gleams_benchmark.py.
3. Materialize a row-aligned post-preprocessing matched corpus with the
   corresponding materialize_external_matched script.
4. Evaluate GLEAMS and every DINO comparator on that exact retained corpus.
5. Record source/matched manifests, GLEAMS checkpoint and charge restriction,
   plus retained row/pair counts.

Never place GLEAMS-filtered and unfiltered-DINO values in the same leaderboard
column.

## Registry-Driven dIon Evaluation


Use `configs/evaluation/representation_benchmark_registry.yaml` and
`scripts/run_representation_benchmark.py` for offline dIon-model evaluation.
The registry binds each candidate to its checkpoint, original model/pretraining
configuration, fixed preprocessing contract, and permitted inference
conditioning modes. It also distinguishes the normal charge-complete `primary`
cohort from the GLEAMS-retained `gleams_matched_charge2to4` cohort.

The current validation matrix contains raw Hybrid-100/Hybrid-300/random DINO
representations in both `conditioned` and `null` modes, plus the five selected
10% SupCon checkpoints in their trained `conditioned` mode only. Do not run a
conditioned SupCon projection head with null input and report it as a null
training result.

The runner writes a separate result directory for each model/conditioning/corpus
cell, including resolved retrieval/pair YAMLs and `run_metadata.json`. It
records the source checkpoint identity, resolved peak cap, precursor inputs,
cohort manifests, code revision, commands, and completion status. On GPU it
selects extraction batch size `1024` for >=70-GB cards and `512` otherwise.

`locked_test` is intentionally blocked unless `--allow-locked-test` is passed.
The GLEAMS-matched locked-test cohort remains unavailable until its external
embeddings and retained artifacts exist.
