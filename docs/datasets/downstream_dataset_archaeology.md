# Downstream Dataset Archaeology

## Scope

Inventory performed 2026-08-20 before rebuilding any downstream benchmark.
Existing artifacts are preserved. `Reusable` means reusable as an input or
implementation starting point, not automatically valid for final reporting.

| Task | Source | Accession | Existing locally | Local path | State | Reusable? | Missing work |
|---|---|---|---|---|---|---|---|
| Historical RT | NineSpecies, *Apis mellifera* | not recorded in artifact | Yes | `${DATA_ROOT}/downstream_data/retention_time/` | 60k spectra plus six cached DINO embedding Parquets | Prototype only | Define run-/peptide-disjoint splits, normalize RT/iRT, and retain provenance |
| Historical SQA dummy | synthetic | - | Yes | `${DATA_ROOT}/SQA_dataset/` | `dummy.lance` and fake-data scripts | No | Never use as a biological benchmark |
| Casanovo-style SQA | MassIVE human Orbitrap HCD | 19 MSV experiments mapped to 19 PXDs | Yes | `${DATA_ROOT}/downstream_data_casanovo/spectrum_quality/` | mzML, joint Sage results, q-value labels, balanced Parquets | Data and label pipeline reusable; split is not | Re-split by held-out run/project and audit duplicates/provenance |
| De novo sequencing | NineSpecies V2 | benchmark corpus | Yes | `${DATA_ROOT}/9_species_V2/lance_species/` | Current multi-GPU trainer/configs and recent checkpoints | Yes, for controlled initialization experiments | Use locked evaluation protocol; compare scratch, plain DINO, H26, hybrid; run frozen and fine-tuned regimes |
| Casanovo glycosylation | DQGlyco mouse-brain glycoproteome | `PXD052447` | Partial | `${DATA_ROOT}/downstream_data_casanovo/glycosylation/PXD052447/` | 468 raw filenames, 244 nonempty raw files, no conversion/search joins/labels | Download starting point only | Identify exact 48 mouse-brain HCD runs, obtain N/O FragPipe results, convert, join labels, run-level split |
| Casanovo chimericity | Carafe-derived human/mouse/yeast data | unresolved | No | no corpus found | Placeholder only | No | Recover Carafe source/code/data and reproduce wide-window FragPipe labels |
| Casanovo phosphorylation | AHLF human phosphoproteome | `PXD012174` | Partial | `${DATA_ROOT}/downstream_data_casanovo/phosphorylation/PXD012174/` | nine downloaded archives, no extracted/joined/labelled data | Download starting point only | Recover AHLF split and preprocessing, complete source acquisition, labels, and cell/tissue-disjoint materialization |
| KitchenSink pretraining | curated heterogeneous corpus | multiple | Yes | `${DATA_ROOT}/denovo_kitchensink_v3/lance/` | canonical cleaned Lance split | Yes, as a source corpus | Derive v4 with charge 1-10 and explicit locked-evaluation project/run exclusions before DINO pretraining |

## Colleague Handoff: Current Assets and Entry Points

This table is intended for contributors picking up an auxiliary downstream
task. "Runnable" means that a materialized dataset and dIon training entry
point both exist. It does not imply that the task is an exact reproduction of
an external paper protocol.

| Task | Data and current state | Existing acquisition / preprocessing code | Training code and configs | First useful contribution |
|---|---|---|---|---|
| **SQA, 19-PXD Casanovo-style** | Canonical unbalanced output: `${DATA_ROOT}/downstream_data_casanovo/spectrum_quality/casanovo_foundation_sqa_v2/`. Balanced, supported-charge derivative used for current runs: `${DATA_ROOT}/downstream_data_casanovo/spectrum_quality/casanovo_foundation_sqa_v2_balanced_charge1_10/`. It contains `train.parquet`, `val.parquet`, `test.parquet`, and a manifest. | End-to-end resumable pipeline: `misc/downstream_tasks_casanovo/spectrum_quality/prepare_casanovo_sqa.py`; fixed MSV/PXD selection: `casanovo_sqa_config.json`; balancing/charge filter: `balance_casanovo_sqa.py`; reproducibility instructions and environment dependencies: `README_casanovo_sqa.md`, `data_env.yml`, `sage14.yml`, `sage_default_params.json`. It uses ThermoRawFileParser and a separate Sage environment. | Runnable via `python -u -m src.main --config <master-config>`. Main configs: `configs/master_sqa_hybrid_casanovo_19pxd_balanced_charge1_10.yaml`, `_null.yaml`, `_random_...yaml`, and `_binned_...yaml`. Task recipes: `configs/downstream/sqa_casanovo_19pxd_frozen.yaml` and `_finetune.yaml`; the exact eight-way screening array explicitly overrides the master recipe with these files: `bash_scripts2/sqa_casanovo_19pxd_hybrid_ablation_array.sbatch`. Implementation: `src/data/sqa_inmem_data_module.py`, `src/wrappers/downstream_wrappers.py`. | Add label-efficiency runs and final locked tests using checkpoints selected by `val_auc`; keep the 19-source/file-selection uncertainty explicit. |
| **SQA, custom two-project fallback** | `${DATA_ROOT}/downstream_data_casanovo/spectrum_quality/custom_pxd004452_pxd009737_run_disjoint_v1/`. Train/validation are run-disjoint within PXD004452; PXD009737 is held out for test. | Materializer: `misc/downstream_tasks_casanovo/spectrum_quality/materialize_custom_project_disjoint_sqa.py`. It reuses local parsed spectra and Sage labels. | `configs/master_sqa_hybrid_custom_project_disjoint.yaml` with `configs/downstream/sqa_custom_project_disjoint.yaml`; same `src.main` entry point. | Preserve as a compact fallback/control, but prefer the 19-PXD benchmark for new comparisons. |
| **De novo sequencing** | KitchenSink training: `${DATA_ROOT}/denovo_kitchensink_v4/lance/`; external NineSpecies V2: `${DATA_ROOT}/9_species_V2/lance_species/`; fixed external head-10k subsets: `${DATA_ROOT}/9_species_V2/lance_species_head10000/`. | No download recovery is required for the current experiments. PA1.1 tokenizer manifest: `configs/tokenizers/pa11.json`. | KitchenSink V4: `configs/master_denovo_kitchensink_v4_encoderld_scratch.yaml`, `configs/master_denovo_dion_hybrid_kitchensink_v4.yaml`, and `configs/master_denovo_dion_hybrid_null_kitchensink_v4.yaml`. Task recipes live beside them under `configs/downstream/`. NineSpecies head-10k eval recipes: `configs/downstream/denovo_ninespecies_v2_head10000_eval_{conditioned,null}.yaml`. Entry point: `python -u -m src.main --config <master> --downstream_weights <full-downstream-checkpoint> --eval_only 1 --test_on_end 1`. | Expose and validate decoded precursor-constrained beam metrics, then run a matched conditioned-hybrid versus Pairwise baseline comparison. Frozen-encoder de novo remains unimplemented as a dedicated experiment. |
| **Retention time (historical prototype)** | `${DATA_ROOT}/downstream_data/retention_time/`. Contains `apis_mellifera_60k.parquet` and cached DINO embedding Parquets at 150/300/1000 peaks. | Historical embedding/probe code: `src/embed.py`; plots and cached artifacts are colocated under `retention_time/plots/`. | No maintained `src.main` task wrapper or current YAML exists. The historical path used a three-layer MLP with random spectrum-level 70/10/20 splits. | Reconstruct from a source with preserved run and peptide provenance; define run-/peptide-disjoint splits and an RT-normalization strategy before implementing a new task module. Do not report the historical split. |
| **Glycosylation, PXD052447** | Raw-download staging directory: `${DATA_ROOT}/downstream_data_casanovo/glycosylation/PXD052447/`. It contains 468 filenames, 244 nonempty RAW files, about 495 GB. No mzML, search-result table, labels, or split exists. | `misc/downstream_tasks_casanovo/download_glyco.py` is raw-download only. | No dIon dataset class, task wrapper, master YAML, or downstream YAML yet. An eventual global O-versus-N classifier can reuse the SQA-style frozen/finetune classifier path after a Parquet/Lance materialization is defined. | Recover the exact 48 mouse-brain HCD runs and the corresponding FragPipe N/O glycopeptide result tables; apply the stated filters, join PSMs to spectra, and materialize a fixed 36/6/6 run-level split. |
| **Chimericity, Carafe-derived** | No local corpus, manifest, raw downloads, labels, or processed data. | `misc/downstream_tasks_casanovo/msv_to_pride.py` contains only an unresolved placeholder; it is not a usable pipeline. | No training config exists. A future binary classifier can likely reuse the SQA classifier infrastructure after data materialization. | Recover the Carafe source/data and its human/train, mouse/validation, yeast/test protocol; reproduce wide-window FragPipe labels before adding code. Do not guess a PXD accession. |
| **Phosphorylation, PXD012174 / AHLF** | `${DATA_ROOT}/downstream_data_casanovo/phosphorylation/PXD012174/` contains nine archive downloads, about 264.5 GB, including `Kit255.zip`. Nothing is extracted, joined, labelled, or split. | `misc/downstream_tasks_casanovo/download_phospho.py` is raw-download only. | No dIon task config exists. A future binary classifier can reuse the SQA infrastructure only after faithful AHLF-compatible materialization. | Recover the AHLF-alpha split and preprocessing; extract the archives, obtain/parse required search results, apply PSM/protein/localization filters, and build cell/tissue-disjoint splits. Treat this as a large, lower-priority recovery. |

### Shared Training Convention

For a materialized global binary task, the current reusable pattern is:

```bash
python -u -m src.main --config configs/master_sqa_<model>_<dataset>.yaml
```

The SQA configs provide the reference implementation for a frozen encoder,
cached embeddings, a small MLP head, validation-based checkpoint selection,
and optional full fine-tuning. New tasks should create their own named dataset
directory and master/downstream YAMLs rather than silently repurposing an SQA
path or claiming an external benchmark has been reproduced.

## Historical Retention Time

`apis_mellifera_60k.parquet` contains 60,000 *Apis mellifera* spectra, 15,178
unmodified and 19,275 modified peptide identities, charges 2-5, and raw
retention times from 42.7 to 7199.8 seconds. Cached embeddings exist for the
historical larger/deeper DINO encoder at 150, 300, and 1000 peak caps.

The old code is in [`src/embed.py`](../../src/embed.py). It shuffles spectra
then uses a 70/10/20 spectrum-level split. It predicts raw RT with a small
three-layer MLP. This cannot support a new generalization claim: spectra from
the same peptide and run can occur on both sides of the split, and raw RT
retains run-scale effects. Preserve it for continuity/reproduction only.


""" """My comment: **Old RT is only a historical prototype:** 60k *Apis mellifera* spectra with cached embeddings, but random spectrum-level splits and raw RT. Not suitable for paper reporting without reconstruction.""" The main problem with this one is that is based off of ninespecies V2. Ideally we'd want to select new source data and construct a larger dataset for such claims.
But does this have peptide labels to even achieve a peptide-disjoint split? But sure youre right that we could split by runs if feasible. """

## Spectral Quality Assessment

The real SQA pipeline is under
`misc/downstream_tasks_casanovo/spectrum_quality/`. It resolves the intended
MSV split to these 19 PXDs:

`PXD000443`, `PXD000447`, `PXD000449`, `PXD000529`, `PXD000533`, `PXD000900`,
`PXD004092`, `PXD004452`, `PXD006798`, `PXD006833`, `PXD009737`, `PXD010093`,
`PXD010142`, `PXD010154`, `PXD014058`, `PXD014083`, `PXD014300`, `PXD019909`,
and `PXD020483`.

`parsed_all_with_qvals.parquet` retains all parsed MS2 spectra and labels
`peptide_q < 0.01` as high quality; unmatched spectra receive q-value 1 and
are negative. `final_data_qvals` is a balanced 50:50 derivative:

| Split | Rows | Negatives | Positives | Unique source files |
|---|---:|---:|---:|---:|
| Train | 406,064 | 203,032 | 203,032 | 19 |
| Validation | 87,014 | 43,507 | 43,507 | 19 |
| Test | 87,014 | 43,507 | 43,507 | 19 |

The existing [`final_data.py`](../../misc/downstream_tasks_casanovo/spectrum_quality/final_data.py)
uses stratified random *row-level* splits. Every one of the 19 source files is
present in train, validation, and test. These files are therefore not valid
for acquisition-disjoint final evaluation.

### Custom run-/project-disjoint replacement

The local Sage-labelled artifact is **not** a partial recovery of the Casanovo
Foundation SQA benchmark. It contains 750,867 spectra from 19 source files,
but only two projects: PXD004452 and PXD009737. The other 17 projects listed
in Casanovo Foundation Appendix S2.1, including every validation project, are
absent. The paper lists 19 MSV accessions while stating 20 runs and does not
identify the selected source file within each project, so an exact recovery
cannot be claimed from local artifacts.

`misc/downstream_tasks_casanovo/spectrum_quality/materialize_custom_project_disjoint_sqa.py`
therefore materializes a separate benchmark at:

`${DATA_ROOT}/downstream_data_casanovo/spectrum_quality/custom_pxd004452_pxd009737_run_disjoint_v1`

It retains natural Sage-label prevalence with whole-source-file splits:

| split | project | source files | spectra | positive | negative |
| --- | --- | ---: | ---: | ---: | ---: |
| train | PXD004452 | 8 | 250,289 | 77,212 | 173,077 |
| validation | PXD004452 | 3 | 62,575 | 19,734 | 42,841 |
| test | PXD009737 | 8 | 438,003 | 193,100 | 244,903 |

The validation files are selected deterministically to approximate 20% of
PXD004452 while retaining its label prevalence. PXD009737 is wholly reserved
for external project-disjoint test. The dataset manifest records all source
files and explicitly marks this as a custom benchmark, not a Casanovo
Foundation reproduction.

The current SQA wrapper and in-memory data module remain in
`src/wrappers/downstream_wrappers.py` and `src/data/sqa_inmem_data_module.py`.
They support frozen encoder caching and AUROC. The maintained frozen-hybrid
entry point is `configs/master_sqa_hybrid_custom_project_disjoint.yaml`. The
q-value label definition is weak supervision, not direct spectral quality
ground truth.
quality ground truth.

## De Novo Sequencing

The maintained path is
`configs/master_denovo_dion_ninespecies_v2.yaml` with
`configs/downstream/denovo_ninespecies_v2.yaml`. It uses
`encoder_larger_deeper`, mass and charge input, a Casanovo decoder, and the
NineSpecies V2 Lance species files. The code supports teacher forcing,
loss-only distributed validation, and beam-search precision evaluation.

Recent KitchenSink de novo checkpoints are present under:

- `${CHECKPOINT_ROOT}/pairwise_denovo_kitchensink_v1_20260721`
- `${CHECKPOINT_ROOT}/pairwise_denovo_kitchensink_v3_20260802`
- `${CHECKPOINT_ROOT}/pairwise_denovo_kitchensink_v3_clean_16gpu_17197281`

This makes de novo the immediately runnable dense-transfer benchmark. It
should be rerun with matched scratch/plain-DINO/H26/hybrid initializations,
then separately with frozen and full-fine-tuning encoder regimes. Precision
evaluation should be run separately from distributed training unless its
generation path is audited for distributed output semantics.

## Casanovo Task Recovery

### Glycosylation: `PXD052447`

The repository has only `download_glyco.py`, which downloads every public raw
file. The local directory has 468 raw filenames, but only 244 nonempty files
(494.7 GB total). No mzML, FragPipe N/O result, peak/PSM join, label file, or
train/validation/test materialization was found.

Do not treat all downloaded files as the Casanovo task. The intended benchmark
uses 48 mouse-brain HCD runs and O- versus N-glycopeptide labels after specific
FragPipe filters. Recover those exact result tables and create a 36/6/6
run-level split before training a head.

### Chimericity

No actual chimericity data, raw-file manifest, or preprocessing run was found.
`misc/downstream_tasks_casanovo/msv_to_pride.py` contains only an explicit
`MSV0000XXXXX` placeholder. The Casanovo paper's human/mouse/yeast protocol
must be recovered through Carafe's source/data trail; no accession should be
invented.

### Phosphorylation: `PXD012174`

Nine archive files are present (264.5 GB total), including `Kit255.zip` at
148.4 GB. No extracted spectra, AHLF split, search-result join, label table,
or Lance/Parquet materialization exists. `download_phospho.py` is raw-download
only. This is not operational and should be deferred until a faithful AHLF
split/preprocessing source is recovered.

## KitchenSink v3 to v4

The canonical v3 source is `${DATA_ROOT}/denovo_kitchensink_v3/lance`.
Use this cleaned root, not `lance_pre_cleanup`. Its final cleanup audit reports
5,957,180 / 175,767 / 349,571 rows for train/validation/test, zero duplicate
physical spectra, zero duplicate exact peak lists, and zero cross-split
peptidoform overlap.

Its provenance is strong but does not make it safe against the DINO paper's
external tests. It includes Kingdoms, PXD010000 regenerated v2,
ProteomeTools `PXD004732`, `PXD010595`, `PXD021013`, MKB2, MSKB v5 seed,
nonenzymatic MassIVE-KB, and highcharge-v3. The existing v4 builder at
`${DATA_ROOT}/denovo_kitchensink_v4/scripts/build_kitchensink_v4_dino.py`
already derives a non-destructive view with charges 1-10 and configurable
project/run/source exclusions. Populate its config from the locked benchmark
provenance before using KitchenSink for DINO pretraining. It intentionally
does not apply peptide filtering.

## Recommended Order

1. Rebuild the existing SQA q-value corpus with run/project-disjoint splits;
   this has the shortest path to a real global downstream task.
2. Rerun matched de novo experiments for hybrid against the current controls;
   this is already operational and tests dense transfer.
3. Recover/curate PXD052447 glycosylation as the next real local-fragment
   global task.
4. Recover the Carafe chimericity protocol and data; it is highly aligned with
   distractor training, so report it alongside unrelated transfer tasks.
5. Rebuild RT only after source provenance permits run-/peptide-disjoint,
   normalized-target evaluation.
6. Defer phosphorylation until the AHLF source split is reproducible.

For all revived tasks, freeze evaluation data and task heads before comparing
new objective variants; retain metadata-only and scratch/binned baselines, and
add 1%/10%/100% label-efficiency curves to at least two global tasks.
