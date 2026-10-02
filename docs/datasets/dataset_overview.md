# Dataset Overview

This inventory maps each evaluation task to its authoritative corpus and split.
Detailed split construction and evaluation rules remain in
[embedding_evaluation_protocol.md](../repro/embedding_evaluation_protocol.md).

## Primary Representation Benchmarks

The charge-complete frozen-embedding table compares dIon representations,
random-transformer controls, binned-spectrum cosine, and Casanovo Foundation
v4.0.0. It uses locked tests, never compact callback subsets.

| Benchmark | Validation / callback artifact | Held-out dataset | Retrieval manifest | Static-pair manifest |
|---|---|---|---|---|
| Bacterial | Full `bacterial_paper_validation_*` manifests; compact `bact_val_cb` online callback subset | PXD010613; run-disjoint from PXD010000 training/validation | `bacterial_paper_test_retrieval.yaml` | `bacterial_paper_test_pairs.yaml` |
| NineSpecies V2 | Full `ninespecies_v2_paper_validation_*` manifests; compact `9spec_v2_val_cb` online callback subset | Locked NineSpecies test | `ninespecies_v2_paper_test_retrieval.yaml` | `ninespecies_v2_paper_test_pairs.yaml` |
| Kingdoms | Full `kingdoms_paper_validation_*` manifests; compact `kingdoms_val_cb` online callback subset | Locked Kingdoms paper test | `kingdoms_paper_test_retrieval.yaml` | `kingdoms_paper_test_pairs.yaml` |

Each benchmark reports broad same-peptide retrieval and static-pair
discrimination, including the same-charge, within-10-ppm ROC-AUC and average
precision slice. Full validation uses the corresponding `*_paper_validation_*.yaml`
manifests for model selection.

The current GLEAMS validation comparison uses a separately materialized
charge-2--4 matched table. It is an external-reference comparison and must not
be combined with the charge-complete primary table. Casanovo Foundation v4 is
tracked on its own retained cohort according to its released preprocessing.

## Downstream And Probe Tasks

| Task | Dataset and split | Validation / callback availability | Role |
|---|---|---|---|
| Spectrum quality assessment | Casanovo-style balanced 19-PXD corpus, charge 1--10 | Dedicated validation split; no generic embedding callback requirement | Global binary downstream task; complete held-out test has 151,256 examples. |
| De novo sequencing | dIon-de-novo-labeled-v1 (DNLv1) training/validation; NineSpecies V2 held-out test | dIon-de-novo-labeled-v1 (DNLv1) `val.lance`; optional named external validations | Primary dense/generative downstream task. dIon-de-novo-labeled-v1 (DNLv1) preserves the official `mskb_final` test component in its test split. |
| Dense de novo probe | Derived short-peptide `mskb_final` probe from official training data | Fixed train/validation/test probe split, invoked as an online callback | Fast developmental dense-representation monitor, not a primary benchmark. |
| Supervised metric learning | Bacterial PXD010000/PXD010613 annotated v3 corpus; nested 1%, 10%, and 100% peptide-identity training manifests | Dedicated validation data plus canonical bacterial `bact_val_cb` / full representation validation callbacks | Label-efficiency fine-tuning task. Test uses canonical bacterial representation manifests. |
| Chimericity, primary | PXD024584 HYE (HeLa, yeast, and *E. coli*) | Dedicated HYE validation split (45,886 spectra) | Co-isolation binary task; train/validation/test have 79,940 / 45,886 / 47,411 spectra. |
| Chimericity, external | PXD010613 four-species corpus | Its own validation split, but external-only and not used for primary checkpoint selection | Out-of-corpus species-held-out evaluation. |
| Oxidized methionine | PXD010613 four-species corpus, species-disjoint split | Dedicated validation split (60,511 spectra) | Local-fragment PTM-chemistry binary task; train/validation/test have 135,926 / 60,511 / 181,479 spectra. |
| Retention-time probe | PXD010613 bacterial corpus, species- and peptidoform-disjoint split with aligned normalized RT targets | Dedicated validation split (130,294 spectra) | Frozen global representation probe; ordinal and scalar-regression heads are compared. Train/validation/test have 400,429 / 130,294 / 411,846 spectra. |
| Glycosylation / phosphorylation | PXD052447 / PXD012174 | No validated dataset split is specified | PTM-chemistry task without a validated protocol. |

## Dataset And Result Discipline

- Store immutable generated records for each reported evaluation, as specified
  by the evaluation protocol. These artifacts are not bundled with the source.
- Record source dataset artifacts, filtering, preprocessing, conditioning,
  peak cap, and any `limit_*_batches` setting in the experiment manifest.
- Do not compare callback, validation, locked-test, or externally
  preprocessing-filtered results as though they came from the same cohort.
- External models retain their released preprocessing. dIon comparators on
  an external matched cohort must use exactly the retained rows.
