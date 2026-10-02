# Kingdoms Fast Evaluation

Computes global de novo metrics on the Kingdoms run-disjoint test set without
writing large prediction tables. Per-species prediction export is a separate
workflow.

## Candidate Models

Evaluate these precursor-conditioned de novo model classes:

1. Hybrid-300 finetuned at 200 peaks.
2. Hybrid-300 finetuned at 1000 peaks.
3. Scratch encoder-LD finetuned at 200 peaks.

Evaluate each available training-origin counterpart separately:

- MSKB-Final-trained.
- DNLv1-trained.

## Fast Evaluation Contract

- Dataset: `denovo_kingdoms/run_disjoint_v1`.
- Mode: `eval_only`, `test_on_end=1`, no training and no validation pass.
- Do not enable `log_predictions`: this avoids multi-gigabyte mzTab files and
  the prediction writer's rank-zero deduplication state.
- Retain ordinary W&B logging. Use a project specific to each training data
  origin, so origin comparisons do not mix with training runs.
- Metrics: existing Casanovo-compatible global test loss, amino-acid
  precision/recall, and peptide precision.

## Per-Species Metrics

The fast evaluation uses a distributed-safe per-species accumulator over the
aggregate capped table. W&B therefore receives the ordinary global metrics and
`denovo_tf_test_species_<species>_{n_spectra,pep_prec,aa_prec,aa_recall}` for
every species, without generating prediction files. The one-species Lance
datasets remain available for debugging or prediction exports.

## Test-Set Scale and Optional Cap

`run_disjoint_v1/test.lance` is already a dIon-compatible Lance dataset.
It has `14,519,567` test spectra across 70 species, with a top-heavy count
distribution: the 15 largest species comprise 52.65% of rows.

The exact full test set remains the reference evaluation. For a lower-cost
screening evaluation, use a deterministic, species-stratified capped copy:

| Per-species cap | Retained spectra | Fraction of full test | Species capped |
| ---: | ---: | ---: | ---: |
| 50,000 | 3,050,761 | 21.01% | 52 |
| 100,000 | 4,926,232 | 33.93% | 33 |
| 200,000 | 8,106,646 | 55.83% | 28 |

At the measured 200-peak one-GPU rate, the full set is about 56.8 GPU-hours
per model. The 50k cap is about 11.9 GPU-hours and the 100k cap about 19.3
GPU-hours; ideal 16-GPU wall times are about 45 and 72 minutes respectively,
before distributed and data-staging overhead.

The capped export contains both `test.lance` for aggregate evaluation and
`test_by_species/<species>.lance` for direct single-species evaluation. The
manifest records the path and retained count for every species dataset.

Any capped dataset must be materialized deterministically using a stable hash
of source identity within each species and recorded as a distinct capped
benchmark. Results from it must not be presented as full-set results.
