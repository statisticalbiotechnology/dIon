# Kingdoms Nearest-Neighbor Benchmark

## Purpose

This benchmark asks whether a learned MS2 embedding improves same-peptide
nearest-neighbor retrieval over conventional spectral similarity. It includes
broad retrieval, precursor-controlled hard retrieval, and static-pair
discrimination. The planned comparisons include frozen DINO embeddings,
supervised-contrastive fine-tunes, GLEAMS, Casanovo encoder embeddings, and a
peak-only binned spectral-cosine/spectral-angle baseline.

## Data And Splits

Source data are under `${DATA_ROOT}/kingdoms/processed`. Retained
rows satisfy `Qvalue <= 0.01`, `chimeric == false`, positive charge, and
positive observed precursor m/z. Materialized artifacts contain charges 1--6.

The immutable artifacts are under:

```text
${DATA_ROOT}/probing_datasets/kingdoms_paper_benchmarks/
```

| Split | Role | Retrieval spectra | Pair-endpoint spectra | Pair records |
| --- | --- | ---: | ---: | ---: |
| `validation/` | Development and model selection | 658,738 | 1,085,786 | 4,553,362 |
| `test/` | Locked final reporting | 4,112,590 | 7,138,447 | 28,487,960 |

For the ten species with multiple acquisition batches, validation holds out a
whole filename-derived acquisition batch selected deterministically. The test
contains remaining raw files and all other species. Thus it is run-disjoint
where the source exposes multiple acquisition batches, but is not claimed to
be universally project-disjoint. The locked test is not used to select models.

## Retrieval

Retrieval is evaluated independently within each species. The truth label is
`peptide_id`; only peptide identities represented by at least two spectra are
eligible. One deterministic query spectrum per identity is selected with seed
`20260817`; the gallery contains all other spectra from that species.

The broad task is ordinary same-peptide nearest-neighbor retrieval using cosine
similarity. It reports mAP, MRR, Recall@k, and Hit@k. Broad retrieval is useful
but can be made artificially easy by precursor mass and charge.

### Same-Charge, Within-10-ppm Retrieval

The central hard retrieval result restricts candidates to spectra that:

- have the same precursor charge as the query;
- have observed precursor m/z within 10 ppm of the query; and
- include both a same-peptide positive and a different-peptide negative.

This removes the direct precursor/charge shortcut and asks whether the
representation separates fragment-pattern evidence among locally
precursor-matched candidates. Queries without both a positive and negative in
the restricted gallery are excluded and the eligible-query fraction is
reported.

The full report also includes cross-charge retrieval. Its strict variant
requires a different charge state and neutral-mass agreement within 10 ppm.

For practical development, validation retrieval is capped at 20,000 peptide
identities per species and 10 spectra per identity. Locked test evaluation has
no such cap.

## Static Pair Discrimination

The pair task uses fixed labelled pairs, reused exactly for every
representation:

- Positive: exact same modified sequence and precursor charge.
- Negative: different peptide identity.

Each species has balanced 1:1 positive:negative pair sets:

1. `all_random`
2. `same_charge_random`
3. `same_charge_10ppm`: different-peptide negatives with the same charge and
   precursor m/z within 10 ppm.

The third set is the primary hard pair comparison. With cosine, a pair score
is cosine distance, `1 - cosine_similarity`. Reported metrics include ROC-AUC
and average precision, plus FNR at fixed balanced-pair FDR operating points.
Balanced-pair FDR means accepted negatives divided by all accepted pairs in a
deliberately 1:1 sampled pair set; it is an embedding-discrimination diagnostic,
not a database-search FDR estimate.

Metrics are computed per species and macro-averaged so large organisms do not
dominate the result.

## Code And Configurations

Configurations:

```text
configs/probing_tasks/kingdoms_paper_validation_retrieval.yaml
configs/probing_tasks/kingdoms_paper_validation_pairs.yaml
configs/probing_tasks/kingdoms_paper_test_retrieval.yaml
configs/probing_tasks/kingdoms_paper_test_pairs.yaml
```

Standard runners:

```text
scripts/evaluate_embeddings.py
scripts/evaluate_pair_discrimination.py
```

Metric implementations:

```text
src/embed_eval/peptide_metrics.py
src/embed_eval/pair_metrics.py
```

The full benchmark card is `docs/datasets/dino_ms2_embedding_benchmark_card.md`.

## Interpretation

The hard 10-ppm metrics are essential. A strong result on broad retrieval can
reflect precursor metadata or ordinary spectral-library-like matching rather
than an embedding that captures peptide-relevant fragment structure. The
same-charge/10-ppm retrieval and pair sets make that distinction explicit.
