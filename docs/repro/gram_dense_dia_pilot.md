# Gram dense-token and DIA pilot

Branch: `experiment/gram-dense-dia`, based on `run/denovo-mskb-final` at
`4adb45e`.

## Checkpoints

```text
Pre-Gram Hybrid-300 EMA teacher
${CHECKPOINT_ROOT}/checkpoint_12_59_01_073022__03_09_26/last.ckpt

Gram-refined EMA teacher, epoch 298
${CHECKPOINT_ROOT}/checkpoint_02_51_05_597141__13_09_26/epoch=298-dinov2_val_loss_epoch=6.43.ckpt
```

Dense probing loads only exact `teacher.backbone.*` weights. Gram-teacher and
projection-head state are excluded, so both checkpoints use the same encoder
representation definition.

## Synthetic dense precursor swap

Let `A` and `B` be clean spectra and `M = merge(A, B)` their fixed synthetic
mixture. Peak tensor `M` is held fixed and encoded twice:

```text
E(M, precursor(A))
E(M, precursor(B))
```

The mixer labels each post-merge peak as A-derived, B-derived, or merged A/B.
For an individually retained A-derived peak at m/z `z`, align its token slot in
`E(M, *)` to the clean-A token slot at matching `z` within 0.01 ppm. The
A-query advantage is:

```text
cos(E(M, precursor(A))[z], E(A, precursor(A))[z])
- cos(E(M, precursor(B))[z], E(A, precursor(A))[z])
```

Average this over A-derived peaks. Define the B metric symmetrically. A
mixture is paired-aligned only when both average advantages are positive.
Merged A/B peaks are excluded because one mixed token cannot be assigned a
single clean-source target.

The apparent "opposite" comparison for an A-derived slot is undefined: it has
an m/z-aligned clean-A token, but generally no corresponding clean-B token. An
all-to-all clean-A versus clean-B token-set score is possible, but requires an
extra set reduction or assignment (for example nearest-neighbor, Chamfer, or
optimal transport). That discards the local peak correspondence and largely
becomes another pooled/set-level source-selection readout. It may be a useful
bridge to the original global metric, but is not the primary dense test.

This is an m/z-to-token-slot alignment, not a claim that a dense token belongs
only to that peak: every final token is contextualized by all peaks and the
precursor. Also, `len(E(M, *))` is generally closer to `len(M)` than `len(A)`
or `len(B)`. The precursor can make mixed token states more A-like or B-like,
but it does not remove the competing-source token slots.

### Completed results

```text
Pre-Gram report
${RESULTS_ROOT}/gram_dense_dia_synthetic_precursor_swap_v1/pre_gram_hybrid300.json

Gram-refined epoch-298 report
${RESULTS_ROOT}/gram_dense_dia_synthetic_precursor_swap_v1/gram_refined_epoch298.json

Pairs per checkpoint: 4,488
  500 pairs in eight species; 488 in Solanum lycopersicum
Skipped pairs: 0
Missing A/B source m/z matches: 0
Matched A-derived token slots: 619,502
Matched B-derived token slots: 469,420
```

| Encoder | Summary | A alignment | B alignment | Paired alignment | Mean advantage | Mean token cosine distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pre-Gram Hybrid-300 | Species macro | 0.99933 | 0.98193 | 0.98149 | 0.57421 | 0.76389 |
| Pre-Gram Hybrid-300 | Pooled | 0.99933 | 0.98195 | 0.98151 | 0.57439 | 0.76401 |
| Gram-refined epoch 298 | Species macro | 0.99978 | 0.98861 | 0.98839 | 0.48851 | 0.59793 |
| Gram-refined epoch 298 | Pooled | 0.99978 | 0.98864 | 0.98841 | 0.48866 | 0.59805 |

Gram refinement increases paired alignment from 0.98151 to 0.98841, while
reducing mean advantage and query-induced token distance. This is a dense
clean-context recovery result only; it does not establish literal per-peak
source attribution, physical spectrum decomposition, or target-peptide
decoding from a mixture.

## De Novo Decoder Protocol

Existing frozen dIon-plus-decoder checkpoints were trained on single-source
spectra. Their decoders were not trained to suppress B-derived evidence in the
longer mixed sequence when querying A. The decoder protocol freezes dIon and trains or fine-tunes a decoder with both targets per mixture:

```text
M + precursor(A) -> peptide(A)
M + precursor(B) -> peptide(B)
```

The implemented Gram-refined DNLv1 recipe is:

```text
config_cluster_b/downstream/denovo_dnlv1_dion_hybrid_gram_target_swapped_frozen.yaml
jobs/ClusterB/denovo_dnlv1_hybrid60_gram_refined_target_swapped_frozen_32gpu_array.sbatch
```

Training-only uses one 95%-A `random_batched` global crop and the shared
`BatchedStudentDistractorMixAugmentation` unchanged: equal-size random B
subset, 10 ppm condition and neutral-mass exclusions, 5 ppm merging, and no
final intensity normalization. One B is selected per A, producing the two target-conditioned decoder rows. The 32-GPU job uses 25 anchors per rank, expanding to 50
rows per rank and retaining the established global decoder batch of 1,600. It
runs 80 anchor epochs (`527680` steps) with the cosine period stretched to the
same horizon; the `6250`-step warmup is unchanged. Its 36-hour request allows
for the longer mixed inputs. Validation is clean DNLv1 and selects
`denovo_tf_val_pep_prec`; no transfer test is launched.

On 2026-09-22, the epoch-298 Gram checkpoint loaded and completed a real DNLv1
one-step smoke on n99 at the original per-rank stress batch: 50 anchors -> 100
mixed target rows, with 127M encoder parameters frozen and 94.6M decoder
parameters trainable.

## Real DIA Dataset

```text
${DATA_ROOT}/dion_aux_benchmarks/dia_fragment_v1/
```

The benchmark contains target-conditioned DIA scan/precursor rows with top-100
peaks and per-peak b/y labels: train 9,707, validation 2,651, test 5,423.
Splits are peptide-disjoint, but scan IDs can cross splits. It uses wide DIA
windows (minimum 26 m/z), and non-target peaks are not confirmed assignments
to a competing peptide. Audit scan overlap and mass-separation strata before
using it for a learned readout or decoder evaluation. A narrow-DIA decomposition evaluation requires a separate narrow-window dataset with trustworthy component labels.

## Oracle-precursor DIA decoder path

The complete immutable Lance materialization is:

```text
${DATA_ROOT}/dia_fragment_v1_oracle_precursor_pa11_unimod4_union/
```

It preserves `train.lance`, `val.lance`, and `test.lance`, and adds
`all_splits_test.lance` for selected-model evaluation over every target query.
The union has exactly 17,781 rows, 17,781 unique `(scan, target peptide,
precursor)` queries, and 9,973 physical MS2 scans. DIA-NN's only observed
unsupported notation, `C(UniMod:4)`, is fail-closed translated to the PA1.1
token `C+57.021`; no rows were rejected. The manifest records source hashes,
translation, counts, and cross-split scan overlap.

This is an **oracle-precursor mixture-decoding** evaluation:

```text
(DIA MS2 mixture, target precursor from DIA-NN) -> target peptide
```

It tests whether the fixed dIon-plus-decoder model can use a supplied target
precursor to decode one labeled component from a real mixture. It is not an
MS1 precursor-feature selection/decomposition experiment: the shared Parquet
benchmark does not retain the preceding MS1 scan or its candidate feature list.

`config_cluster_b/downstream/denovo_dia_fragment_v1_oracle_precursor_200peaks_prediction.yaml`
is a direct copy of the established Kingdoms 200-peak prediction recipe, with
only Lance paths and DIA provenance fields changed. It retains 200 peaks,
conditioned precursor input, one autoregressive beam, native peptide confidence,
streamed CSV writing, and exact distributed evaluation sharding. The CSV now
includes canonical index, scan, split/source row, target precursor, and DIA
isolation window; it flushes after each batch and rejects duplicate indices.

A real-GPU smoke loaded the selected target-swapped frozen checkpoint
`epoch=8-denovo_tf_val_pep_prec=0.40.ckpt`, decoded 200 DIA target queries,
and wrote 200 unique canonical rows with all query fields populated and no
no-predictions. The disposable smoke result is under:

```text
${RESULTS_ROOT}/gram_dense_dia/oracle_precursor_target_swapped_epoch8_smoke_20260923_03/
```

### Full target-swapped frozen-decoder result

The selected stable checkpoint from the target-swapped training run was
`epoch=8-denovo_tf_val_pep_prec=0.40.ckpt`. Later training steps became
numerically unstable, so this is an early selected checkpoint rather than the
terminal epoch. On 2026-09-23 it was evaluated once over the complete oracle
union using autoregressive one-beam decoding:

| Metric | Value |
| --- | ---: |
| Target-query denominator | 17,781 |
| Peptide precision at 100% coverage | 0.105675 |
| AA precision | 0.225070 |
| AA recall | 0.227574 |
| Teacher-forced loss | 1.675812 |
| Runtime | 3m36s (178 batches; 0.87 batch/s) |
| Precision-coverage AUC | 0.327754 |

```text
Run output
${RESULTS_ROOT}/gram_dense_dia/oracle_precursor_target_swapped_epoch8_full_20260923_222754/

Prediction CSV
.../logs_22_28_08_909951__23_09_26/predictions.csv

CSV SHA-256
48816982e2789290315e849302ce171989e002f80ec06bb19272e3916e21b8e6
```

The model was pretrained only on synthetic two-source asymmetric mixtures,
with constrained B-peak selection, and the decoder used a limited synthetic
target-swapped training regime. This protocol supplies the DIA-NN target
precursor directly; it does not evaluate practical precursor selection or
spectrum deconvolution. Evaluating those tasks requires observed MS1 precursor
features, candidate feature lists, and narrower DIA windows.

### Transfer bundle

The import-ready bundle is:

```text
${RESULTS_ROOT}/gram_dense_dia/oracle_precursor_target_swapped_epoch8_full_20260923_222754_transfer_bundle_v1/
```

It contains a copied validated `predictions.csv`, corpus-level canonical-matcher
`metrics.json`, confidence-ranked `precision_coverage.csv`, and `manifest.json`
with SHA-256 values for every transferred artifact, checkpoint, config, dataset
manifest, and source prediction CSV. The package's corpus-level peptide precision
is `0.1056745965` and precision-coverage AUC is `0.3277542729`.
