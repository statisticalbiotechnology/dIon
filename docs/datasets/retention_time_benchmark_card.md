# Retention-Time Benchmark (dataset card)

Predict where in the chromatographic gradient a peptide eluted, from its MS2
spectrum alone. Materialized at
`${DATA_ROOT}/retention_time_benchmark`.

**No external precedent.** The Casanovo Foundation paper has no RT task -- its
four are spectrum quality, chimericity, phosphorylation and glycosylation, and
retention time appears only in its background section describing prior work.
So there is nothing to reproduce and no corpus of theirs to reuse; every
choice below is ours and is justified on measurement. This also replaces the
historical *Apis mellifera* prototype, which cannot support a generalization
claim (random spectrum-level splits, raw RT, NineSpecies-derived).

## Intended use: relative comparison, not absolute claims

This is an auxiliary probe for **ranking trainings against each other**, not a
task whose absolute score means anything externally. Two consequences:

* There is no external number to match and no protocol to imitate, so the
  design is chosen on measurement (above) rather than on precedent.
* The task need not probe an axis independent of de novo sequencing. Predicting
  RT from fragments requires inferring composition/hydrophobicity, so it
  overlaps with sequencing by construction. That is acceptable for relative
  use; it would not be acceptable as evidence of an independent capability.

What relative use *does* require, and what must be reported with any ranking:

1. **The metric's own noise.** Fit the probe head with >= 3 seeds on one frozen
   checkpoint and report the spread. A difference between two trainings smaller
   than roughly twice that spread is not a difference -- the same rule the
   pretraining screening protocol uses for `same_charge_10ppm`.
2. **The metadata floor** (0.167 normalized / 15.9 min, r = 0.66). A model that
   does not clearly beat it has learned nothing chromatographic, whatever its
   rank.
3. **Frozen eval data, frozen head, identical preprocessing** across every
   model compared. Per-split floors differ ~2x (see below), so scores are
   comparable within a split and never across splits.

## Target: normalized, not raw minutes

Raw RT carries per-run offset and gradient scale that the model cannot observe
and therefore cannot predict -- as a target it is partly irreducible noise.
The field agrees: Prosit predicts indexed RT (iRT) and reports Delta-t95;
DeepLC predicts native units but pairs them with a per-run piecewise-linear
calibration, i.e. treats cross-run comparability as calibration rather than as
a unit.

Three columns are stored:

| column | meaning |
| --- | --- |
| `rt_seconds` | raw, so any other scale (iRT included) stays recomputable |
| `nrt` | position in the run's own gradient, robust 0.5/99.5 percentiles -> [0,1] |
| `nrt_aligned` | **primary target**: `nrt` after isotonic alignment onto the species' reference run, over shared peptidoforms |

All 23 runs share a 90-96 min gradient, so percent-of-gradient already removes
most run-to-run variation; isotonic alignment absorbs the remaining nonlinear
gradient-shape differences and measurably helps (replicate spread 2.24 -> 1.90
min). Alignment is **within-species**, and splits are species-disjoint, so a
run is only ever calibrated against runs in its own split -- no cross-split
leakage.

## Splits: species-disjoint, then peptidoform-disjoint

| split | species | rows | peptidoforms | target mean/std | within-split floor |
| --- | --- | ---: | ---: | --- | ---: |
| train | *C. crescentus* | 400,429 | 140,506 | 0.445 / 0.261 | 0.0175 |
| val | *E. faecalis* + *A. muciniphila* | 130,294 | 67,430 | 0.471 / 0.269 | 0.0122 |
| test | *H. congolense* | 411,846 | 113,401 | 0.486 / 0.263 | 0.0228 |

RT is a property of the peptide, so a peptidoform appearing in two splits is a
memorizable answer. Species-disjointness nearly guarantees disjointness
already; the 432 peptidoforms that still spanned splits are **dropped**, making
the split exactly peptidoform-disjoint.

## Model inputs, and the one column that must not be one

The task is **spectrum -> RT**. Model inputs are `mz_array`,
`intensity_array`, `precursor_mz`, `precursor_charge` -- the same input
semantics as every other dIon task, so this drops into the existing
embedding path unchanged. The target is `nrt_aligned`.

`seq` **is stored but is not a feature.** Retention time is essentially
determined by sequence (hydrophobicity), and sequence-based predictors solve it
to near the replicate floor, so feeding `seq` to a model measures
sequence -> RT and is a different, far easier task. It is kept for three
legitimate uses: auditing peptidoform-disjointness of the splits, grouping
replicate observations, and building a deliberate **sequence oracle** baseline
-- a sequence-only predictor is a useful upper reference, since it says how
much of RT is sequence-determined at all, and therefore how much a
spectrum-only model could in principle recover. Report it as an oracle, never
as a result for this task. The same applies to `peak_file`, `species` and
`aligned`: metadata for analysis and grouping, not inputs.

## Measured dynamic range (normalized units; x95 for approximate minutes)

| | MAE | as minutes |
| --- | ---: | ---: |
| predict-the-mean | 0.234 | 22.3 |
| precursor m/z + charge (gradient boosting) | 0.167 | 15.9 |
| **noise floor** (replicate spread, aligned) | **0.020** | **1.90** |

The metadata baseline reaches Pearson r = 0.66 -- peptide mass genuinely
correlates with hydrophobicity, so **report it alongside any model number**; a
model that does not clearly beat r = 0.66 has learned nothing chromatographic.
Roughly eight-fold headroom separates that baseline from the floor.

**Caveat on Delta-t95.** The replicate spread is heavy-tailed: MAE 1.90 min but
Delta-t95 ~9.9 min. The tail is most plausibly misassigned PSMs (a spectrum
labelled with the wrong peptide lands far from that peptide's median) rather
than chromatographic irreproducibility. Report both statistics; do not read
Delta-t95 as a chromatographic bound, and do not filter on it to flatter a
model, since filtering on the target is circular.

## Recommended head: soft-label ordinal, not scalar regression

Transformers regress poorly -- a scalar MSE head is badly conditioned and
outlier-dominated. `src/soft_ordinal.py` implements the standard alternative
(AlphaFold's distogram is the canonical instance; DORN and label-distribution
learning are the same family): discretize the target, train against a soft
distribution centred on the truth, decode a continuous value as the
expectation over bin centres.

```python
from src.soft_ordinal import BinSpec, soft_labels, soft_cross_entropy, decode_expectation

spec = BinSpec(low=0.0, high=1.0, n_bins=64, sigma=0.01993)   # from the manifest
loss = soft_cross_entropy(logits, soft_labels(nrt_aligned, spec))
prediction = decode_expectation(logits, spec)
```

`sigma` is **not a hyperparameter**: it is the measured replicate spread, so
the label's fuzziness is exactly the measurement's own uncertainty. Bin width
(1/64 = 0.0156) follows from it. `decode_std` returns the head's uncertainty
for free, and `decode_mode_refined` (parabolic interpolation at the peak) is
the better decoder when the posterior is bimodal, where the expectation gets
dragged between modes. Keeping a scalar-regression head as a control is worth
it -- the comparison is itself a result.

## Rebuild

```bash
python scripts/materialize_retention_time_benchmark.py \
    --lance <pooled .../PXD010000__PXD010613__PXD019911__PXD019912_ms1/test.lance> \
    --output-root <out>
```

Convert the immutable Parquet output for normal streaming dIon probing:

```bash
python scripts/data_prep/convert_auxiliary_benchmark_parquet_to_lance.py \
    --input-root ${DATA_ROOT}/retention_time_benchmark
```

The resulting `lance/` directory is the data root. The RT probe must select
only spectrum/precursor fields and `nrt_aligned`; `seq`, run/species metadata,
and other target/audit columns are not model inputs.
