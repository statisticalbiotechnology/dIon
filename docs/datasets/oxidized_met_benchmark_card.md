# Oxidized-Methionine Benchmark (dataset card)

Binary task: given an MS2 spectrum of a **methionine-containing** peptide, is
the methionine oxidized (M+15.995)? Scored by AUROC.

Purpose: the dIon benchmark suite has a global-binary task (SQA), a dense
generative task (de novo), and a co-isolation task (chimericity), but no
*local fragment* task -- the axis the Casanovo suite covers with glycosylation
and phosphorylation, both of which need large recoveries before they exist.
Detecting a +15.995 shift on the fragment ions containing one residue is that
axis, available today at zero acquisition cost.

Materialized at
`${DATA_ROOT}/oxidized_met_benchmark`.

## Corpus and labels

PRIDE PXD010613, the held-out project of the dIon bacterial corpora -- no
spectrum here was seen during pretraining. Labels are free: the v3 label
policy writes oxidation as `M+15.995` in `seq` and ignores every other
modification, so no database search is required. Weak supervision, like SQA's
q-values: it is MSGF+'s assignment, not physical ground truth.

## Why the negatives are restricted

The universe is spectra whose peptide contains at least one M; negatives are
unoxidized *M-containing* peptides. Against all peptides the task collapses
into "does this peptide contain M at all", which is a composition question
answerable from mass and amino-acid statistics rather than from the fragments.

## Splits (species-disjoint)

| split | species | rows | oxidized | matched-backbone rows | unique backbones |
| --- | --- | ---: | ---: | ---: | ---: |
| train | *C. crescentus* | 135,926 | 40.0% | 52,449 (38.6%) | 56,173 |
| val | *E. faecalis* + *A. muciniphila* | 60,511 | 45.6% | 22,922 (37.9%) | 30,097 |
| test | *H. congolense* | 181,479 | 38.3% | 88,547 (48.8%) | 50,109 |

377,916 spectra, prevalence 38-46%. Species-disjointness delivers near
peptide-disjointness for free: only 81 backbones are shared between train and
test (0.14% of either split's unique backbones), 59 between test and val, 58
between train and val.

## The sequence-controlled subset

`matched_backbone == 1` marks spectra whose backbone occurs **both** oxidized
and unoxidized *within the same split*. On that subset the peptide is
controlled by construction, so a model cannot score by recognising which
peptides tend to be oxidized -- only by reading the modification off the
fragments. **Report both numbers**: full-set AUROC and matched-subset AUROC.
The matched subset is large (23k-89k rows per split) and its prevalence is
lower (0.31 train / 0.43 val / 0.31 test) than the full set's.

## Baselines (measured, train -> test)

| model input | AUROC |
| --- | --- |
| precursor m/z + charge | **0.523** (chance) |
| precursor m/z + charge, matched subset | **0.442** (below chance) |
| + `n_methionine` | 0.628 |

The first two rows are the point: the task is not solvable from precursor
metadata, and on the matched subset metadata is actively useless. The third
row is a **warning, not a baseline** -- `n_methionine` is an analysis column
describing the answer's opportunity space (more methionines, more chances one
is oxidized). It is not a model input and must never be fed to one; it is
stored only so the peptides-per-M distribution can be audited.

## Shortcut guards

- `seq` is **not** stored. Only `backbone` (the sequence with M unmodified),
  because the labelled sequence would hand over the answer.
- `matched_backbone` as above.
- Cross-split backbone overlap is reported in the manifest; keep it near zero
  if the splits are ever redrawn.

## Rebuild

```bash
python scripts/materialize_oxidized_met_benchmark.py \
    --lance <pooled .../PXD010000__PXD010613__PXD019911__PXD019912_ms1/test.lance> \
    --output-root <out>
```

Convert the immutable Parquet output for normal streaming dIon training:

```bash
python scripts/data_prep/convert_auxiliary_benchmark_parquet_to_lance.py \
    --input-root ${DATA_ROOT}/oxidized_met_benchmark
```

The resulting `lance/` directory is the data root. The task loader must use
only the spectrum and precursor fields plus `label`; `backbone`,
`n_methionine`, `n_oxidized`, and `matched_backbone` are audit/evaluation
fields and must never be model inputs.
