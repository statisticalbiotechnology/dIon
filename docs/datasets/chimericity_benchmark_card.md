# Chimericity Benchmark (dataset card)

Binary task: given one MS2 spectrum, was it produced by a single peptide or by
several co-isolated peptides? Scored by AUROC. Built to the protocol of the
Casanovo Foundation chimericity task (arXiv 2505.10848), which reports 0.780
for their model against 0.684 (binned embedding) and 0.711 (end-to-end
transformer).

## Why this is not a reproduction

The reference corpus is Carafe-derived human/mouse/yeast data. Neither the
Casanovo paper nor the Carafe preprint publishes an accession for it -- the
paper says only that samples were "prepared using the method described in
[41] and analyzed using an Orbitrap Fusion Lumos". An exact rebuild is
therefore impossible, and no accession should be guessed. What follows is the
*same protocol* on named, independent corpora. Report it that way.

## Label

FragPipe 23.0 / MSFragger 4.2, stock `WWA.workflow` (wide precursor window
-20/+20 Da, `deisotope=1`, `output_report_topN_dda_plus=5`), data type `DDA+`,
each corpus searched against its own target + reversed-decoy database.

- `label` = 1 when more than one distinct peptide passes PSM q <= 0.01.
- Spectra with no wide-window assignment are **excluded**, not labelled
  negative (as in the reference protocol).
- `label_cross_species` = 1 when the co-identified peptides come from
  *different organisms*. This is an addition, definable only in a multi-species
  mixture, and it is the stricter label: within one proteome, two peptide
  assignments can reflect homology or search ambiguity, whereas a human peptide
  plus a yeast peptide in one spectrum is co-isolation beyond argument. On HYE,
  61% of `label` positives are also `label_cross_species` positives.

## Corpora

| | primary | secondary |
| --- | --- | --- |
| accession | PXD024584 (HYE) | PXD010613 |
| sample | HeLa + yeast + *E. coli* mixtures | 4 bacterial/archaeal species |
| instrument | Q Exactive Plus, DDA top-10, HCD NCE 28, 1.6 m/z isolation | Orbitrap (LC-MS/MS), MSGF+-era runs |
| database | SwissProt human 20,416 + yeast 6,067 + *E. coli* 4,403 + cRAP 116 | per-species reference proteomes |
| split axis | mixture-disjoint (A/B/C differ in H:Y:E ratio) | species-disjoint |
| labelled spectra | 173,237 | 656,799 |
| chimeric | 29.8% (18.1% cross-species) | 34-39% per split |

Materialized at
`${DATA_ROOT}/chimericity_benchmarks/{PXD024584_HYE_v1,PXD010613_bacteria_v1}`.

### Splits

HYE (mixture-disjoint; mixture composition from the corpus README, settled
empirically by organism share of confident PSMs):

| split | mixture | rows | chimeric | cross-species | median purity |
| --- | --- | ---: | ---: | ---: | ---: |
| train | C (*E. coli*-heavy) | 79,940 | 26.5% | 17.1% | 0.610 |
| val | B (yeast-heavy) | 45,886 | 30.6% | 18.6% | 0.602 |
| test | A (human-heavy) | 47,411 | 34.7% | 19.5% | 0.593 |

Chimericity rises monotonically with human content (10.5% -> 36% -> 64.7% of
confident PSMs) while window purity falls in step -- the complexity ordering
you would predict physically, and independent evidence that the label tracks
co-isolation rather than search noise. Note the prevalence shift across splits
is real; AUROC is prevalence-insensitive, but a threshold-based metric would
not be.

Bacteria (species-disjoint):

| split | species | rows | chimeric | median purity |
| --- | --- | ---: | ---: | ---: |
| train | *C. crescentus* | 276,834 | 34.2% | 0.699 |
| val | *E. faecalis* + *A. muciniphila* | 92,603 | 39.1% | 0.650 |
| test | *H. congolense* | 287,362 | 36.5% | 0.691 |

Both are disjoint from dIon pretraining: PXD010613 is the held-out project
of the bacterial corpora, and PXD024584 appears in neither the bacterial
pretraining set nor KitchenSink v3's source list. **Re-check this before using
either with a differently-pretrained encoder.**

The bacterial corpus is single-organism per run, so `label_cross_species` is
null there; it serves as an out-of-corpus test set (different instrument,
different organisms) for models trained on HYE.

## Auxiliary column, not a target

`ms1_window_purity` is the fraction of isolation-window intensity belonging to
the precursor's isotope envelope, computed from the MS1 sidecars. It measures
*potential* co-isolation from the survey scan, which is a different quantity
from realized co-identification: measured on 26,748 spectra carrying both,
(1 - purity) predicts the search label with AUROC 0.70, and median purity is
0.603 for chimeric versus 0.782 for non-chimeric spectra. Useful for analysis;
using it as the target would be a different, easier task.

## Rebuild

```bash
# search (CPU partition; see the FragPipe recipe in the ClusterB notes)
sbatch <corpus>/search_*.sbatch
# MS1 windows for the mzML-sourced corpus
python scripts/extract_ms1_isolation_windows.py --mzml-root <mzML> --output-root <sidecars>
# materialize
python scripts/materialize_chimericity_benchmark.py --search-root <work> \
    --mzml-root <mzML> --ms1-sidecars <sidecars> --output-root <out>   # HYE
python scripts/materialize_chimericity_benchmark.py --search-root <work> \
    --lance <pooled test.lance> --output-root <out>                    # bacteria
```

Each output directory carries `train/val/test.parquet` plus a `manifest.json`
recording the protocol, the split map, per-split label prevalence, the
peptides-per-spectrum histogram, and the deviations above.

For dIon downstream training, convert the immutable Parquet output once to
the standard streaming Lance layout:

```bash
python scripts/data_prep/convert_auxiliary_benchmark_parquet_to_lance.py \
    --input-root ${DATA_ROOT}/chimericity_benchmarks/PXD024584_HYE_v1
```

Use the resulting `lance/` directory as `downstream_root_dir`. The converter
preserves every source column; task loaders must explicitly select model inputs
and must not feed label-derived audit columns such as `n_peptides` or
`ms1_window_purity` to the model.
