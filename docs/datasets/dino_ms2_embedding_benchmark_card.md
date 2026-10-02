# DINO MS2 Embedding Benchmarks

## Purpose

These labelled benchmarks evaluate global MS2 embeddings. They are not used as
pretraining labels. Retrieval uses repeated modified-peptide groups; pair
discrimination uses static, balanced positive and negative peptide-ion pairs.
All random selection uses seed `20260817`; the on-disk JSON manifests record
source fingerprints, counts, pair availability, and artifact hashes.

## Common Protocol

- Model input: observed precursor m/z, charge, and peak lists.
- Supported charge range: 1-10. All materialized retrieval artifacts satisfy it.
- Retrieval truth: identical `peptide_id` within each reported species.
- Pair truth: positive pairs share exact modified sequence and charge; negative
  pairs differ in peptide identity.
- Pair sets: `all_random`, `same_charge_random`, and `same_charge_10ppm`.
  The latter uses different-peptide, same-charge negatives within 10 ppm of
  precursor m/z. Each set is balanced 1:1 positive:negative per partition.
- Primary development metrics: broad retrieval mAP; hard-pair cosine ROC-AUC
  and Average Precision. Full reports retain compactness and additional
  retrieval metrics.

## Bacterial PXD010000/PXD010613

Source root: `${DATA_ROOT}/bacteria_PXD010000__PXD010613/annotated_regenerated_v3`.
The v3 Lance data retain only valid precursor charges 1-10; values above 10 are
removed rather than clamped. The regeneration manifest is stored beside the
Lance splits.

| Split | Role | Retrieval spectra | Pair spectra | Pair records |
|---|---|---:|---:|---:|
| `train.lance` | Future SSL pretraining | - | - | - |
| `val.lance` | Development validation | 51,770 | 120,054 | 269,684 |
| `test.lance` | Locked external test | 518,804 | 821,366 | 5,705,236 |

Validation comes from PXD010000. The locked test comes from the run-disjoint
PXD010613 source and excludes exact modified sequences present in both
PXD010000 train and validation; 28,994 test rows were removed by that sequence
exclusion. Benchmark artifacts live under
`${DATA_ROOT}/probing_datasets/bacterial_paper_benchmarks/`.

## NineSpecies V2

Source: `${DATA_ROOT}/9_species_V2/lance_species`.
The source field named `precursor_mass` contains observed precursor m/z and is
standardized to canonical `precursor_mz` in benchmark artifacts. Charges range
from 2 to 5.

The development benchmark at
`${DATA_ROOT}/probing_datasets/ninespecies_v2_paper_validation/`
contains 6,119 retrieval spectra, 14,948 pair spectra, and 54,076 pair records
across nine species. It excludes 98,794 normalized peptide backbones appearing
in the locked NineSpecies test, using `uppercase_amino_acids_v1`
normalization. The exclusion list is stored as
`paper_test_excluded_backbones.json`.

`ninespecies_v2_paper_test` is reserved for final reporting. The compact
end-amino-acid probe is development-only and must not be described as an
independent final test.

## Kingdoms

Source: `${DATA_ROOT}/kingdoms/processed`.
Rows require `Qvalue <= 0.01`, `chimeric == false`, a positive charge, and a
positive observed precursor m/z (historically stored in `precursor_mass`). The
materialized artifacts contain charges 1-6.

For the ten species with multiple acquisition batches, validation holds out one
whole filename-derived acquisition batch selected deterministically. The test
uses all remaining raw files and all other species. This is run-disjoint where
the source exposes multiple acquisition batches; it is not a universal
project-disjoint guarantee.

| Split | Retrieval spectra | Pair spectra | Pair records |
|---|---:|---:|---:|
| Validation | 658,738 | 1,085,786 | 4,553,362 |
| Locked test | 4,112,590 | 7,138,447 | 28,487,960 |

Artifacts live under
`${DATA_ROOT}/probing_datasets/kingdoms_paper_benchmarks/`.

## External References

GLEAMS is evaluated only as a supervised external reference. Its charge-2-5
restriction and preprocessing can remove rows; external reports must state the
retained corpus. GLEAMS results are not an upper bound and must not be compared
as a leakage-free SSL baseline without a provenance audit.

## Scope and Limitations

Peptide identities derive from search-engine annotations and are weak labels.
The hard-pair balanced-pair FDR metrics are embedding-discrimination metrics,
not search-engine false discovery rates. Future pretraining corpora must be
checked against locked test projects/runs and documented before use.
