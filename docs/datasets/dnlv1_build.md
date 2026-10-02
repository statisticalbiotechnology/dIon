# dIon-de-novo-labeled-v1 (DNLv1) Build Record

## Status

dIon-de-novo-labeled-v1 (DNLv1) has been materialized and audited at:

```text
${DATA_ROOT}/denovo_dnlv1/lance
```

This is a replacement build from final KitchenSink V4 Lance data plus the
official `mskb_final` Lance source. It is not a raw-source rebuild.

The paper name is **dIon-de-novo-labeled-v1**, abbreviated **DNLv1**.
Historical run IDs, checkpoint aliases, and immutable result records may retain
the former internal `v5` identifier; those names refer to this same dataset.

## Construction

1. Reuse every final V4 row except the legacy MSKB component, whose V4
   provenance value is `mskb_v5_seed`.
2. Add official `mskb_final` test to DNLv1 test.
3. From official `mskb_final` train, select validation peptidoforms with seed
   `42`: sort canonical peptidoforms by a stable BLAKE2b hash and accumulate
   whole-peptidoform groups to a target of 25,000 spectra.
4. Add remaining official-train peptidoforms to DNLv1 train.
5. Apply the preserved V3 cleanup: global `test > val > train` peptidoform
   precedence, followed by one representative per exact m/z peak list.

The terminology for the replacement source is **`mskb_final`**. The string
`mskb_v5_seed` appears only where needed to identify the legacy V4 component
being replaced.

## Inputs

| Input | Role |
| --- | --- |
| `${DATA_ROOT}/denovo_kitchensink_v4/lance` | Final reusable non-MSKB V4 rows |
| `${DATA_ROOT}/casanovo_official/v5_mskb_final/lance_no_peak_cap/train.lance` | Official `mskb_final` train source |
| `${DATA_ROOT}/casanovo_official/v5_mskb_final/lance_no_peak_cap/test.lance` | Official `mskb_final` test source |
| `scripts/data_prep/legacy/kitchensink_v3_reference/cleanup_kitchensink_v3_lance.py` | Preserved final cleanup policy |

## Final Audit

| Split | Spectra | Unique peptidoforms |
| --- | ---: | ---: |
| Train | 5,275,467 | 2,083,675 |
| Validation | 170,140 | 91,803 |
| Test | 345,414 | 114,526 |

The final cleanup audit reported:

- `train_val_peptidoforms = 0`
- `train_test_peptidoforms = 0`
- `val_test_peptidoforms = 0`
- duplicate physical rows = `0`
- duplicate exact-m/z peak-list rows = `0`

The official validation carve-out contains 25,002 spectra across 7,355
canonical peptidoforms. Its held-out membership removed 7,273 V4 training rows
from other source provenances, as required by the V3 split policy.

## Official-Test Qualification

The official `mskb_final` test source has 200,000 rows. The inherited exact-m/z
peak-list cleanup removed eight rows that were internal exact-m/z duplicates:
`source_row` `32154`, `32158`, `32165`, `32167`, `32179`, `32181`, `32184`, and
`94718`. Therefore DNLv1 test retains 199,992 official-test rows.

This preserves the established V3 exact-peak uniqueness invariant. DNLv1 should
not be described as containing the official test split byte-for-byte or with
zero row deletions. It does contain all non-duplicate official-test rows and
uses the official test as its test membership source.

## Reproduction

```bash
${WORK_ROOT}/conda-envs/dIon-env/bin/python -u \
  scripts/data_prep/materialize_dnlv1.py \
  --stage all \
  --output-root ${DATA_ROOT}/denovo_dnlv1 \
  --seed 42 \
  --validation-target-spectra 25000 \
  --batch-size 8192 \
  --rows-per-write 50000
```

The materialized root retains `build_summary_pre_cleanup.json`,
`cleanup_config.json`, `cleanup_summary.json`, and `lance_pre_cleanup/`.
The pre-build audit can be rerun with
`scripts/data_prep/audit_dnlv1.py`.
