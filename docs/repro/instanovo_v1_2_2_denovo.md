# InstaNovo v1.2.2 Held-Out De Novo Baseline and Audit

## Current State

This document is the single reproducibility record for the InstaNovo v1.2.2
baseline. It uses the official supervised `instanovo-v1.2.0.ckpt` checkpoint,
decoded by InstaNovo v1.2.2 with genuine greedy decoding
(`num_beams=1`, `use_knapsack=false`).

The validated MSKB-final charge-<5 result uses the raw, byte-preserving
charge-only subset of the official MGF:

```text
${DATA_ROOT}/casanovo_official/v5_mskb_final/mgf_data/mskb_final/charge1to4_raw/official_test.mgf
```

All `196,979` charge-1--4 records emitted a prediction. The full-coverage
metrics are peptide precision `0.9160468882`, amino-acid precision
`0.9608621576`, amino-acid recall `0.9609358826`, and precision--coverage AUC
`0.9936419417`. The immutable report is:

```text
${RESULTS_ROOT}/denovo_eval/instanovo_v1_2_2/mskb_final_charge_lt5_raw_mgf/metrics.json
```

## Isolated Installation and Source Provenance

No existing dIon, Casanovo, or local working environment was modified.
The baseline was installed in a dedicated conda prefix:

```bash
conda create -y -p ${WORK_ROOT}/conda-envs/instanovo_1_2_2 python=3.11
```

The official repository was cloned separately from dIon and pinned to the
v1.2.2 source revision:

```bash
git clone https://github.com/instadeepai/InstaNovo.git \
  ${WORK_ROOT}/external/instanovo_1_2_2
cd ${WORK_ROOT}/external/instanovo_1_2_2
git checkout 5a2dd39d602132486662a070354b54412f6e21c7
```

The isolated environment has InstaNovo installed from that local clone. Its
verified state is:

```text
Python:    3.11.16
InstaNovo: 1.2.2
Source:    ${WORK_ROOT}/external/instanovo_1_2_2
Commit:    5a2dd39d602132486662a070354b54412f6e21c7
PyTorch:   2.8.0+cu128
CUDA:      12.8
```

The conda transaction history records the Python-3.11 prefix creation. Pip
does not retain the original shell command, but `pip freeze` confirms that
`instanovo` is installed from the local pinned clone rather than a project
environment or a wheel from a different revision. A clean reconstruction is:

```bash
${WORK_ROOT}/conda-envs/instanovo_1_2_2/bin/python -m pip install -e \
  ${WORK_ROOT}/external/instanovo_1_2_2
```

The stock supervised checkpoint is kept separately from the package source:

```text
${CHECKPOINT_ROOT}/instanovo_v1_2_0/instanovo-v1.2.0.ckpt
```

## Inputs

| Cohort | Input | Rows | Charge policy |
| --- | --- | ---: | --- |
| MSKB-final | `${DATA_ROOT}/casanovo_official/v5_mskb_final/mgf_data/mskb_final/charge1to4_raw/official_test.mgf` | 196,979 | Canonical charge `<5` comparison cohort |
| Kingdoms | `${DATA_ROOT}/denovo_kingdoms/run_disjoint_v1/run_disjoint_v1_species_cap100k/instanovo_v1_2_2_charge1to10/test_charge1to10_for_instanovo_v1_2_2.mgf` | 4,926,232 | Full charge range, 1--6 |

The MGF fields follow the normal de novo convention required by InstaNovo:
precursor `PEPMASS`, `CHARGE`, scan-stable title/ordinal, fragment peaks, and
annotated target sequence retained outside inference for scoring.

## Exact Inference Configuration

The standard InstaNovo configuration defaults to multi-beam decoding. This
baseline explicitly overrides it to compare one-path greedy decoding with the
dIon de novo runs:

```text
model_checkpoint = instanovo-v1.2.0.ckpt
instanovo_package = 1.2.2
decoder = transformer greedy
num_beams = 1
use_knapsack = false
batch_size = 1024
num_workers = 8
fp16 = true
refinement = false
```

The production command is implemented in
[`jobs/portable/instanovo_v1_2_2_denovo_locked_test.sh`](../../jobs/portable/instanovo_v1_2_2_denovo_locked_test.sh):

```bash
instanovo transformer predict \
  --data-path "${MGF}" \
  --output-path "${PREDICTIONS_PARTIAL}" \
  --instanovo-model "${INSTANOVO_CKPT}" \
  --denovo \
  num_beams=1 \
  use_knapsack=false \
  batch_size=1024 \
  num_workers=8 \
  fp16=true
```

`log_probs` is InstaNovo's scalar sequence-level confidence score. Larger,
less-negative values rank ahead in precision--coverage curves; it is not
treated as a calibrated probability. `token_log_probs` and `delta_mass_ppm`
remain in the raw CSV.

Output is persistent: prediction CSVs are written first as
`predictions.csv.partial` under
`${RESULTS_ROOT}/denovo_eval/instanovo_v1_2_2/`, validated for complete
unique ordinals, then atomically renamed to `predictions.csv`. Node-local
`/tmp` is used only for InstaNovo worker temporaries.

Measured A100 80GB end-to-end rates with this configuration were 230.5
spectra/s for MSKB and 202.1 spectra/s for Kingdoms. The corresponding Slurm
jobs request one and nine hours:

```bash
ALLOW_LOCKED_TEST=1 sbatch jobs/ClusterA/instanovo_v1_2_2_mskb_final_locked_test.sbatch
ALLOW_LOCKED_TEST=1 sbatch jobs/ClusterA/instanovo_v1_2_2_kingdoms_locked_test.sbatch
```

## Canonical Evaluation Protocol

The evaluator is
[`scripts/evaluate_released_instanovo_csv.py`](../../scripts/evaluate_released_instanovo_csv.py).
It deliberately converts InstaNovo ProForma/UNIMOD output to the same symbolic
tokens and canonical residue masses used for dIon and Casanovo results:

- residue-level mass tolerance: `0.1 Da`;
- cumulative prefix/suffix mass tolerance: `0.5 Da`;
- `I` and `L` are mass-equivalent;
- `M[UNIMOD:35]`, `C[UNIMOD:4]`, deamidation, phosphorylation, acetylation,
  carbamylation, and ammonia loss map to the common canonical token masses.

In particular, the scorer uses the canonical phospho delta (`79.966331 Da`),
not InstaNovo's checkpoint-vocabulary rounded delta. This prevents a
model-specific vocabulary convention from changing the cross-model matching
rule.

The completed MSKB output contains exactly `196,979 / 196,979` rows, with
`spectrum_id=test:<ordinal>` and `prediction_id=<ordinal>` matching the MGF
record ordinal. There are no missing or duplicate predictions.

## Scoring Audit

The validated output contains exactly `196,979 / 196,979` rows, with
`spectrum_id=test:<ordinal>` and `prediction_id=<ordinal>` matching the raw-MGF
record ordinal. There are no missing or duplicate predictions. The scorer
converts InstaNovo ProForma/UNIMOD output to the same canonical symbolic tokens
and residue masses used for dIon and Casanovo evaluation.

## Interpretation Constraint

This is the valid greedy-decoding result for the released checkpoint and the
fixed MSKB-final charge-<5 cohort. Audit training-data overlap between the
released InstaNovo training material and this held-out cohort before comparing
results, as recorded in the experiment log.
