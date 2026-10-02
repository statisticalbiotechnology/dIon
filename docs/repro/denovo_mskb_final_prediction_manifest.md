# MSKB-Final Charge-Limited Prediction Manifest

This is the authoritative list of downstream de novo models eligible for
prediction on the MSKB held-out set as of 2026-09-14.

## Prediction Dataset

Use this test split, rather than the original canonical MSKB-final test split:

```text
${DATA_ROOT}/denovo_mskb_final/lance_peptidoform_val10k_seed42/test_charge_lt5_for_casanovo_v_gt_5_0/
```

It is the canonical held-out MSKB-final test set with approximately 1% of
examples removed because their precursor charge exceeds four. This is required
for comparison with Casanovo >= 5.2.1.

## Eligibility Rule

A model is included only when it has a retained downstream checkpoint selected
by `denovo_tf_val_pep_prec`, and its native de novo training phase completed
the intended 80 epochs. `last.ckpt` is not used for prediction selection.

For DNLv1, a later failure in the optional MSKB transfer-evaluation stage does
not invalidate a checkpoint when native DNLv1 training completed and the selected
checkpoint exists. That qualification is recorded below.

## MSKB-Final Models

### Selected Models

| Model | W&B run | Checkpoint ID | Notes |
| --- | --- | --- | --- |
| Hybrid-300, 1000 peaks | `ig70dslj` | `mskb-h300-p1000` | Precursor-conditioned finetune; validated prediction-mode peptide precision `0.821727`, precision-coverage AUC `0.969301`. |
| Hybrid-300 | `u9m4zqba` | `mskb-h300-cond` | Precursor-conditioned; complete rerun. |
| Scratch encoder-LD | `mn4nwa8l` | `mskb-scratch` | Precursor-conditioned; complete rerun. |

### Frozen Controls

| Model | W&B run | Checkpoint ID | Notes |
| --- | --- | --- | --- |
| Hybrid-300 frozen encoder | `085qmbs5` | `mskb-h300-frozen` | Precursor-conditioned frozen control. |
| Local-60 Gram refined, frozen encoder | `woz4tzcz` | `mskb-l60-gram-frozen` | Precursor-conditioned frozen control. |

### Additional 1000-Peak Baseline

| Model | W&B run | Checkpoint ID | Status |
| --- | --- | --- | --- |
| Scratch encoder-LD, 1000 peaks | `7g17kss8` | `mskb-scratch-p1000` | Precursor-conditioned; completed all 80 epochs. Best native validation peptide precision `0.763678` at epoch 79 / step 99,439. Awaiting final held-out prediction. |

### Architecture Baselines

| Model | W&B run | Available file | Status |
| --- | --- | --- | --- |
| Scratch pairwise encoder | `5yg1ilay` | `mskb-pairwise-old-last` | `last.ckpt` exists at epoch 28, but the run used old loss-based checkpoint selection. Checkpoint selection does not satisfy the primary protocol. |
| Scratch base architecture | `t7skd719` | `mskb-base-arch-old-last` | `last.ckpt` exists at epoch 38, but the run used old loss-based checkpoint selection. Checkpoint selection does not satisfy the primary protocol. |

## DNLv1-native Models Eligible for MSKB Prediction

The models below are the selected de facto converged DNLv1 candidates for the
charge-limited MSKB prediction sweep. Partial-run details are retained to document checkpoint provenance.

### Selected Models

| Model | W&B run | Checkpoint ID | Native DNLv1 progress | Notes |
| --- | --- | --- | --- | --- |
| Hybrid-300, 1000 peaks | `v5598800` | `v5-h300-p1000` | Epoch 27, step 92,315 | De facto converged by inspection; selected epoch 27. Validated prediction-mode peptide precision `0.855579`, precision-coverage AUC `0.979456`. |
| Hybrid-300 | `v5442602` | `v5-h300` | Epoch 66, step 220,898 | De facto converged by inspection; selected epoch 37. |
| Scratch encoder-LD | `v5462670` | `v5-scratch` | Epoch 67, step 223,849 | De facto converged by inspection; selected epoch 41. |

### Frozen Controls

| Model | W&B run | Checkpoint ID | Native DNLv1 progress | Notes |
| --- | --- | --- | --- | --- |
| Local-60 Gram refined, frozen encoder | `v5603300` | `v5-l60-gram-frozen` | Epoch 79, step 263,759 | Native DNLv1 rises smoothly from 0.212 to 0.467. Best and `last.ckpt` are the same epoch-79 model; no epoch-0 checkpoint remains. |
| Hybrid-300 frozen encoder | `v5445180` | `v5-h300-frozen-best` | Epoch 79, step 263,759 | Native DNLv1 rises to a selected epoch-39 score of 0.35. |
| Hybrid-300 frozen encoder | `v5445180` | `v5-h300-frozen-late` | Epoch 79, step 263,759 | Retained late `last.ckpt` is epoch 43; evaluate alongside selected epoch 39. No epoch-0 checkpoint remains. |

### Additional 1000-Peak Baseline

| Model | W&B run | Checkpoint ID | Native DNLv1 progress | Status |
| --- | --- | --- | --- | --- |
| Scratch encoder-LD, 1000 peaks | `v5650440` | `v5-scratch-p1000` | Last epoch 27, step 92,315 | Container filesystem failure ended training after checkpoints were saved. Best native validation peptide precision `0.591252` is epoch 26; retain as a de facto converged baseline candidate, pending final held-out prediction. |

### Additional Model

| Model | W&B run | Checkpoint ID | Native DNLv1 progress | Notes |
| --- | --- | --- | --- | --- |
| Local-60 Gram refined | `v5564640` | `v5-l60-gram` | Epoch 68, step 224,399 | De facto converged by inspection; selected epoch 41. Timed out after the checkpoint was written. |

### Architecture Baseline

| Model | W&B run | Checkpoint ID | Status |
| --- | --- | --- | --- |
| Scratch pairwise encoder | `v5462981` | `v5-pairwise` | Native DNLv1 training and test completed; selected epoch 66. Retained as an architecture baseline. |

The apparent high initial validation points in the two frozen W&B charts are
not native DNLv1 training measurements. Each job reused its W&B run ID for the
subsequent MSKB `eval_only` stage, which reset `trainer/global_step` to zero
and logged the same unprefixed metric names. The points are MSKB validation:
`0.70337` for Hybrid-300 frozen and `0.83107` for Gram-frozen.

## Exact Checkpoint Paths

```text
mskb-scratch = ${CHECKPOINT_ROOT}/denovo_mskb_final/denovo_mskb_final_encoderld_scratch_conditioned_2344805_0/checkpoint_01_47_09_792668__13_09_26/epoch=78-denovo_tf_val_pep_prec=0.74.ckpt
mskb-h300-cond = ${CHECKPOINT_ROOT}/denovo_mskb_final/denovo_mskb_final_encoderld_hybrid300_conditioned_2344884_2/checkpoint_02_03_48_526498__13_09_26/epoch=72-denovo_tf_val_pep_prec=0.80.ckpt
mskb-h300-frozen = ${CHECKPOINT_ROOT}/denovo_mskb_final_frozen/denovo_mskb_final_encoderld_hybrid300_conditioned_frozen_2344517_0/checkpoint_06_53_54_873542__13_09_26/epoch=74-denovo_tf_val_pep_prec=0.51.ckpt
mskb-l60-gram-frozen = ${CHECKPOINT_ROOT}/denovo_mskb_final_gram_refined_frozen/denovo_mskb_final_encoderld_hybrid60_gram_refined_conditioned_frozen_2360329_0/checkpoint_17_54_00_299663__13_09_26/epoch=76-denovo_tf_val_pep_prec=0.67.ckpt
mskb-h300-p1000 = ${CHECKPOINT_ROOT}/denovo_mskb_final_maxpeaks1000/denovo_mskb_final_encoderld_hybrid300_conditioned_maxpeaks1000_2359879_0/checkpoint_20_04_28_378965__13_09_26/epoch=77-denovo_tf_val_pep_prec=0.82.ckpt
mskb-scratch-p1000 = ${CHECKPOINT_ROOT}/denovo_mskb_final_maxpeaks1000/denovo_mskb_final_encoderld_scratch_encoderld_conditioned_maxpeaks1000_2465043_0/checkpoint_03_37_19_137171__15_09_26/epoch=79-denovo_tf_val_pep_prec=0.76.ckpt
v5-h300-p1000 = ${CHECKPOINT_ROOT}/denovo_dnlv1_maxpeaks1000/denovo_dnlv1_encoderld_hybrid300_conditioned_maxpeaks1000_2359880_0/checkpoint_01_33_53_276019__14_09_26/epoch=27-denovo_tf_val_pep_prec=0.62.ckpt
v5-scratch-p1000 = ${CHECKPOINT_ROOT}/denovo_dnlv1_maxpeaks1000/denovo_dnlv1_encoderld_scratch_encoderld_conditioned_maxpeaks1000_2465044_0/checkpoint_03_48_58_284008__15_09_26/epoch=26-denovo_tf_val_pep_prec=0.59.ckpt
v5-h300 = ${CHECKPOINT_ROOT}/denovo_dnlv1/denovo_dnlv1_encoderld_hybrid300_conditioned_2344260_2/checkpoint_06_55_38_836668__13_09_26/epoch=37-denovo_tf_val_pep_prec=0.60.ckpt
v5-scratch = ${CHECKPOINT_ROOT}/denovo_dnlv1/denovo_dnlv1_encoderld_scratch_encoderld_conditioned_2346267_0/checkpoint_04_27_32_377495__13_09_26/epoch=41-denovo_tf_val_pep_prec=0.57.ckpt
v5-l60-gram = ${CHECKPOINT_ROOT}/denovo_dnlv1_gram_refined/denovo_dnlv1_encoderld_hybrid60_gram_refined_conditioned_2356464_0/checkpoint_15_18_10_789472__13_09_26/epoch=41-denovo_tf_val_pep_prec=0.60.ckpt
v5-h300-frozen-best = ${CHECKPOINT_ROOT}/denovo_dnlv1_frozen/denovo_dnlv1_encoderld_hybrid300_conditioned_frozen_2344518_0/checkpoint_06_56_55_277799__13_09_26/epoch=39-denovo_tf_val_pep_prec=0.35.ckpt
v5-h300-frozen-late = ${CHECKPOINT_ROOT}/denovo_dnlv1_frozen/denovo_dnlv1_encoderld_hybrid300_conditioned_frozen_2344518_0/checkpoint_06_56_55_277799__13_09_26/last.ckpt
v5-pairwise = ${CHECKPOINT_ROOT}/denovo_dnlv1/denovo_dnlv1_encoderld_scratch_encoder_pairwise_conditioned_2346298_1/checkpoint_04_35_36_562967__13_09_26/epoch=66-denovo_tf_val_pep_prec=0.57.ckpt
v5-l60-gram-frozen = ${CHECKPOINT_ROOT}/denovo_dnlv1_gram_refined_frozen/denovo_dnlv1_encoderld_hybrid60_gram_refined_conditioned_frozen_2360330_0/checkpoint_17_55_52_479329__13_09_26/epoch=79-denovo_tf_val_pep_prec=0.47.ckpt
mskb-pairwise-old-last = ${CHECKPOINT_ROOT}/denovo_mskb_final/denovo_mskb_final_encoderld_scratch_encoder_pairwise_conditioned_2332261_1/checkpoint_17_21_50_325489__12_09_26/last.ckpt
mskb-base-arch-old-last = ${CHECKPOINT_ROOT}/denovo_mskb_final/denovo_mskb_final_encoderld_scratch_encoder_base_arch_conditioned_2332626_0/checkpoint_17_21_34_668065__12_09_26/last.ckpt
```

## Excluded Models

Do not use the following under this prediction protocol:

- The first MSKB array (`2318060` / `2318076`): old loss-monitored checkpoint behavior left stale early `last.ckpt` files and no peptide-precision-selected checkpoint pointer.
- All other complete MSKB models, including null-conditioned and conditioned variants not selected by this protocol: intentionally out of scope for this final prediction sweep. The two scratch-architecture runs remain listed above as architecture baselines.
- Failed 1000-peak MSKB run `2356466` and first failed DNLv1 1000-peak run `2356467`. The later DNLv1 1000-peak checkpoint `v5-h300-p1000` is retained as de facto converged by explicit decision above.
