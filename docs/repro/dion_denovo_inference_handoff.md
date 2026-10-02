# dIon De Novo Inference Protocol

Validated for unlabeled yeast-run/proteome-mapping inference at commit
`f0ac27444c8494e094635753c7ef9b50db7a0110`.

## Selected Complete De Novo Models

Both models are precursor-conditioned, end-to-end dIon-de-novo-labeled-v1 (DNLv1) fine-tunes of
the same dIon encoder from the 300-epoch pretraining campaign. Each downstream `.ckpt` contains the complete
encoder and Casanovo-style decoder. Supply it with `--downstream_weights`, not
`--encoder_weights`. Selection used **native DNLv1 autoregressive validation
peptide precision** (`denovo_tf_val_pep_prec`, maximum), not test performance.

| Model | W&B run | Selected checkpoint | Inference peak cap |
| --- | --- | --- | ---: |
| dIon, DNLv1 fine-tune | `user/denovo-dnlv1/v5442602` | `epoch=37-denovo_tf_val_pep_prec=0.60.ckpt` | 200 |
| dIon, DNLv1 fine-tune, 1,000 peaks | `user/denovo-dnlv1/v5598800` | `epoch=27-denovo_tf_val_pep_prec=0.62.ckpt` | 1,000 |

Exact Cluster B paths:

```text
200 peaks:
${CHECKPOINT_ROOT}/denovo_dnlv1/denovo_dnlv1_encoderld_hybrid300_conditioned_2344260_2/checkpoint_06_55_38_836668__13_09_26/epoch=37-denovo_tf_val_pep_prec=0.60.ckpt

1,000 peaks:
${CHECKPOINT_ROOT}/denovo_dnlv1_maxpeaks1000/denovo_dnlv1_encoderld_hybrid300_conditioned_maxpeaks1000_2359880_0/checkpoint_01_33_53_276019__14_09_26/epoch=27-denovo_tf_val_pep_prec=0.62.ckpt
```

Transfer checksums: 200-peak full model `569fd6dcb7c9b1a636aa019d49d5d23147d3cbdb84dbe7a0974f67587fabb272` (2,663,559,902 bytes); 1,000-peak full model `863831fa76ccf11217936856c9ff71eebcbee1b5ebcee1ae38afffb6048ad188` (2,663,560,094 bytes); original dIon encoder `5ab03816737ae7f45b305e499e54c29f161a01ea00d21875e7998f138311a8c0` (2,280,948,299 bytes).

The 1,000-peak model uses the same architecture and tokenizer, but inference
requires `--max_peaks 1000 --disable_cudnn_sdp 1`. It is substantially slower
and more memory hungry. Its training used 32 GPUs (batch 50 per GPU; global
batch 1,600), while the 200-peak run used 16 GPUs (batch 100 per GPU; global
batch 1,600). Kingdoms fast inference uses 16 GPUs with batch 25 per rank; a separate MSKB prediction job runs on one GPU with a reduced batch. Inference does not require 32 GPUs. See the two training job files below for provenance.

## Original dIon Pretraining Checkpoint

Both fine-tunes above started from this **same** archived dIon encoder
checkpoint, supplied during training as `--encoder_weights`:

```text
${CHECKPOINT_ROOT}/checkpoint_12_59_01_073022__03_09_26/last.ckpt
```

This is an encoder pretraining checkpoint, not a complete de novo model. The
file exists on Cluster B. It should be shared separately only for experiments
that need the original encoder. For peptide prediction, use one of the two
complete downstream checkpoints above.

## Code And Reproducible Settings

Create a worktree from the dIon repository at the recorded branch/commit,
then inspect these tracked files:

```bash
git fetch origin run/denovo-mskb-final
git worktree add ../dIon-denovo-inference -b run/dion-yeast-inference origin/run/denovo-mskb-final
cd ../dIon-denovo-inference
git rev-parse HEAD
```

If the branch has moved, use the recorded commit above to reproduce these
specific paths/settings. Training provenance:

- `jobs/ClusterB/denovo_dnlv1_encoderld_16gpu_array.sbatch`, array
  index 2: selected 200-peak fine-tune.
- `jobs/ClusterB/denovo_dnlv1_hybrid300_maxpeaks1000_32gpu_array.sbatch`,
  array index 0: selected 1,000-peak fine-tune.
- `config_cluster_b/master_denovo_dion_hybrid_dnlv1.yaml` and
  `config_cluster_b/downstream/denovo_dnlv1_dion_hybrid.yaml`: model,
  tokenizer, and preprocessing settings.

Use `encoder_larger_deeper`, `casanovo_decoder`, the numeric-mass-delta PA11
tokenizer (`configs/tokenizers/pa11.json`), precursor conditioning `conditioned`,
default peak filtering, basepeak intensity scaling, m/z 0-2500, minimum
intensity 0.0001, maximum peptide length 100, `n_beams: 1`, `top_match: 1`,
precursor tolerance 50 ppm, and isotope error range 0-1. Set `--max_peaks`
to the selected model's cap. Run `src.main` with `--eval_only 1`,
`--validate_on_end 0`, `--test_on_end 1`, and the complete checkpoint as
`--downstream_weights`.

## Evaluation And Output Modes

For a **labeled** Lance test set, the established metrics-only example is
`jobs/ClusterB/denovo_kingdoms_v5_fast_eval_16gpu_array.sbatch` with
`config_cluster_b/downstream/denovo_kingdoms_fast_eval.yaml`. Its `test_step`
decodes autoregressively and reports peptide precision from the repository's
Casanovo-compatible matcher, even though the metric namespace says
`denovo_tf`. It writes no per-spectrum predictions. The array's indices 1 and
0 select the 200- and 1,000-peak dIon models respectively.

For a **labeled** Lance test set needing per-spectrum output, see
`jobs/ClusterB/denovo_kingdoms_v5_200peaks_prediction_16gpu_array.sbatch`
and `config_cluster_b/downstream/denovo_kingdoms_200peaks_prediction.yaml`.
Its array index 0 loads the 200-peak model. It streams a CSV with canonical
Lance row index, spectrum ID, species, truth, predicted peptide, the original
beam-search scalar confidence, and an explicit no-prediction flag. To use the
1,000-peak model, make a copy with its checkpoint and `--max_peaks 1000
--disable_cudnn_sdp 1`; test memory and throughput before the full run.

**Unlabeled yeast data require a small inference-path change.** The current
peptide collator requires `seq`, and `DeNovoTeacherForcing.test_step` computes
teacher-forced loss and truth-based metrics before beam decoding. Feeding an
unlabeled Lance table to either job above will fail; a placeholder sequence
would contaminate metrics. Add a prediction-only step that reads spectra and
precursors, calls the existing `beam_search_decode`, and writes the same
sequence/confidence/no-prediction fields without requesting or inventing a
true sequence. Retain stable source spectrum IDs for yeast-proteome mapping.

The existing distributed metrics jobs use a padded evaluation sampler. On
the 4,926,232-row capped Kingdoms set, 16 ranks repeated eight leading
spectra. For a new per-spectrum table, opt in to `exact_eval_sharding: true`
and `prediction_table: true` as in the prediction config, then verify that
every source index appears exactly once. For another dataset size and GPU
count, ensure ranks execute the same number of batches before using the
current per-batch distributed output gather; otherwise shard jobs explicitly
and merge by source index. Do not score duplicated or missing rows.

For the yeast experiment, specify proteome matching in advance, including
handling of I/L ambiguity, modifications, target/decoy or shuffled-proteome
controls, and whether the reported denominator is all spectra or emitted
calls. The matching fraction is not the same as labeled PSM peptide precision.

## Optional Labeled Peptide-Match Scoring (Main Branch)

For an **annotated** MGF with `SEQ=` labels, use the released-baseline
scorers on dIon `main`, not Casanovo's or InstaNovo's built-in
`--evaluate` output:

```bash
python scripts/evaluate_released_casanovo_mztab.py \
  --mgf labeled.mgf --mztab casanovo_predictions.mztab \
  --checkpoint casanovo_orbitrap_v5-2-0.ckpt \
  --allow-missing-predictions --output casanovo_metrics.json \
  --precision-coverage-output casanovo_precision_coverage.csv

python scripts/evaluate_released_instanovo_csv.py \
  --mgf labeled.mgf --predictions-csv instanovo_predictions.csv \
  --mass-checkpoint casanovo_orbitrap_v5-2-0.ckpt \
  --output instanovo_metrics.json \
  --precision-coverage-output instanovo_precision_coverage.csv
```

These scripts align predictions to MGF spectrum ordinals, translate
supported InstaNovo UNIMOD tokens to dIon names, and normalize both truth
and predictions with `configs/tokenizers/pa11.json`. Both use the same
`src/casanovo_eval.py` peptide/AA matcher and PA1.1 residue masses as dIon;
neither uses the baseline's own evaluation metrics. The checkpoint arguments
are optional provenance, not mass sources. For the Casanovo v5 mzTab,
`--checkpoint` also selects the ProForma prediction field rather than the
plain sequence field. Missing calls remain errors in the denominator. For
charge-restricted inputs evaluated against a larger canonical cohort, supply
each script's `--full-denominator-count` and `--supported-to-full-index`
flags.

Neither script applies to the **unlabeled yeast** proteome-mapping question:
without `SEQ=` truth, define a separate common proteome-mapping rule and
apply it to all three models' emitted calls.
