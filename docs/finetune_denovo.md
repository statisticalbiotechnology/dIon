# Fine-Tune dIon for De Novo Sequencing

This guide adapts dIon to a new labeled de novo dataset. Two initialization
modes are supported:

1. Start from the dIon foundation encoder and initialize a new decoder.
2. Start from a released dIon de novo checkpoint and continue adapting its
   encoder and decoder with a fresh optimizer.

Download release `v0.1.0` from [alfred-n/dIon](https://huggingface.co/alfred-n/dIon) before using these commands:

```bash
hf download alfred-n/dIon --revision v0.1.0 --local-dir dion-v0.1
```

## Prepare the Dataset

Create three Lance datasets under one directory:

```text
/path/to/new_dataset/
  train.lance
  val.lance
  test.lance
```

Each row must contain:

| Column | Type | Meaning |
|---|---|---|
| `mz_array` | list of floats | Fragment m/z values |
| `intensity_array` | list of floats | Fragment intensities aligned with `mz_array` |
| `precursor_mz` | float | Precursor m/z |
| `precursor_charge` | integer | Precursor charge |
| `seq` | string | Peptide sequence in the configured tokenizer notation |

The default template uses the PA1.1 numeric mass-delta tokenizer at
`configs/tokenizers/pa11.json`. Convert unsupported modification notation
before training or create an explicit tokenizer manifest and update
`tokenizer.manifest`. Do not silently map unsupported residues or
modifications.

Construct train, validation, and test splits by peptide or peptidoform identity
as appropriate for the scientific question. Do not select checkpoints on the
test split.

The portable templates are:

```text
configs/master_denovo_dion_finetune.yaml
configs/downstream/denovo_custom_lance.yaml
```

Copy the downstream template for a new experiment and record any changed peak
cap, tokenizer, precursor conditioning, optimizer, or decoding settings.

## Option A: Foundation Encoder and New Decoder

This mode loads the pretrained encoder with `--encoder_weights` and initializes
a new Casanovo-style decoder. By default, both components are trainable.

```bash
export DATA_ROOT=/path/to/new_dataset
export DION_ENCODER_CKPT="$PWD/dion-v0.1/checkpoints/dion-v0.1-foundation.ckpt"
export OUTPUT_ROOT=/path/to/output/foundation_finetune

python -m src.main \
  --config configs/master_denovo_dion_finetune.yaml \
  --encoder_weights "$DION_ENCODER_CKPT" \
  --downstream_root_dir "$DATA_ROOT" \
  --output_dir "$OUTPUT_ROOT/checkpoints" \
  --log_dir "$OUTPUT_ROOT/logs" \
  --log_wandb 0
```

To train only the new decoder, add `--freeze_encoder 1`. A frozen-encoder run
answers a different question and should be labeled separately from end-to-end
fine-tuning.

## Option B: Adapt the Released De Novo Model

This mode loads the complete encoder-decoder state with
`--downstream_weights`. Keep `--resume 0`: the checkpoint supplies model
weights, while the new dataset starts with a fresh optimizer, scheduler, epoch,
and global-step clock.

```bash
export DATA_ROOT=/path/to/new_dataset
export DION_DENOVO_CKPT="$PWD/dion-v0.1/checkpoints/dion-v0.1-denovo-200peaks.ckpt"
export OUTPUT_ROOT=/path/to/output/denovo_adaptation

python -m src.main \
  --config configs/master_denovo_dion_finetune.yaml \
  --downstream_weights "$DION_DENOVO_CKPT" \
  --resume 0 \
  --downstream_root_dir "$DATA_ROOT" \
  --output_dir "$OUTPUT_ROOT/checkpoints" \
  --log_dir "$OUTPUT_ROOT/logs" \
  --log_wandb 0
```

Do not also pass `--encoder_weights` in this mode. The downstream checkpoint
already contains the encoder and decoder. Use `--resume 1` only to continue an
interrupted run on the same training recipe when optimizer and scheduler state
must be restored.

## Hyperparameters to Reconsider

The template values are conservative starting points, not universal settings.
Set these from the size and composition of the new dataset:

- `denovo_tf.batch_size`
- `denovo_tf.learning_rate`
- `denovo_tf.warmup_steps`
- `denovo_tf.cosine_period_steps`
- `denovo_tf.epochs`
- `denovo_tf.decoder_dropout`
- `max_peaks` and `top_peaks`
- `pep_length`, charge range, precursor tolerance, and isotope-error range

When changing per-rank batch size or world size, record the effective global
batch. This code does not automatically rescale learning rate in the supplied
fine-tuning template.

For the released 200-peak model, retain `max_peaks: 200`, `top_peaks: 200`,
base-peak intensity scaling, precursor conditioning, and the PA1.1 tokenizer
unless the change is deliberate and reported. The released 1,000-peak checkpoint must be loaded with `max_peaks: 1000`,
`top_peaks: 1000`, and `--disable_cudnn_sdp 1`; it is slower and requires more
GPU memory than the standard 200-peak model.

## Multi-GPU Training

Use one process per GPU. For example, four GPUs on one node:

```bash
srun python -m src.main \
  --config configs/master_denovo_dion_finetune.yaml \
  --encoder_weights "$DION_ENCODER_CKPT" \
  --downstream_root_dir "$DATA_ROOT" \
  --output_dir "$OUTPUT_ROOT/checkpoints" \
  --log_dir "$OUTPUT_ROOT/logs" \
  --accelerator gpu \
  --strategy ddp \
  --num_devices 4 \
  --num_nodes 1 \
  --log_wandb 0
```

Use the Apptainer and Slingshot environment described in the README when
running on a Slingshot cluster.

## Validate Before a Full Run

Run one real training and validation batch with the intended model and tensor
sizes:

```bash
python -m src.main \
  --config configs/master_denovo_dion_finetune.yaml \
  --encoder_weights "$DION_ENCODER_CKPT" \
  --downstream_root_dir "$DATA_ROOT" \
  --output_dir "$OUTPUT_ROOT/smoke_checkpoints" \
  --log_dir "$OUTPUT_ROOT/smoke_logs" \
  --epochs 1 \
  --limit_train_batches 1 \
  --limit_val_batches 1 \
  --save_top_k 0 \
  --save_last 0 \
  --log_wandb 0
```

Verify that the checkpoint loads strictly, all sequence tokens are supported,
losses are finite, and the intended batch fits GPU memory. Then launch the full
run without the smoke-test limits.

The default checkpoint monitor is validation peptide precision
(`denovo_tf_val_pep_prec`, maximum). Final test evaluation should use the
selected validation checkpoint and must not influence checkpoint selection.
