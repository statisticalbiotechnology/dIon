# dIon

dIon is a foundation encoder for tandem mass spectra and an associated de novo
peptide-sequencing model. This repository contains the model implementation,
training and evaluation code, dataset construction utilities, and the exact
configuration family used in the accompanying publication.

## Checkpoints

Release `v0.1.0` is hosted at [alfred-n/dIon](https://huggingface.co/alfred-n/dIon):

| Checkpoint | Contents | CLI argument |
|---|---|---|
| `dion-v0.1-foundation.ckpt` | Pretrained encoder, EMA teacher, and pretraining state | `--encoder_weights` |
| `dion-v0.1-denovo-200peaks.ckpt` | Fine-tuned encoder and Casanovo-style decoder; standard 200-peak model | `--downstream_weights` |
| `dion-v0.1-denovo-1000peaks.ckpt` | Fine-tuned encoder and decoder; slower 1,000-peak model | `--downstream_weights` |

Download the release with the Hugging Face CLI and set the paths used below:

```bash
hf download alfred-n/dIon --revision v0.1.0 --local-dir dion-v0.1

export DION_ENCODER_CKPT="$PWD/dion-v0.1/checkpoints/dion-v0.1-foundation.ckpt"
export DION_DENOVO_CKPT="$PWD/dion-v0.1/checkpoints/dion-v0.1-denovo-200peaks.ckpt"
export DION_DENOVO_1000_CKPT="$PWD/dion-v0.1/checkpoints/dion-v0.1-denovo-1000peaks.ckpt"
```

Do not pass a complete de novo checkpoint as `--encoder_weights`: downstream
checkpoints include both the encoder and decoder and must be loaded with
`--downstream_weights`.

## Installation

Clone the repository first:

```bash
git clone <repository-url> dIon
cd dIon
```

### Conda

This is the standard installation used on conventional CUDA clusters:

```bash
mamba env create -f environment.yml
conda activate dIon-env
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`conda env create -f environment.yml` can be used when Mamba is unavailable.

### Apptainer on Slingshot clusters

The Slingshot image extends NVIDIA's PyTorch container with the host's
libfabric/CXI libraries and the AWS OFI NCCL plugin. Build it on a Slingshot
login or build node where the files referenced by
`apptainer/pytorch_slingshot.def` are available:

```bash
apptainer build pytorch-ngc-26.06-slingshot.sif \
  apptainer/pytorch_slingshot.def
apptainer build dion-ngc-26.06-slingshot.sif \
  apptainer/dion_slingshot.def
```

Run with GPU support and bind the repository, data, and checkpoint locations:

```bash
export DATA_ROOT=/path/to/data
export CHECKPOINT_ROOT=/path/to/checkpoints

apptainer exec --nv \
  --bind "$PWD:$PWD" \
  --bind "$DATA_ROOT:$DATA_ROOT" \
  --bind "$CHECKPOINT_ROOT:$CHECKPOINT_ROOT" \
  dion-ngc-26.06-slingshot.sif \
  python -m src.main --help
```

Multi-node Slingshot jobs additionally require the site-specific OFI/NCCL
environment shown in the templates under `jobs/ClusterB/` and `bash_scripts/`.

## Use the Foundation Encoder

The encoder architecture and pretraining wrapper must match the checkpoint.
For downstream fine-tuning, provide the foundation checkpoint and a task
configuration:

```bash
python -m src.main \
  --config configs/master_denovo_dion_hybrid_dnlv1.yaml \
  --pretrain_config configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml \
  --pretraining_task dion \
  --encoder_weights "$DION_ENCODER_CKPT" \
  --downstream_config configs/downstream/denovo_dnlv1_dion_hybrid.yaml \
  --downstream_task denovo_tf \
  --data_root_dir "$DATA_ROOT/pretraining" \
  --downstream_root_dir "$DATA_ROOT/denovo" \
  --output_dir "$CHECKPOINT_ROOT/finetuning" \
  --log_dir "$CHECKPOINT_ROOT/logs" \
  --log_wandb 0
```

For frozen embedding evaluation, use the standalone evaluators with the same
`--config`, `--pretrain_config`, `--pretraining_task dion`, and
`--encoder_weights` arguments. The complete evaluation protocol is documented
in [docs/repro/embedding_evaluation_protocol.md](docs/repro/embedding_evaluation_protocol.md).

## Use the De Novo Model

Evaluate a complete fine-tuned checkpoint on a compatible labeled Lance test
set with:

```bash
python -m src.main \
  --config configs/master_denovo_dion_hybrid_dnlv1.yaml \
  --pretrain_config configs/pretrain/dion_hybrid_distractor_null_local_100pct.yaml \
  --pretraining_task dion \
  --downstream_config configs/downstream/denovo_ninespecies_v2_eval.yaml \
  --downstream_task denovo_tf \
  --downstream_weights "$DION_DENOVO_CKPT" \
  --downstream_root_dir "$DATA_ROOT/ninespecies_v2" \
  --eval_only 1 \
  --validate_on_end 0 \
  --test_on_end 1 \
  --save_top_k 0 \
  --save_last 0 \
  --log_wandb 0
```

The released checkpoint's peak cap, precursor-conditioning mode, encoder
architecture, decoder architecture, and tokenizer must be retained. To adapt
either released checkpoint to new labeled spectra, see
[docs/finetune_denovo.md](docs/finetune_denovo.md). Exact
checkpoint provenance and evaluation constraints are documented in
[docs/repro/dion_denovo_inference_handoff.md](docs/repro/dion_denovo_inference_handoff.md).

## Reproduce Pretraining

The canonical 300-epoch pretraining recipe is intentionally kept out of this
quick-start README. See
[docs/repro/dion_300epoch_pretraining.md](docs/repro/dion_300epoch_pretraining.md)
for the objective, configuration paths, distributed launch, checkpointing, and
resume procedure.

## Data and Evaluation

Set portable roots before using dataset and evaluation scripts:

```bash
export DATA_ROOT=/path/to/data
export RESULTS_ROOT=/path/to/generated-results
export CHECKPOINT_ROOT=/path/to/checkpoints
export WORK_ROOT=/path/to/external-software
```

Dataset construction and benchmark definitions are under `docs/datasets/`.
Reproduction and external-baseline protocols are under `docs/repro/`. Generated
results are not included in this repository; reported values are provided in
the accompanying publication.

## Repository Layout

| Path | Contents |
|---|---|
| `src/` | Models, wrappers, data loaders, callbacks, and training entrypoint |
| `configs/` | Portable model, task, tokenizer, and evaluation configurations |
| `config_cluster_b/` | Large-scale training configurations used on the Slingshot cluster |
| `scripts/` | Dataset preparation, evaluation, and export utilities |
| `jobs/` | Cluster and portable job templates |
| `docs/datasets/` | Dataset cards and construction procedures |
| `docs/repro/` | Training, evaluation, and baseline reproduction protocols |

CLI arguments override configuration values. Configuration values override
parser defaults. Run `python -m src.main --help` for the complete CLI.
