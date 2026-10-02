# Canonical 300-Epoch dIon Pretraining

This protocol reproduces the canonical dIon foundation encoder. It uses the
dual objective with clean conditioned teacher views, distractor-mixed
conditioned student global views, and clean null-precursor student local views.
iBOT and KoLeo are disabled.

## Canonical Configuration

The source-of-truth files are:

```text
config_cluster_b/master_dion_hybrid_distractor_null_local_300epochs_part1.yaml
config_cluster_b/pretrain/dion_hybrid_distractor_null_local_100pct_300epochs_part1.yaml
config_cluster_b/master_dion_hybrid_distractor_null_local_300epochs.yaml
bash_scripts/dion_cluster_b_hybrid_300epochs_part1_64gpu.sbatch
bash_scripts/dion_cluster_b_hybrid_300epochs_part2_64gpu.sbatch
```

The master configurations contain historical cluster paths. Override all data,
checkpoint, log, W&B, and topology values for the target system rather than
using those paths unchanged.

## Training Contract

| Setting | Value |
|---|---:|
| Encoder | `encoder_larger_deeper` |
| Epoch schedule | 300 |
| Peak cap | 200 |
| Per-rank batch | 128 |
| Reference batch | 256 |
| Canonical world size | 64 GPUs |
| Effective global batch | 8,192 spectra |
| Precision | `bf16-mixed` |
| Global crops | 2 at 95% retained peaks |
| Local crops | 2 at 60% retained peaks |
| Student global views | distractor mixed, real precursor retained |
| Student local views | clean, null precursor |
| DINO group weights | 0.5 conditioned global, 0.5 null local |
| DINO / iBOT / KoLeo weights | 1.0 / 0.0 / 0.0 |
| Optimizer LR | `blr=8e-5`, no automatic batch scaling |
| LR schedule | 40,000-step warmup; decay duration 450,000 steps |
| Teacher momentum | 0.998 to 1.0 |

Consult the YAML itself for every remaining model and optimizer field. Do not
silently rescale the learning rate when changing world size; preserve the
effective global batch or treat the change as a new training recipe.

## Data

The training corpus is the annotated bacterial PXD010000/PXD010613 Lance
dataset described by the repository's dataset documentation. Point
`--data_root_dir` at its local materialization. The pretraining config expects
`dataset_name: bacteria`.

## Phase 1

The canonical run used 16 Slingshot nodes with four GH200 GPUs per node. The
following is the training command inside the Conda environment or container;
the surrounding `srun` and OFI/NCCL settings are site-specific.

```bash
python -m src.main \
  --config config_cluster_b/master_dion_hybrid_distractor_null_local_300epochs_part1.yaml \
  --data_root_dir "$DATA_ROOT/bacteria_PXD010000__PXD010613/annotated_regenerated_v3" \
  --output_dir "$CHECKPOINT_ROOT/dion_300epoch" \
  --log_dir "$RESULTS_ROOT/dion_300epoch/logs" \
  --embedding_dir "$RESULTS_ROOT/dion_300epoch/embedding_cache" \
  --accelerator gpu \
  --strategy ddp \
  --num_devices 4 \
  --num_nodes 16 \
  --batch_size 128 \
  --max_peaks 200 \
  --num_workers 2 \
  --limit_train_batches 1.0 \
  --limit_val_batches 100 \
  --log_wandb 0
```

The phase-one pretraining configuration has `stop_after_epoch: 250`, allowing
the run to stop cleanly while retaining a nominal 300-epoch schedule.

## Resume to 300 Epochs

Resume from the phase-one segment checkpoint with full trainer state. Do not
load it as weights only: optimizer, scheduler, EMA-teacher, epoch, and global
step state must continue together.

```bash
export RESUME_CKPT=/path/to/phase1/segment_end.ckpt

python -m src.main \
  --config config_cluster_b/master_dion_hybrid_distractor_null_local_300epochs.yaml \
  --encoder_weights "$RESUME_CKPT" \
  --resume 1 \
  --data_root_dir "$DATA_ROOT/bacteria_PXD010000__PXD010613/annotated_regenerated_v3" \
  --output_dir "$CHECKPOINT_ROOT/dion_300epoch" \
  --log_dir "$RESULTS_ROOT/dion_300epoch/logs" \
  --embedding_dir "$RESULTS_ROOT/dion_300epoch/embedding_cache" \
  --accelerator gpu \
  --strategy ddp \
  --num_devices 4 \
  --num_nodes 16 \
  --batch_size 128 \
  --max_peaks 200 \
  --num_workers 2 \
  --limit_train_batches 1.0 \
  --limit_val_batches 100 \
  --log_wandb 0
```

For Slingshot, wrap these commands with `srun`, `apptainer exec --nv`, and the
site OFI/NCCL environment from the tracked 64-GPU job templates. For another
scheduler, preserve one process per GPU and the 64-rank effective batch.

## Smoke Test

Before allocating the full job, run the same model, peak cap, and per-rank
batch on one GPU with one training and one validation batch:

```bash
python -m src.main \
  --config config_cluster_b/master_dion_hybrid_distractor_null_local_300epochs_part1.yaml \
  --data_root_dir "$DATA_ROOT/bacteria_PXD010000__PXD010613/annotated_regenerated_v3" \
  --output_dir "$CHECKPOINT_ROOT/dion_smoke" \
  --log_dir "$RESULTS_ROOT/dion_smoke/logs" \
  --accelerator gpu \
  --strategy auto \
  --num_devices 1 \
  --num_nodes 1 \
  --batch_size 128 \
  --max_peaks 200 \
  --num_workers 2 \
  --epochs 1 \
  --limit_train_batches 1 \
  --limit_val_batches 1 \
  --save_top_k 0 \
  --save_last 0 \
  --log_wandb 0
```

Use `--epochs 1` for this bounded smoke. In this training path,
`--limit_train_batches 1.0` means the complete epoch, not one batch.
