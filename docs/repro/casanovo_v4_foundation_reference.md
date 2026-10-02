# Casanovo v4 Foundation Reference

This is the released Casanovo Foundation reference. It uses the
released Casanovo v4.0.0 MassIVE-KB checkpoint and does not modify dIon's
training environment or model code.

The baseline is a supervised de novo pretrained encoder, not a label-free SSL
baseline. Keep that distinction in results metadata and comparison tables.

## Exact representation

The foundation-model paper uses only the Casanovo encoder from v4.0.0:

- nine Transformer encoder blocks;
- 512-dimensional peak representations;
- eight attention heads;
- spectrum embedding: arithmetic mean of the non-padding **peak** embeddings.

The encoder prepends its own learned spectrum token. The default runner mode,
`--pooling peak_mean`, excludes that token from the mean, matching the paper's
"individual peak embeddings" description. The runner also exposes
`--pooling all_tokens_mean` as a recorded ablation; it averages the contextualized
spectrum token and valid peak tokens together.

## Isolated setup

Do not install Casanovo into `dIon-env`. Define portable paths before use:

```bash
export PROJECT_ROOT=/path/to/dIon
export CASANOVO_V4_ROOT=/path/to/casanovo_foundation_v4
export CASANOVO_V4_ENV=/path/to/conda/envs/casanovo40
export CASANOVO_V4_PY="$CASANOVO_V4_ENV/bin/python"
export CASANOVO_V4_CKPT=/path/to/casanovo_massivekb_v4.0.0.ckpt
```

The clean clone must be tag `v4.0.0`, commit
`3c2d3f5ad91144528ff4ff4da63081391638973f`. Install it editable into the
dedicated environment after dependencies are installed:

```bash
"$CASANOVO_V4_PY" -m pip install -e "$CASANOVO_V4_ROOT"
"$CASANOVO_V4_PY" - <<'PY'
import os
from casanovo.denovo.model import Spec2Pep
model = Spec2Pep.load_from_checkpoint(os.environ["CASANOVO_V4_CKPT"], map_location="cpu")
print(model.hparams.dim_model, model.hparams.n_layers, model.hparams.n_head)
PY
```

`environments/casanovo40.yml` records the dependency versions. The environment
requires `setuptools<81` because Lightning 2.1 imports `pkg_resources`.

Download the `casanovo_massivekb.ckpt` asset from the
[v4.0.0 release](https://github.com/Noble-Lab/casanovo/releases/tag/v4.0.0).
Record its SHA-256 in every evaluation manifest.

## Casanovo preprocessing

`scripts/embed_casanovo_v4_benchmark.py` reproduces v4's dataset processing:

1. retain m/z 50--2500;
2. remove peaks within 2 Da of precursor m/z;
3. retain peaks at least 1% of base-peak intensity and the 150 most intense;
4. root-scale intensity (exponent 1/2) and L2-normalize intensities.

This differs from dIon preprocessing. It is intentional: the external model
is evaluated under its own released input contract. Any skipped rows are
recorded, and dIon models must be reevaluated on the corresponding retained
row artifact for a strictly matched comparison.

The extracted v4 encoder consumes only processed peak m/z and intensity values.
The released checkpoint's full Casanovo decoder has `max_charge=10`, but that
decoder is not used here; the released-model mean-pooled encoder has no
separate charge restriction. Use the common charge-2--4 subset only when
making a directly matched GLEAMS comparison, not as an intrinsic Casanovo
eligibility rule.

## Frozen downstream-head protocol

For SQA, chimericity, oxidized methionine, and retention time, Casanovo v4 is
an external **frozen representation** baseline. dIon does not import
Casanovo or `depthcharge` during downstream fitting. Instead, it uses the
existing SQA-derived Lightning wrappers, decoder heads, optimizers, callbacks,
checkpointing, and task splits; only the encoder feature source is replaced by
a complete, split-indexed Casanovo cache.

For each ordinary downstream split, export under `dIon-env`, embed under the
isolated v4 environment, then materialize one cache directory:

```bash
# Repeat for each split. Use --input-parquet for SQA and --input-lance for the
# corresponding train.lance, val.lance, or test.lance auxiliary split.
python -u scripts/export_external_downstream_input.py \
  --input-lance "$SOURCE/train.lance" \
  --output-npz "$WORK/train_input.npz"

"$CASANOVO_V4_PY" -u scripts/embed_casanovo_v4_benchmark.py \
  --input-npz "$WORK/train_input.npz" \
  --checkpoint "$CASANOVO_V4_CKPT" \
  --output-npz "$WORK/train_casanovo_v4.npz" \
  --pooling peak_mean --batch-size 1024

python -u scripts/materialize_external_downstream_embeddings.py \
  --train-embeddings "$WORK/train_casanovo_v4.npz" \
  --val-embeddings "$WORK/val_casanovo_v4.npz" \
  --test-embeddings "$WORK/test_casanovo_v4.npz" \
  --output-cache "$CACHE"
```

The materializer requires each source split's `export_row_index` to contain
exactly every original dIon dataset index once. It fails if Casanovo
preprocessing skipped a spectrum: do not silently train or report a result on
an altered split. The cache manifest records source hashes, external artifacts,
row counts, and embedding width.

Run the normal frozen task configuration with the cache path and no dIon
encoder checkpoint:

```bash
python -u -m src.main --config "$MASTER" \
  --freeze_encoder 1 \
  --external_embedding_cache "$CACHE" \
  --encoder_weights "" \
  --pretrain 0
```

`--external_embedding_cache` is intentionally limited to frozen SQA-derived
tasks. It cannot be combined with a dIon checkpoint or full-encoder
fine-tuning. De novo remains a native Casanovo decoding baseline, and
representation retrieval/pair evaluation continues to use the external matched
artifact flow below.

## Evaluation flow

Run export and evaluation under dIon-env, but embedding under the dedicated
Casanovo environment:

```bash
python -u scripts/export_external_embedding_input.py \
  --input-parquet "$SOURCE_PARQUET" \
  --output-npz "$WORK/retrieval_input.npz" \
  --peptide-id-column peptide_id \
  --partition-column species \
  --seed 42 \
  --max-peptides-per-partition 100 \
  --max-spectra-per-peptide 10 \
  --metadata-columns peptide_id species source_row_index

"$CASANOVO_V4_PY" -u scripts/embed_casanovo_v4_benchmark.py \
  --input-npz "$WORK/retrieval_input.npz" \
  --checkpoint "$CASANOVO_V4_CKPT" \
  --output-npz "$WORK/retrieval_casanovo_v4.npz" \
  --batch-size 256

python -u scripts/materialize_external_matched_retrieval.py \
  --source-parquet "$SOURCE_PARQUET" \
  --external-embeddings "$WORK/retrieval_casanovo_v4.npz" \
  --output-parquet "$WORK/retrieval_casanovo_v4_matched.parquet"

python -u scripts/evaluate_external_embeddings.py \
  --embeddings-npz "$WORK/retrieval_casanovo_v4.npz" \
  --dataset-parquet "$WORK/retrieval_casanovo_v4_matched.parquet" \
  --evaluation-config "$RETRIEVAL_CONFIG" \
  --output-report "$RESULT_DIR/casanovo_v4.embedding_eval.json"
```

For pair evaluation, export every endpoint referenced by the fixed pair
manifest, embed those rows, then materialize the retained endpoint artifact and
evaluate the same static pairs after excluding any pair with a dropped endpoint.
