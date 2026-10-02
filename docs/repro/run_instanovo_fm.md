# InstaNovo-FM External Baseline

This protocol evaluates released InstaNovo-FM v0.1.0 as a frozen external
embedding model. dIon continues to own input selection, artifact alignment,
retrieval/pair scoring, frozen downstream heads, manifests, and result
ingestion. The isolated InstaNovo-FM environment performs only embedding.

## Reference and environment

The audited upstream clone is intentionally outside this repository:

```
${WORK_ROOT}/instanovo_fm
```

The clone checked for this integration was revision
`190d2ae51971d30f15aee57e1b7774e3939c956d`. It is a reference for the
released API/configuration, not a runtime import dependency.

Create an isolated environment under project storage rather than a home
directory. The verified ClusterB combination is Python 3.13.5,
`instanovo-fm==0.1.0`, `instanovo==1.2.2`, and CPU PyTorch. That CPU
environment is suitable for small smoke tests only; it cannot use CUDA.

```
module load Python/3.13.5-bare-gcc-2025b-eb
python -m venv ${WORK_ROOT}/venvs/instanovo-fm
```

### ClusterA encoder-only environment

As of 2026-09-12, the full PyPI dependency solve on ClusterA fails because
`pyopenms>=3.4.0` has no compatible wheel. That package belongs to InstaNovo's
theoretical-spectrum tooling and is not imported by the released foundation
encoder. Use the isolated encoder-only environment below instead; do not mix
it with dIon-env.

```
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba create -y -p ${WORK_ROOT}/conda-envs/instanovo-fm-py313 python=3.13 pip

INSTANOVO_PY=${WORK_ROOT}/conda-envs/instanovo-fm-py313/bin/python
$INSTANOVO_PY -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
$INSTANOVO_PY -m pip install --no-deps instanovo-fm==0.1.0 instanovo==1.2.2
$INSTANOVO_PY -m pip install \
  antlr4-python3-runtime omegaconf hydra-core jaxtyping einops pyyaml tqdm \
  numpy==2.2.6 rich typer polars pandas pyarrow==22.0.0 spectrum-utils \
  scikit-learn accelerate datasets gitpython jiwer matchms
```

This exact environment loaded the released `FoundationModel`, verified the
checkpoint checksum below, and embedded a dIon artifact on CPU. It omits
only pyOpenMS and unrelated analysis dependencies.

### ClusterA GPU execution

On ClusterA, run the pure-Python InstaNovo-FM package overlay through
dIon-env's existing CUDA-enabled PyTorch. The overlay is not installed into
dIon-env: `PYTHONPATH` only makes the isolated `instanovo` and
`instanovo_fm` Python modules visible to that one subprocess. The package
wheel is `py3-none-any`; PyTorch remains dIon-env's `cu128` build.

```bash
PROJECT_PY=${WORK_ROOT}/conda-envs/dIon-env/bin/python
INSTANOVO_OVERLAY=${WORK_ROOT}/conda-envs/instanovo-fm-gpu-overlay
PYTHONPATH="$INSTANOVO_OVERLAY" "$PROJECT_PY" -c   'import torch; from instanovo_fm.model.encoder import FoundationModel; print(torch.__version__, torch.cuda.is_available(), FoundationModel.__module__)'
```

This was verified on 2026-09-12 as dIon-env `torch 2.9.1+cu128` with CUDA
available and `instanovo-fm==0.1.0` loaded from the overlay. Large retrieval
and pair jobs must use this composition or an equivalent CUDA-enabled Python.
Do not enable AMP for canonical artifacts unless separately validated and
recorded.

On an A100 80 GB, use `INSTANOVO_BATCH_SIZE=2048`; start 40 GB A100 nodes at
`1024`, with eight CPU preprocessing workers.

Use an explicit local released checkpoint on compute nodes. It avoids a late
attempt to download through the package cache, which compute nodes cannot do.
The ClusterA checkpoint is:

```
${CHECKPOINT_ROOT}/instanovo_fm/instanovo-fm-v0.1.0.ckpt
```

Its expected MD5 is `255ad030baa987f0fefd6ecd61e6a803`.

## Released preprocessing and representation

`scripts/embed_instanovo_fm_benchmark.py` implements the released Foundation
Model's selected PyTorch preprocessing path:

1. retain m/z in `[50, 2500]`;
2. remove precursor peaks at 0 Da tolerance (therefore none for this release);
3. retain intensity >= 0.01;
4. retain the 200 most intense peaks when necessary;
5. square-root intensities and L2-normalize them;
6. divide m/z by 2500.

This is mandatory: the encoder expects normalized m/z and intensity values; it
does not perform these transforms internally. No precursor, charge,
instrument, or other metadata enters the encoder. Input metadata is retained
solely to align the emitted embedding to dIon's benchmark rows.

The canonical readout is `--readout mean_pool`, the L2-normalized mean of
valid contextualized peak-token embeddings. `--readout latent` is supported as
an explicitly labelled secondary analysis only.

## Single artifact

First use dIon-env to export a deterministic selected input NPZ. Then run the
embedder through the isolated package overlay and dIon-env CUDA. The resulting NPZ is accepted directly by
`materialize_external_matched_retrieval.py`,
`materialize_external_matched_pairs.py`, and
`materialize_external_downstream_embeddings.py`.

```
PROJECT_PY=${WORK_ROOT}/conda-envs/dIon-env/bin/python
INSTANOVO_OVERLAY=${WORK_ROOT}/conda-envs/instanovo-fm-gpu-overlay
INSTANOVO_CKPT=${CHECKPOINT_ROOT}/instanovo_fm/instanovo-fm-v0.1.0.ckpt

PYTHONPATH="$INSTANOVO_OVERLAY" "$PROJECT_PY" -u scripts/embed_instanovo_fm_benchmark.py \
  --input-npz <selected_input.npz> \
  --output-npz <instanovo_embeddings.npz> \
  --checkpoint "$INSTANOVO_CKPT" \
  --readout mean_pool \
  --device cuda \
  --threads 8 \
  --num-preprocess-workers 8 \
  --batch-size 1024
```

The output manifest records preprocessing, readout, package version, checkpoint
path/checksum, retained count, and any spectra removed during preprocessing.
For large GPU arrays, use the documented CUDA profiles above. Do not set `set -u`
in ClusterA job scripts.

## Evaluation Arrays

All arrays use the canonical released `mean_pool` readout and stage result JSON
outside the repository. The default launcher composes dIon-env CUDA with the
isolated package overlay; do not override `INSTANOVO_FM_PY` unless supplying an
equivalent CUDA-enabled interpreter.

Validation/model development:

```bash
# Shared charge-2--4 table with GLEAMS.
sbatch --array=0-2 jobs/ClusterA/instanovo_fm_representation_array.sbatch

# Charge-complete table, reported separately from the GLEAMS comparison.
sbatch --array=0-2 jobs/ClusterA/instanovo_fm_full_charge_validation_array.sbatch

# Frozen SQA, chimericity, oxidized-Met, and both RT heads.
CACHE_JOB=$(sbatch --parsable --array=0-3   jobs/ClusterA/instanovo_fm_downstream_validation_cache_array.sbatch)
sbatch --dependency="afterok:${CACHE_JOB}" --array=0-4   jobs/ClusterA/instanovo_fm_downstream_validation_array.sbatch
```

After validation selection is frozen, locked test only:

```bash
# Shared charge-2--4 and full-charge representation tables.
sbatch --array=0-2 --export=ALL,BENCHMARK_SPLIT=heldout_test,ALLOW_LOCKED_TEST=1   jobs/ClusterA/instanovo_fm_representation_array.sbatch
sbatch --array=0-2 --export=ALL,ALLOW_LOCKED_TEST=1   jobs/ClusterA/instanovo_fm_full_charge_locked_test_array.sbatch

# Append test features, then invoke the generic selected downstream tester with
# a frozen selection manifest pointing to the InstaNovo cache/checkpoints.
TEST_CACHE_JOB=$(sbatch --parsable --array=0-3 --export=ALL,ALLOW_LOCKED_TEST=1   jobs/ClusterA/instanovo_fm_downstream_locked_test_cache_array.sbatch)
```

The downstream cache materializer requires exact full-row retention. If released
InstaNovo preprocessing excludes any input spectrum, the cache array fails
rather than allowing an implicit split change; resolve that before training or
reporting a downstream baseline.
