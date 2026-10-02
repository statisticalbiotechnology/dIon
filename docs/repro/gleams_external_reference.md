# GLEAMS External Reference Runner

This is the canonical procedure for the supervised GLEAMS external reference.
It is intentionally separate from `dIon-env`: TensorFlow 2.7/GLEAMS runs in
`gleams39`, while dIon creates and evaluates the row-aligned artifacts.

## Audited Local Installation

| Item | Value |
| --- | --- |
| Environment | `${WORK_ROOT}/conda-envs/gleams39` |
| Python | 3.9.19 |
| TensorFlow | 2.7.0 |
| Installed GLEAMS | `0.4.dev7+g13ebc74.d20250502` |
| Clone | `${WORK_ROOT}/GLEAMS` |
| Clone commit | `13ebc74490399b69ca13c9f182d64ad6a09c67c1` |
| Released weights | `gleams_82c0124b.hdf5` resolved by `gleams.config` |
| Embedding size | 32 |

The clone is deliberately treated as an external dependency. At audit time it
contains a small tracked edit to its legacy `environment.yml` (obsolete Intel
channel/runtime removal) and several untracked exploratory helper files,
including `generate_embeds_lance.py`. None of those helpers is canonical and
none is invoked by dIon.

`environments/gleams39.yml` is the dIon-owned recreation specification. It
does not silently install a moving Git revision. After creation, install an
explicitly checked-out clone with `pip install -e /path/to/GLEAMS`, verify the
commit above, and record the resulting environment/package versions in the
result manifest.

## Local Path Setup

Set these once in the shell or job script on any machine. The paths below are
examples only; do not copy the ClusterA-specific audited paths unless they are
actually valid on that host.

```bash
export PROJECT_ROOT=/path/to/dIon
export GLEAMS_CLONE=/path/to/GLEAMS
export GLEAMS_ENV=/path/to/conda/envs/gleams39
export GLEAMS_PYTHON="$GLEAMS_ENV/bin/python"
# Required for TensorFlow in this legacy CUDA-11 GLEAMS environment to find CUDA.
export LD_LIBRARY_PATH="$GLEAMS_ENV/lib:${LD_LIBRARY_PATH:-}"

# Required when recreating the environment or auditing the installed package.
test -d "$GLEAMS_CLONE/.git"
test -x "$GLEAMS_PYTHON"
"$GLEAMS_PYTHON" -m pip install -e "$GLEAMS_CLONE"
```

Run dIon export, materialization, and evaluation commands in `dIon-env`.
Invoke `$GLEAMS_PYTHON` directly for the GLEAMS embedding step; activation is
not required. On a cluster, set these variables inside the Sbatch script rather
than relying on interactive shell startup files.

The runner sets `TF_CPP_MIN_LOG_LEVEL=3` before importing GLEAMS to hide the
repetitive non-numerical Grappler `TensorDataset` diagnostics emitted for each
upstream-compatible prediction batch. Do not suppress Python package compatibility
warnings: repair the environment instead. TensorFlow 2.7 requires
`tensorflow-addons==0.15.0` for this runner.

## Scope and Charge Policy

The official released GLEAMS training configuration is `charges = (2, 5)`.
Although its precursor feature encoder has seven one-hot charge positions and
clips higher charges, that is not evidence that the released model was trained
or validated above charge 5. dIon's canonical runner therefore retains only
precursor charges 2, 3, 4, and 5. It records every rejected charge in the
embedding manifest as `skipped_charge_outside_2_to_5`.

The GLEAMS embedding includes precursor m/z, neutral mass, charge, binned
fragment features, and similarity to 500 reference spectra. It is a supervised
external reference, not a spectrum-only baseline and not a leakage-free upper
bound for SSL.

## Canonical Artifact Pipeline

The scripts form one row-preserving bridge. Do not use the external clone's
hard-coded Lance helpers or write embeddings back into a source dataset.

1. In `dIon-env`, export raw peaks and alignment keys using either
   `scripts/export_external_embedding_input.py` (retrieval) or
   `scripts/export_external_pair_input.py` (static pairs).
2. In `gleams39`, run `scripts/embed_gleams_benchmark.py`. It applies the
   released GLEAMS preprocessing, performs the charge-2--5 restriction, uses
   seed 42 for the reference spectra, and writes a 32-D embedding NPZ plus a
   hash-bearing manifest. For a GPU run, `--num-preprocess-workers 8` uses deterministic spawned CPU workers for the expensive peak preprocessing, while `--batch-size 4096` is validated on an A100 80 GB. The GPU inference remains ordered and the output matches the serial feature encoding exactly.
3. Back in `dIon-env`, materialize the post-GLEAMS retained retrieval/pair
   corpus with `scripts/materialize_external_matched_retrieval.py` or
   `scripts/materialize_external_matched_pairs.py`.
4. Evaluate the external artifact with `scripts/evaluate_external_embeddings.py`
   or `scripts/evaluate_external_pairs.py`. Evaluate every DINO comparator on
   exactly that same retained artifact/corpus.

The embedding script uses GLEAMS' `predict_on_batch` route rather than its
high-level helper because that is the route used by the upstream example and
remains compatible with TensorFlow 2.7's Keras input handling.

## Example Retrieval Invocation

Set paths for one immutable source benchmark. The source Parquet must expose
`mz_array`, `intensity_array`, `precursor_mz`, `precursor_charge`, the peptide
identity, partition, and unique alignment columns.

```bash
# dIon-env: export the exact source rows.
python -u scripts/export_external_embedding_input.py \
  --input-parquet "$SOURCE_PARQUET" \
  --output-npz "$WORK/source.npz" \
  --peptide-id-column peptide_id \
  --partition-column species \
  --metadata-columns peptide_id species source_row_index

# gleams39: embed only the externally valid rows. No activation required.
"$GLEAMS_PYTHON" -u scripts/embed_gleams_benchmark.py \
  --input-npz "$WORK/source.npz" \
  --output-npz "$WORK/gleams.npz" \
  --seed 42 \
  --batch-size 4096 \
  --num-preprocess-workers 8 \
  --preprocess-chunk-size 256

# dIon-env: materialize the row-aligned retained corpus, then evaluate it.
python -u scripts/materialize_external_matched_retrieval.py --help
python -u scripts/evaluate_external_embeddings.py --help
```

For a locked test evaluation, the source manifest, exported NPZ manifest, GLEAMS
embedding manifest, post-filtered corpus manifest, and both external/DINO
reports must be copied into the canonical `results/` experiment directory.
Never compare a GLEAMS-filtered number with an unfiltered DINO number.
