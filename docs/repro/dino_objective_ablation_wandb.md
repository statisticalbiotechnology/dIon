# DINO Objective Ablation W&B Curves

This package preserves the training curves for the approximately 100-epoch
objective ablation separately from the final-metric ledger.

## Canonical runs

| Variant | W&B run | Role | Peak cap |
|---|---|---|---:|
| Pure distractor-global | `user/dion-pretraining/7cee11xe` | primary collapse/null control | 200 |
| Full dual objective | `user/dion-pretraining/hjdp7mn2` | primary selected objective | 200 |
| H26 inverse control | `user/dion-pretraining/8fdixqb0` | primary mechanism control | 200 |
| Plain DINO | `user/dion-pretraining/7waklwb8` | translation control | 300 |

The three primary runs use the same bacterial corpus, architecture, optimizer,
schedule, preprocessing, filtering, compact validation callbacks, and roughly
100-epoch horizon. Plain DINO predates the matched 200-peak ablation and is
therefore retained as contextual evidence rather than presented as perfectly
matched.

## Export

From the dIon repository with an authenticated W&B account:

```bash
${WORK_ROOT}/conda-envs/dIon-env/bin/python -u \
  scripts/export_dino_objective_ablation_wandb.py
```

This writes:

```text
results/representation/dino_objective_ablation_100epoch_wandb/
  curves_long.csv
  manifest.json
```

The CSV is plot-ready long-form data. The manifest records exact W&B paths,
run IDs, names, states, complete run configurations, metric availability,
point counts, endpoints, source revision, and a SHA-256 digest of the CSV.

## Axis and integrity checks

The exporter uses W&B `scan_history`, not sampled `history`. Strict pair
callback metrics are indexed by their co-logged `trainer/global_step` values;
probe and effective-rank metrics are indexed by their co-logged `epoch` values.
It rejects absent required metrics, non-finite values, duplicate or decreasing
axes, unexpected run state or peak cap, and disagreement between each final
history value and its W&B summary value.

The pair curves are same-charge, 10-ppm validation callback results for the
aggregate, bacterial, Kingdoms, and NineSpecies V2 development subsets. Plain
DINO has no corresponding pair callback history. Retrieval is not logged as a
W&B trajectory for these runs; any final offline retrieval numbers must retain
their own report provenance and must not be represented as training curves.

These validation trajectories are suitable for objective ablation and model
selection. They are not held-out-test estimates and must remain separate from
frozen test evaluations.
