# src/callbacks/linprobe_callback.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from functools import partial
from tqdm import tqdm
from src.utils import pad_peaks
from src.precursor_conditioning import condition_precursor_inputs, validate_precursor_conditioning
from src.embed_eval.metrics import evaluate_embedding_suite


EMBEDDING_QUALITY_METRICS = {
    "numerical_linear_algebra.stable_rank": "stable_rank",
    "numerical_linear_algebra.pseudo_condition_number": "cond",
    "linear_classifier_perspective.coherence_max": "coh_max",
    "high_dimensional_probability.self_cluster": "self_cluster",
    "related_work.rankme_normalized": "rankme_norm",
    "related_work.nesum": "nesum",
    "related_work.alpha_req": "alpha",
}


def train_manual(
    model: nn.Module,
    train_emb,
    train_lbl,
    val_emb,
    val_lbl,
    test_emb,
    test_lbl,
    lr: float,
    n_epochs: int,
    batch_size: int,
    device: torch.device,
):
    """
    Linear probing of embeddings, returns probe classifier metrics.
    """
    tr_loader = DataLoader(
        TensorDataset(train_emb, train_lbl), batch_size=batch_size, shuffle=True
    )
    va_loader = DataLoader(
        TensorDataset(val_emb, val_lbl), batch_size=batch_size, shuffle=False
    )
    te_loader = (
        DataLoader(
            TensorDataset(test_emb, test_lbl), batch_size=batch_size, shuffle=False
        )
        if test_emb is not None
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float("inf")
    best_state = None

    train_loss = val_loss = val_acc = test_acc = None

    for _ in tqdm(range(n_epochs), desc="Lin. probing embeddings"):
        # — train —
        model.train()
        loss_sum = 0.0
        n_total = 0
        for X, Y in tr_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, Y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * X.size(0)
            n_total += X.size(0)
        train_loss = loss_sum / n_total

        # — val —
        model.eval()
        loss_sum = 0.0
        n_total = 0
        corr = 0
        with torch.no_grad():
            for X, Y in va_loader:
                X, Y = X.to(device), Y.to(device)
                logits = model(X)
                loss = F.cross_entropy(logits, Y)
                loss_sum += loss.item() * X.size(0)
                n_total += X.size(0)
                corr += (logits.argmax(-1) == Y).sum().item()
        val_loss = loss_sum / n_total
        val_acc = corr / n_total

        # — track best —
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # reload best
    if best_state is not None:
        model.load_state_dict(best_state)

    # — test —
    if te_loader:
        model.eval()
        corr = n_total = 0
        with torch.no_grad():
            for X, Y in te_loader:
                X, Y = X.to(device), Y.to(device)
                pred = model(X).argmax(-1)
                corr += (pred == Y).sum().item()
                n_total += X.size(0)
        test_acc = corr / n_total

    return {
        "probe/lin_train_loss": train_loss,
        "probe/lin_val_loss": val_loss,
        "probe/lin_val_acc": val_acc,
        "probe/best_lin_val_loss": best_val_loss,
        "probe/lin_test_acc": test_acc,
    }


class ParquetSpectrumDataset(Dataset):
    """Loads one of the probe .parquet files you wrote."""

    def __init__(self, path):
        self.df = pd.read_parquet(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "mz_array": torch.tensor(row["mz_array"], dtype=torch.float32),
            "intensity_array": torch.tensor(
                row["intensity_array"], dtype=torch.float32
            ),
            "precursor_mz": torch.tensor(row["precursor_mz"], dtype=torch.float32),
            "precursor_charge": torch.tensor(row["precursor_charge"], dtype=torch.long),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


class ProbeDataModule(pl.LightningDataModule):
    """A tiny DataModule for your end-AA probe parquet files."""

    def __init__(self, cfg, collate_fn, batch_size):
        super().__init__()
        self.cfg = cfg
        self._collate = collate_fn
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = ParquetSpectrumDataset(self.cfg["train_parquet"])
        self.val_ds = ParquetSpectrumDataset(self.cfg["val_parquet"])
        self.test_ds = ParquetSpectrumDataset(self.cfg["test_parquet"])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )


class EndAAProbeCallback(Callback):
    """End-amino-acid probing for dIon embedders.

    This is also used by ``scripts/evaluate_end_aa_probe.py``. Its
    ``mass_input`` defaults to the canonical precursor mass emitted by
    ``pad_peaks``, matching DINO/DINOv2 pretraining.
    """

    VALID_MASS_INPUTS = {"precursor_mass", "precursor_mz"}

    def __init__(
        self,
        probing_cfg: dict,
        global_args,
        embedder_batch_size=64,
        precursor_conditioning: str = "conditioned",
        probe_every_n_steps: int | None = None,
        metric_prefix: str = "",
    ):
        super().__init__()
        self.cfg = probing_cfg["end_aa_pred"]
        self.K = self.cfg["k"]
        self.mass_input = self.cfg.get("mass_input", "precursor_mass")
        if self.mass_input not in self.VALID_MASS_INPUTS:
            raise ValueError(
                "end_aa_pred.mass_input must be one of "
                f"{sorted(self.VALID_MASS_INPUTS)}, got {self.mass_input!r}."
            )
        self.precursor_conditioning = validate_precursor_conditioning(
            precursor_conditioning
        )
        self.metric_prefix = metric_prefix.strip("/")
        self.probe_on_fit_start = bool(global_args.probe_on_fit_start)
        self.every_n_steps = int(
            global_args.probe_every_n_steps
            if probe_every_n_steps is None
            else probe_every_n_steps
        )
        if self.every_n_steps < 1:
            raise ValueError("probe_every_n_steps must be >= 1.")
        self._last_probe_step = None
        self._baseline_logged = False

        quality_cfg = self.cfg.get("embedding_quality", {})
        self.embedding_quality_enabled = bool(quality_cfg.get("enabled", True))
        self.embedding_quality_split = quality_cfg.get("split", "val")
        self.embedding_quality_center = bool(quality_cfg.get("center", True))
        self.embedding_quality_max_samples = quality_cfg.get("max_samples", 5000)
        self.embedding_quality_max_samples_self_cluster = quality_cfg.get(
            "max_samples_self_cluster", 5000
        )
        self.embedding_quality_seed = int(quality_cfg.get("seed", 0))
        self.embedding_quality_fn = (
            evaluate_embedding_suite if self.embedding_quality_enabled else None
        )

        # recreate the exact collate you use in your main
        self.collate_fn = partial(
            pad_peaks,
            max_peaks=global_args.max_peaks,
            precursor_mz_name="precursor_mz",
            precursor_mass_name=False,
            filter_method=global_args.peak_filter_method,
            intensity_scaling=global_args.intensity_scaling,
            min_mz=global_args.min_mz,
            max_mz=global_args.max_mz,
            min_intensity=global_args.min_intensity,
            remove_precursor_tol=global_args.remove_precursor_tol,
            min_peaks=global_args.min_peaks,
        )

        self.dm = ProbeDataModule(
            self.cfg, collate_fn=self.collate_fn, batch_size=embedder_batch_size
        )
        self.dm.setup()

    def _cache_embeddings(
        self, loader, embedder, device, *, split_name, show_progress
    ):
        """Cache labelled probe embeddings using the configured precursor input."""
        embeddings_list, labels_list = [], []
        batches = loader
        if show_progress:
            batches = tqdm(
                loader,
                desc=f"Caching {split_name} probe embeddings",
                unit="batch",
            )

        with torch.no_grad():
            for batch in batches:
                mz = batch["mz_array"].to(device)
                intensity = batch["intensity_array"].to(device)
                spectra = torch.stack([mz, intensity], dim=-1)
                batch_size, sequence_length, _ = spectra.shape
                lengths = batch["peak_lengths"].view(batch_size).to(device)
                pad_mask = (
                    torch.arange(sequence_length, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, sequence_length)
                    .ge(lengths.unsqueeze(1))
                )
                mass = (
                    batch[self.mass_input].to(device)
                    if embedder.use_mass
                    else None
                )
                charge = (
                    batch["precursor_charge"].to(device)
                    if embedder.use_charge
                    else None
                )
                mass, charge = condition_precursor_inputs(
                    mass, charge, self.precursor_conditioning
                )
                embeddings = embedder(spectra, pad_mask, mass, charge)
                embeddings_list.append(embeddings.cpu())
                labels_list.append(batch["labels"].cpu())

        if not embeddings_list:
            raise ValueError(f"The {split_name} end-AA probe split is empty.")
        return torch.cat(embeddings_list, 0), torch.cat(labels_list, 0)

    def evaluate_embedder(self, embedder, device, *, show_progress=False):
        """Run the full end-AA evaluation and return ``probe/...`` metrics."""
        # The embedder can wrap live training modules, so restore every touched
        # module's prior mode after cache extraction.
        modules_with_mode = []
        seen = set()
        for module in embedder.modules():
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            modules_with_mode.append((module, module.training))
        embedder.eval()

        try:
            train_embeddings, train_labels = self._cache_embeddings(
                self.dm.train_dataloader(),
                embedder,
                device,
                split_name="train",
                show_progress=show_progress,
            )
            val_embeddings, val_labels = self._cache_embeddings(
                self.dm.val_dataloader(),
                embedder,
                device,
                split_name="validation",
                show_progress=show_progress,
            )
            test_embeddings, test_labels = self._cache_embeddings(
                self.dm.test_dataloader(),
                embedder,
                device,
                split_name="test",
                show_progress=show_progress,
            )

            embedding_quality_metrics = self._embedding_quality_metrics(
                train_embeddings, val_embeddings, test_embeddings
            )

            if self.K > len(train_embeddings):
                raise ValueError(
                    f"end_aa_pred.k={self.K} exceeds the {len(train_embeddings)} "
                    "training examples available to the kNN probe."
                )
            normalized_train = F.normalize(train_embeddings, dim=1)
            normalized_val = F.normalize(val_embeddings, dim=1)
            similarities = normalized_val @ normalized_train.t()
            neighbors = similarities.topk(self.K, dim=1).indices
            predictions = torch.stack(
                [torch.bincount(train_labels[index]).argmax() for index in neighbors]
            )
            knn_acc = (predictions == val_labels).float().mean().item()

            head = nn.Linear(
                embedder.running_units, int(train_labels.unique().numel())
            )
            probe_metrics = train_manual(
                head,
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                test_embeddings,
                test_labels,
                lr=self.cfg["blr"],
                n_epochs=self.cfg["epochs"],
                batch_size=self.cfg["batch_size"],
                device=device,
            )
            return {
                "probe/knn_val_acc": knn_acc,
                **embedding_quality_metrics,
                **probe_metrics,
            }
        finally:
            for module, was_training in modules_with_mode:
                module.train(was_training)

    @rank_zero_only
    def _run_probe(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *,
        is_baseline: bool,
    ):
        device = pl_module.device
        embedder = pl_module.get_embedder().to(device)
        metrics = self.evaluate_embedder(embedder, device)
        if self.metric_prefix:
            metrics = {self.metric_prefix + "/" + name: value for name, value in metrics.items()}

        # Attach the full condition-specific probe result to the current step.
        if isinstance(trainer.logger, WandbLogger):
            knn_metric = (
                (self.metric_prefix + "/" if self.metric_prefix else "")
                + "probe/knn_val_acc"
            )
            trainer.logger.experiment.log({knn_metric: metrics[knn_metric]}, commit=False)
            trainer.logger.experiment.log(
                {
                    "global_step": int(trainer.global_step),
                    "epoch": trainer.current_epoch,
                    (self.metric_prefix + "/" if self.metric_prefix else "")
                    + "probe/is_baseline": int(is_baseline),
                    **metrics,
                },
            )

    def _embedding_quality_metrics(self, X_tr, X_val, X_te):
        if not self.embedding_quality_enabled or self.embedding_quality_fn is None:
            return {}

        split = self.embedding_quality_split
        if split == "train":
            embeddings = X_tr
        elif split == "val":
            embeddings = X_val
        elif split == "test":
            embeddings = X_te
        elif split == "all":
            embeddings = torch.cat([X_tr, X_val, X_te], dim=0)
        else:
            raise ValueError(
                "embedding_quality.split must be one of "
                "['all', 'test', 'train', 'val'], "
                f"got {split!r}."
            )
        max_samples = self.embedding_quality_max_samples
        if max_samples is not None and embeddings.shape[0] > int(max_samples):
            generator = torch.Generator(device=embeddings.device)
            generator.manual_seed(self.embedding_quality_seed)
            indices = torch.randperm(
                embeddings.shape[0], generator=generator, device=embeddings.device
            )[: int(max_samples)]
            embeddings = embeddings[indices]

        raw_metrics = self.embedding_quality_fn(
            embeddings,
            center=self.embedding_quality_center,
            device="cpu",
            max_samples_self_cluster=self.embedding_quality_max_samples_self_cluster,
        )
        metrics = {}
        for key, name in EMBEDDING_QUALITY_METRICS.items():
            value = float(raw_metrics[key])
            if not math.isfinite(value):
                continue
            metrics[f"probe/{split}/{name}"] = value
        return metrics

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.probe_on_fit_start:
            return
        if self._baseline_logged:
            return
        self._run_probe(trainer, pl_module, is_baseline=True)
        # Ensure fit starts in train mode even if Lightning has not toggled yet.
        pl_module.train()
        self._baseline_logged = True
        self._last_probe_step = int(trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        del outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step < 1:
            return
        if step % self.every_n_steps != 0:
            return
        if self._last_probe_step == step:
            return

        self._run_probe(trainer, pl_module, is_baseline=False)
        self._last_probe_step = step
