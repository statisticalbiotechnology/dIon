"""Frozen dense-encoder de novo monitor for DINO/DINOv2 training.

The callback extracts each split's EMA-teacher peak-token memory once per
trigger. It then trains a small decoder only on that CPU-resident cache, so
neither decoder optimization nor validation reruns the encoder.
"""

from __future__ import annotations

from functools import partial
from typing import Iterable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.collate_functions import pad_peptides
import src.casanovo_eval as evaluate
from src.data.lance_data_module import _rows_to_batch_dict
from src.data.lance_datasets import SafeLanceDataset
from src.data.lance_loader_utils import make_lance_safe_loader
from src.data.unified_tokenizer import PeptideTokenizer, UNIFIED_RESIDUES
from src.models.casanovo.decoder_interface import PeptideDecoder
from src.models.casanovo.masses import PeptideMass
from src.precursor_conditioning import (
    VALID_PRECURSOR_CONDITIONING,
    condition_precursor_inputs,
)
from src.wrappers.beam_search import BeamSearchInterface


_CACHE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class _CachedDenseBeamSearch(BeamSearchInterface):
    """Canonical Casanovo beam search over already-cached encoder memory."""

    def __init__(
        self,
        decoder: PeptideDecoder,
        tokenizer: PeptideTokenizer,
        cfg: dict,
        max_length: int,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.residues = tokenizer.residues
        self.pad_token = tokenizer.pad_token_id
        self.stop_token = tokenizer.eos_token_id
        self.max_length = int(cfg.get("max_length", max_length))
        self.precursor_mass_tol = float(cfg.get("precursor_mass_tol", 50))
        self.isotope_error_range = tuple(cfg.get("isotope_error_range", (0, 1)))
        self.min_peptide_len = int(cfg.get("min_peptide_len", 6))
        self.n_beams = int(cfg.get("n_beams", 1))
        self.top_match = int(cfg.get("top_match", 1))
        self.peptide_mass_calculator = PeptideMass(self.residues)

    @property
    def device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def _encode_spectra(self, spectra, spectrum_padding_mask, precursors=None):
        del precursors
        return {"emb": spectra, "mask": spectrum_padding_mask}


def _modules_with_mode(module: torch.nn.Module) -> list[tuple[torch.nn.Module, bool]]:
    """Capture each nested module's prior train/eval mode once."""
    captured = []
    seen = set()
    for child in module.modules():
        if id(child) not in seen:
            seen.add(id(child))
            captured.append((child, child.training))
    return captured


class DenseDeNovoProbeCallback(Callback):
    """Train a tiny decoder on cached frozen per-peak encoder memories.

    This is a development monitor, not the final de novo benchmark. Its data
    use a deliberately reduced vocabulary and short unmodified peptides, so
    repeated evaluation is feasible during SSL training.
    """

    def __init__(
        self,
        probing_cfg: dict,
        global_args,
        embedder_batch_size: int = 64,
        precursor_conditioning: str = "conditioned",
        probe_every_n_steps: int | None = None,
    ):
        super().__init__()
        self.cfg = probing_cfg["dense_denovo_probe"]
        self.global_args = global_args
        self.probe_on_fit_start = bool(global_args.probe_on_fit_start)
        self.every_n_steps = int(
            global_args.probe_every_n_steps
            if probe_every_n_steps is None
            else probe_every_n_steps
        )
        if self.every_n_steps < 1:
            raise ValueError("probe_every_n_steps must be >= 1.")
        self._last_probe_step = None

        self.dataset_cfg = self.cfg["dataset"]
        self.label_name = self.dataset_cfg.get("label_name", "seq")
        self.precursor_mz_name = self.dataset_cfg.get("precursor_mz_name", "precursor_mz")
        self.precursor_mass_name = self.dataset_cfg.get("precursor_mass_name", False)
        if not self.precursor_mz_name and not self.precursor_mass_name:
            raise ValueError("dense_denovo_probe requires precursor m/z or mass input.")

        vocab = self.cfg.get("vocab", "ADEFIKLRSTV")
        if not isinstance(vocab, str) or not vocab:
            raise ValueError("dense_denovo_probe.vocab must be a non-empty residue string.")
        if len(set(vocab)) != len(vocab) or any(token not in UNIFIED_RESIDUES for token in vocab):
            raise ValueError("dense_denovo_probe.vocab must contain unique supported residue tokens.")
        self.tokenizer = PeptideTokenizer(
            residues={token: UNIFIED_RESIDUES[token] for token in vocab},
            add_unk_token=False,
        )

        extraction_cfg = self.cfg.get("extraction", {})
        self.extraction_batch_size = int(extraction_cfg.get("batch_size", embedder_batch_size))
        self.extraction_num_workers = int(extraction_cfg.get("num_workers", 0))
        self.extraction_pin_memory = bool(extraction_cfg.get("pin_memory", True))
        cache_dtype_name = self.cfg.get("cache_dtype", "float16")
        if cache_dtype_name not in _CACHE_DTYPES:
            raise ValueError(
                "dense_denovo_probe.cache_dtype must be one of "
                f"{sorted(_CACHE_DTYPES)}, got {cache_dtype_name!r}."
            )
        self.cache_dtype = _CACHE_DTYPES[cache_dtype_name]
        self.cache_dtype_name = cache_dtype_name

        self.training_cfg = self.cfg.get("training", {})
        self.decoder_cfg = self.cfg.get("decoder", {})
        self.seed = int(self.cfg.get("seed", global_args.seed))
        self.encoder_precursor_conditioning = precursor_conditioning
        if self.encoder_precursor_conditioning not in VALID_PRECURSOR_CONDITIONING:
            raise ValueError(
                "dense_denovo_probe mode must be one of "
                f"{sorted(VALID_PRECURSOR_CONDITIONING)}."
            )

        self.max_peptide_length = int(self.cfg.get("max_peptide_length", 10))
        self.collate_fn = partial(
            pad_peptides,
            max_peaks=global_args.max_peaks,
            max_length=self.max_peptide_length,
            pad_token_id=self.tokenizer.pad_token_id,
            tokenizer=self.tokenizer,
            label_name=self.label_name,
            precursor_mz_name=self.precursor_mz_name,
            precursor_mass_name=self.precursor_mass_name,
            filter_method=global_args.peak_filter_method,
            intensity_scaling=global_args.intensity_scaling,
            min_mz=global_args.min_mz,
            max_mz=global_args.max_mz,
            min_intensity=global_args.min_intensity,
            remove_precursor_tol=global_args.remove_precursor_tol,
        )
        columns = [self.label_name, "precursor_charge"]
        columns.extend(name for name in (self.precursor_mz_name, self.precursor_mass_name) if name)
        self.columns = list(dict.fromkeys(columns))
        self.datasets = {
            split: SafeLanceDataset(self.dataset_cfg[f"{split}_lance"], columns=self.columns)
            for split in ("train", "val", "test")
        }

    def _loader(self, split: str) -> DataLoader:
        return make_lance_safe_loader(
            dataset=self.datasets[split],
            batch_size=self.extraction_batch_size,
            num_workers=self.extraction_num_workers,
            drop_last=False,
            pin_memory=self.extraction_pin_memory,
            shuffle=False,
            collate_fn=partial(_rows_to_batch_dict, collate_fn=self.collate_fn),
            use_safe=True,
        )

    @staticmethod
    def _padding_mask(spectra: torch.Tensor, peak_lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(spectra.shape[1], device=spectra.device).unsqueeze(0)
        return positions >= peak_lengths.view(-1, 1).to(spectra.device)

    def _cache_split(
        self,
        loader: Iterable[dict],
        encoder: torch.nn.Module,
        device: torch.device,
        *,
        split_name: str,
        show_progress: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a split once and retain only CPU decoder inputs."""
        memories, memory_masks, precursors, input_tokens, target_tokens = [], [], [], [], []
        batches = tqdm(loader, desc=f"Caching dense {split_name} memories", unit="batch") if show_progress else loader
        with torch.no_grad():
            for batch in batches:
                mz = batch["mz_array"].to(device)
                intensity = batch["intensity_array"].to(device)
                spectra = torch.stack([mz, intensity], dim=-1)
                padding_mask = self._padding_mask(spectra, batch["peak_lengths"])
                # ``pad_peptides`` pads only to each batch's local maximum.
                # Use the fixed DINO cap before encoding so cached memories can
                # be concatenated across extraction batches without changing the
                # masked encoder input.
                fixed_peak_count = int(self.global_args.max_peaks)
                if spectra.shape[1] < fixed_peak_count:
                    spectra = F.pad(spectra, (0, 0, 0, fixed_peak_count - spectra.shape[1]))
                    padding_mask = F.pad(
                        padding_mask, (0, fixed_peak_count - padding_mask.shape[1]), value=True
                    )
                charge = torch.clamp(
                    batch["precursor_charge"].to(device), 1, int(self.global_args.max_charge)
                )
                mass = batch["precursor_mass"].to(device)
                mz_precursor = batch["precursor_mz"].to(device)
                conditioned_mass, conditioned_charge = condition_precursor_inputs(
                    mass, charge, self.encoder_precursor_conditioning
                )
                encoded = encoder(
                    spectra,
                    key_padding_mask=padding_mask,
                    mass=conditioned_mass if getattr(encoder, "use_mass", False) else None,
                    charge=conditioned_charge if getattr(encoder, "use_charge", False) else None,
                )
                memory = encoded["emb"].detach().to("cpu", dtype=self.cache_dtype)
                memory_mask = encoded.get("mask")
                if memory_mask is None:
                    memory_mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=device)
                memory_mask = memory_mask.detach().to("cpu", dtype=torch.bool)

                tokens = batch["intseq"]
                peptide_lengths = batch["peptide_lengths"].view(-1)
                if tokens.shape[1] > self.max_peptide_length:
                    raise ValueError("Dense probe peptide exceeds configured max_peptide_length.")
                tokens = F.pad(
                    tokens, (0, self.max_peptide_length - tokens.shape[1]),
                    value=self.tokenizer.pad_token_id,
                )
                targets = torch.full(
                    (tokens.shape[0], self.max_peptide_length + 1),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                targets[:, : self.max_peptide_length] = tokens
                targets[torch.arange(tokens.shape[0]), peptide_lengths] = self.tokenizer.eos_token_id

                memories.append(memory)
                memory_masks.append(memory_mask)
                precursors.append(torch.stack([mass.cpu(), charge.cpu(), mz_precursor.cpu()], dim=1))
                input_tokens.append(tokens.cpu())
                target_tokens.append(targets)

        if not memories:
            raise ValueError(f"The {split_name} dense de novo probe split is empty.")
        return tuple(torch.cat(parts, dim=0) for parts in (
            memories, memory_masks, precursors, input_tokens, target_tokens
        ))

    def _decoder(self, d_model: int) -> PeptideDecoder:
        n_head = int(self.decoder_cfg.get("n_head", 2))
        if d_model % n_head:
            raise ValueError(
                f"dense probe decoder d_model={d_model} is not divisible by n_head={n_head}."
            )
        return PeptideDecoder(
            dim_model=d_model,
            n_head=n_head,
            dim_feedforward=int(self.decoder_cfg.get("dim_feedforward", 128)),
            n_layers=int(self.decoder_cfg.get("n_layers", 2)),
            dropout=float(self.decoder_cfg.get("dropout", 0.0)),
            pos_encoder=True,
            reverse=False,
            vocab_size=self.tokenizer.vocab_size,
            max_charge=int(self.global_args.max_charge),
            padding_idx=self.tokenizer.pad_token_id,
        )

    def _teacher_forced_loss(
        self,
        decoder: PeptideDecoder,
        cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
        batch_size: int,
    ) -> float:
        """Teacher-forced cross-entropy used only to optimize the tiny head."""
        loader = DataLoader(TensorDataset(*cached), batch_size=batch_size, shuffle=False)
        total_loss = total_tokens = 0
        decoder.eval()
        with torch.no_grad():
            for memory, memory_mask, precursors, tokens, targets in loader:
                logits, _ = decoder(
                    tokens.to(device=device, dtype=torch.long),
                    precursors.to(device=device, dtype=torch.float32),
                    memory.to(device=device, dtype=torch.float32),
                    memory_mask.to(device=device, dtype=torch.bool),
                )
                targets = targets.to(device=device, dtype=torch.long)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.flatten(),
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction="sum",
                )
                total_loss += float(loss.item())
                total_tokens += int((targets != self.tokenizer.pad_token_id).sum().item())
        return total_loss / max(total_tokens, 1)

    def _canonical_denovo_metrics(
        self,
        decoder: PeptideDecoder,
        cached: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
        batch_size: int,
    ) -> dict[str, float]:
        """Use canonical beam search and Casanovo mass-aware match metrics."""
        loader = DataLoader(TensorDataset(*cached), batch_size=batch_size, shuffle=False)
        search = _CachedDenseBeamSearch(
            decoder,
            self.tokenizer,
            self.cfg.get("decoding", {}),
            self.max_peptide_length,
        )
        peptides_true, peptides_pred = [], []
        decoder.eval()
        with torch.no_grad():
            for memory, memory_mask, precursors, _, targets in loader:
                targets = targets.to(device=device, dtype=torch.long)
                beam = search.beam_search_decode(
                    memory.to(device=device, dtype=torch.float32),
                    precursors.to(device=device, dtype=torch.float32),
                    memory_mask.to(device=device, dtype=torch.bool),
                )
                peptides_true.extend(
                    self.tokenizer.detokenize(
                        targets,
                        pad_token_idx=self.tokenizer.pad_token_id,
                        EOS_token_idx=self.tokenizer.eos_token_id,
                        exclude_stop=True,
                    )
                )
                peptides_pred.extend(
                    predictions[0][2] if predictions else [] for predictions in beam
                )
        aa_prec, aa_recall, pep_prec = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(peptides_true, peptides_pred, self.tokenizer.residues)
        )
        return {
            "aa_prec": aa_prec,
            "aa_recall": aa_recall,
            "pep_prec": pep_prec,
        }

    def _train_decoder(
        self,
        cached: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        device: torch.device,
        *,
        show_progress: bool,
    ) -> tuple[PeptideDecoder, int]:
        decoder = self._decoder(int(cached["train"][0].shape[-1])).to(device)
        batch_size = int(self.training_cfg.get("batch_size", 64))
        epochs = int(self.training_cfg.get("epochs", 4))
        if epochs < 1:
            raise ValueError("Dense de novo probe training.epochs must be >= 1.")
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=float(self.training_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.0)),
        )
        train_loader = DataLoader(
            TensorDataset(*cached["train"]),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        epoch_range = range(epochs)
        if show_progress:
            epoch_range = tqdm(
                epoch_range, desc="Training cached dense de novo probe", unit="epoch"
            )
        for epoch in epoch_range:
            decoder.train()
            for memory, memory_mask, precursors, tokens, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = decoder(
                    tokens.to(device=device, dtype=torch.long),
                    precursors.to(device=device, dtype=torch.float32),
                    memory.to(device=device, dtype=torch.float32),
                    memory_mask.to(device=device, dtype=torch.bool),
                )
                target = targets.to(device=device, dtype=torch.long)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.flatten(),
                    ignore_index=self.tokenizer.pad_token_id,
                )
                loss.backward()
                optimizer.step()
            val_loss = self._teacher_forced_loss(decoder, cached["val"], device, batch_size)
            if show_progress:
                epoch_range.set_postfix(
                    val_loss=f"{val_loss:.4f}",
                    epoch=epoch + 1,
                )
        return decoder, epochs

    def evaluate_encoder(
        self, encoder: torch.nn.Module, device: torch.device, *, show_progress: bool = False
    ) -> dict[str, float]:
        """Run one complete cache-then-decode monitor evaluation."""
        modes = _modules_with_mode(encoder)
        encoder.eval()
        try:
            cached = {
                split: self._cache_split(
                    self._loader(split), encoder, device, split_name=split, show_progress=show_progress
                )
                for split in ("train", "val", "test")
            }
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(self.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(self.seed)
                decoder, epochs = self._train_decoder(cached, device, show_progress=show_progress)
            batch_size = int(self.training_cfg.get("batch_size", 64))
            metrics = {
                "probe/dense_denovo_decoder_epochs": float(epochs),
                "probe/dense_denovo_cache_train_examples": float(cached["train"][0].shape[0]),
                "probe/dense_denovo_cache_val_examples": float(cached["val"][0].shape[0]),
                "probe/dense_denovo_cache_test_examples": float(cached["test"][0].shape[0]),
            }
            for split in ("train", "val", "test"):
                metrics[f"probe/dense_denovo_{split}_loss"] = self._teacher_forced_loss(
                    decoder, cached[split], device, batch_size
                )
                for name, value in self._canonical_denovo_metrics(
                    decoder, cached[split], device, batch_size
                ).items():
                    metrics[f"probe/dense_denovo_{split}_{name}"] = value
            return metrics
        finally:
            for module, was_training in modes:
                module.train(was_training)

    @rank_zero_only
    def _run_probe(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *, is_baseline: bool):
        step = int(trainer.global_step)
        print(
            "[dense-denovo-probe] start "
            f"step={step} baseline={int(is_baseline)} "
            f"conditioning={self.encoder_precursor_conditioning}"
        )
        encoder = pl_module.get_encoder(trainable=False).to(pl_module.device)
        metrics = self.evaluate_encoder(encoder, pl_module.device)
        print(
            "[dense-denovo-probe] complete "
            f"step={step} val_aa_prec={metrics['probe/dense_denovo_val_aa_prec']:.4f} "
            f"val_pep_prec={metrics['probe/dense_denovo_val_pep_prec']:.4f}"
        )
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(
                {
                    "global_step": step,
                    "epoch": trainer.current_epoch,
                    "probe/dense_denovo_is_baseline": int(is_baseline),
                    **metrics,
                }
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.probe_on_fit_start:
            self._run_probe(trainer, pl_module, is_baseline=True)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx
    ) -> None:
        del outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step < 1 or step % self.every_n_steps != 0 or step == self._last_probe_step:
            return
        self._last_probe_step = step
        self._run_probe(trainer, pl_module, is_baseline=False)
