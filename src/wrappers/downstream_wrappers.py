import math
import os
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
from src.wrappers.base_wrapper import BaseDownstreamWrapper
import hashlib

from src.casanovo_eval import aa_match_batch, aa_match_metrics
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchmetrics.classification import BinaryAUROC
from src.precursor_conditioning import (
    VALID_PRECURSOR_CONDITIONING,
    condition_precursor_inputs,
)
from src.wrappers.beam_search import BeamSearchInterface
from src.schedulers import CosineWarmupScheduler
import src.casanovo_eval as evaluate
from src.models.casanovo.masses import PeptideMass
import einops
from src.data_augmentation import BatchedRandomSelectionAugmentation
from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation


def _calc_binary_metrics(logits, targets):
    # NumPy cannot convert bfloat16 tensors directly. Metrics are not part of
    # the gradient path, so promote only this detached copy for sklearn.
    probs = torch.sigmoid(logits).detach().float().cpu().squeeze(-1).numpy()
    probs = probs.reshape(-1)
    preds = (probs > 0.5).astype(int)
    y = targets.detach().cpu().numpy().reshape(-1).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
    }

    return {k: float(np.float32(v)) for k, v in metrics.items()}


def fill_pad_after_first_EOS(prediction, pad_token, eos_token):
    pred_without_eos = prediction.clone()
    eos_mask = prediction == eos_token
    absent_eos = eos_mask.sum(1) == 0
    # Find the position of the first predicted EOS token
    eos_positions = torch.argmax(eos_mask.int(), dim=1)
    eos_positions[absent_eos] = prediction.shape[-1]

    inds = (
        torch.arange(prediction.shape[1], device=prediction.device)
        .unsqueeze(0)
        .repeat((prediction.shape[0], 1))
    )
    forward_fill_mask = inds > torch.ones_like(inds) * eos_positions.unsqueeze(1)
    pred_without_eos[forward_fill_mask] = pad_token
    return pred_without_eos


class DeNovoTeacherForcing(BaseDownstreamWrapper, BeamSearchInterface):
    """Canonical teacher-forced de novo trainer with Casanovo-compatible decoding."""

    def __init__(self, encoder, decoder, global_args, collate_fn=None, tokenizer=None, task_dict=None):
        super().__init__(encoder, decoder, global_args, collate_fn=collate_fn, task_dict=task_dict)
        BeamSearchInterface.__init__(self)
        if tokenizer is None:
            raise ValueError("Tokenizer is required for de novo training.")
        self.automatic_optimization = True
        self.tokenizer = tokenizer
        self.residues = tokenizer.residues
        self.pad_token = tokenizer.pad_token_id
        self.stop_token = tokenizer.eos_token_id
        self.vocab_size = tokenizer.vocab_size
        if hasattr(self.decoder, "reverse"):
            self.decoder.reverse = tokenizer.reverse
        self.max_length = int(task_dict.get("max_length", global_args.max_length))
        self.precursor_mass_tol = task_dict.get("precursor_mass_tol", 50)
        self.isotope_error_range = task_dict.get("isotope_error_range", (0, 1))
        self.min_peptide_len = task_dict.get("min_peptide_len", 6)
        self.n_beams = task_dict.get("n_beams", 1)
        self.top_match = task_dict.get("top_match", 1)
        self.calculate_precision = task_dict.get("calculate_precision", True)
        self.log_predictions = bool(task_dict.get("log_predictions", False))
        self.log_species_metrics = bool(task_dict.get("log_species_metrics", False))
        self._test_species_counts: dict[str, list[int]] = {}
        self.cheap_val = bool(task_dict.get("cheap_validation", task_dict.get("cheap_val", False)))
        self.train_label_smoothing = task_dict.get("train_label_smoothing", 0.0)
        self.learning_rate = float(task_dict.get("learning_rate", self.lr))
        self.weight_decay = float(task_dict.get("weight_decay", 0.0))
        self.warmup_steps = int(task_dict.get("warmup_steps", 0))
        self.cosine_period_steps = int(task_dict.get("cosine_period_steps", 1))
        self.encoder_precursor_conditioning = task_dict.get(
            "encoder_precursor_conditioning", "conditioned"
        )
        self.target_swapped_distractor_mix = None
        self.target_swapped_global_crop = None
        target_swapped_mix = task_dict.get("target_swapped_distractor_mix", {})
        if target_swapped_mix.get("enabled", False):
            if not bool(global_args.freeze_encoder):
                raise ValueError(
                    "target_swapped_distractor_mix is only supported with a frozen encoder."
                )
            selection_mode = target_swapped_mix.get("selection_mode")
            if selection_mode != "random_batched":
                raise ValueError(
                    "target_swapped_distractor_mix must use DINO's "
                    "selection_mode='random_batched'."
                )
            global_crops_scale = target_swapped_mix.get("global_crops_scale")
            if global_crops_scale != [0.95, 0.95]:
                raise ValueError(
                    "target_swapped_distractor_mix must use the source "
                    "Hybrid/Gram global_crops_scale [0.95, 0.95]."
                )
            expected_mix_values = {
                "strength": 1.0,
                "condition_separation_ppm": 10.0,
                "neutral_mass_separation_ppm": 10.0,
                "merge_ppm": 5.0,
                "intensity_normalization": "none",
            }
            for name, expected in expected_mix_values.items():
                configured = target_swapped_mix.get(name)
                if configured != expected:
                    raise ValueError(
                        "target_swapped_distractor_mix must preserve the "
                        f"source Gram setting {name}={expected!r}, got {configured!r}."
                    )
            self.target_swapped_global_crop = BatchedRandomSelectionAugmentation(
                global_crops_scale=global_crops_scale,
                local_crops_scale=target_swapped_mix.get("local_crops_scale", [0.6, 0.6]),
                num_global_crops=1,
                num_local_crops=0,
                padding_value=0,
            )
            self.target_swapped_distractor_mix = BatchedStudentDistractorMixAugmentation(
                mix_apply_to="global_only",
                distractor_sampling="per_view",
                condition_separation_ppm=target_swapped_mix.get(
                    "condition_separation_ppm", 10.0
                ),
                neutral_mass_separation_ppm=target_swapped_mix.get(
                    "neutral_mass_separation_ppm", 10.0
                ),
                merge_ppm=target_swapped_mix.get("merge_ppm", 5.0),
                intensity_normalization=target_swapped_mix.get(
                    "intensity_normalization", "none"
                ),
                padding_value=0,
            )
            self.target_swapped_mix_strength = float(target_swapped_mix.get("strength", 1.0))
        self.freeze_encoder = bool(global_args.freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        valid_encoder_conditioning = set(VALID_PRECURSOR_CONDITIONING) | {"none"}
        if self.encoder_precursor_conditioning not in valid_encoder_conditioning:
            raise ValueError(
                "encoder_precursor_conditioning must be one of "
                f"{sorted(valid_encoder_conditioning)}, got "
                f"{self.encoder_precursor_conditioning!r}."
            )
        self.peptide_mass_calculator = PeptideMass(self.residues)
        self.softmax = torch.nn.Softmax(dim=2)
        self.celoss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token, label_smoothing=self.train_label_smoothing)
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token)
        self.TASK_NAME = "denovo_tf"

    def _get_padding_mask(self, spectra: torch.Tensor, seq_lengths: torch.Tensor):
        positions = torch.arange(spectra.shape[1], device=spectra.device).unsqueeze(0)
        return positions >= seq_lengths.to(spectra.device).reshape(-1, 1)

    def _encode_spectra(self, spectra, spectrum_padding_mask, precursors=None):
        if self.encoder_precursor_conditioning == "none":
            return self.encoder(spectra, key_padding_mask=spectrum_padding_mask)
        if precursors is None:
            raise ValueError("Precursor-conditioned encoder input requires precursors.")
        mass, charge = condition_precursor_inputs(
            precursors[:, 0], precursors[:, 1], self.encoder_precursor_conditioning
        )
        return self.encoder(
            spectra,
            key_padding_mask=spectrum_padding_mask,
            mass=mass if getattr(self.encoder, "use_mass", False) else None,
            charge=charge if getattr(self.encoder, "use_charge", False) else None,
        )

    def _parse_batch(self, batch, Eval=False):
        mz_array = batch["mz_array"].to(self.device)
        intensity_array = batch["intensity_array"].to(self.device)
        spectra = torch.stack([mz_array, intensity_array], dim=-1)
        peak_lengths = batch["peak_lengths"].to(self.device).reshape(-1)
        spectrum_padding_mask = self._get_padding_mask(spectra, peak_lengths)
        charge = torch.clamp(batch["precursor_charge"].to(self.device), 1, 10)
        mass = batch["precursor_mass"].to(self.device)
        mz = batch["precursor_mz"].to(self.device)
        precursors = torch.stack([mass, charge, mz], dim=1)
        tokens = batch["intseq"].to(self.device)
        peptide_lengths = batch["peptide_lengths"].squeeze(1).to(self.device)

        if self.target_swapped_distractor_mix is not None and not Eval:
            # This is one of the original 95%-A DINO global crops.  The shared
            # mixer then samples B and applies the unchanged subset/merge path.
            anchor_peaks, anchor_padding = self.target_swapped_global_crop(
                spectra, peak_lengths.unsqueeze(1), rand_size=False
            )[0]
            mixed_peaks, mixed_padding, _, partner_indices, has_eligible = (
                self.target_swapped_distractor_mix.mix_sampled_distractors(
                    anchor_peaks,
                    anchor_padding,
                    spectra,
                    peak_lengths,
                    mz,
                    batch["precursor_charge"].to(self.device),
                    strength=self.target_swapped_mix_strength,
                    return_provenance=True,
                )
            )
            if not bool(has_eligible.all()):
                missing = int((~has_eligible).sum().item())
                raise RuntimeError(
                    "Target-swapped mixture batch has "
                    f"{missing} anchors without an eligible in-batch distractor."
                )

            # Same M twice: precursor/label A first, then selected partner B.
            spectra = torch.cat([mixed_peaks, mixed_peaks], dim=0)
            spectrum_padding_mask = torch.cat([mixed_padding, mixed_padding], dim=0)
            precursors = torch.cat([precursors, precursors[partner_indices]], dim=0)
            tokens = torch.cat([tokens, tokens[partner_indices]], dim=0)
            peptide_lengths = torch.cat(
                [peptide_lengths, peptide_lengths[partner_indices]], dim=0
            )

        batch_size, seq_len = tokens.shape
        target_tokens = torch.full(
            (batch_size, seq_len + 1),
            self.pad_token,
            dtype=torch.long,
            device=self.device,
        )
        target_tokens[:, :seq_len] = tokens
        target_tokens[
            torch.arange(batch_size, device=self.device), peptide_lengths
        ] = self.stop_token
        return {
            "spectra": spectra,
            "precursors": precursors,
            "input_tokens": tokens,
            "target_tokens": target_tokens,
            "spectrum_padding_mask": spectrum_padding_mask,
        }, batch_size

    def forward(self, parsed_batch, **kwargs):
        del kwargs
        if self.freeze_encoder:
            # Frozen controls train the fresh decoder only while keeping the
            # pretrained encoder deterministic (including its dropout state).
            with torch.no_grad():
                encoded = self._encode_spectra(
                    parsed_batch["spectra"], parsed_batch["spectrum_padding_mask"], parsed_batch["precursors"]
                )
        else:
            encoded = self._encode_spectra(
                parsed_batch["spectra"], parsed_batch["spectrum_padding_mask"], parsed_batch["precursors"]
            )
        logits, _ = self.decoder(
            parsed_batch["input_tokens"], parsed_batch["precursors"], encoded["emb"], encoded["mask"]
        )
        return logits

    def _loss(self, logits, target, criterion):
        return criterion(logits.reshape(-1, logits.shape[-1]), target.flatten())

    # BaseDownstreamWrapper declares these hooks for its manual-optimization path.
    # This class overrides all Lightning steps below, but keeps small concrete
    # implementations so it remains a valid subclass without affecting that path.
    def _get_train_stats(self, returns, parsed_batch):
        loss = self._loss(returns, parsed_batch["target_tokens"], self.celoss)
        return loss, {"loss": loss}

    def _get_eval_stats(self, returns, parsed_batch, split=None):
        del split
        return {"loss": self._loss(returns, parsed_batch["target_tokens"], self.val_celoss)}

    def training_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch)
        loss = self._loss(self.forward(parsed_batch), parsed_batch["target_tokens"], self.celoss)
        self.log(
            f"{self.TASK_NAME}_train_loss", loss, on_step=True, on_epoch=True,
            sync_dist=True, batch_size=batch_size, add_dataloader_idx=False, prog_bar=True,
        )
        return loss

    def on_train_epoch_start(self):
        if self.freeze_encoder:
            self.encoder.eval()

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        return "".join(character if character.isalnum() else "_" for character in str(name)).strip("_")

    def _validation_mode(self, dataloader_idx: int) -> str:
        names = getattr(self.trainer.datamodule, "val_dataloader_names", ["val"])
        if dataloader_idx == 0:
            return "val"
        name = names[dataloader_idx] if dataloader_idx < len(names) else f"extra{dataloader_idx}"
        return f"val_{self._sanitize_metric_name(name)}"

    def _log_sequence_metrics(self, mode, parsed_batch, batch_size):
        beam = self.beam_search_decode(
            parsed_batch["spectra"], parsed_batch["precursors"], parsed_batch["spectrum_padding_mask"]
        )
        peptides_true = self.tokenizer.detokenize(
            parsed_batch["target_tokens"], pad_token_idx=self.pad_token, EOS_token_idx=self.stop_token, exclude_stop=True
        )
        peptides_pred = [prediction for spectrum_predictions in beam for _, _, prediction in spectrum_predictions]
        aa_precision, aa_recall, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(peptides_true, peptides_pred, self.residues)
        )
        log_args = dict(on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, add_dataloader_idx=False)
        self.log(f"{self.TASK_NAME}_{mode}_aa_prec", aa_precision, **log_args)
        self.log(f"{self.TASK_NAME}_{mode}_aa_recall", aa_recall, **log_args)
        self.log(f"{self.TASK_NAME}_{mode}_pep_prec", pep_precision, **log_args)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        parsed_batch, batch_size = self._parse_batch(batch, Eval=True)
        logits = self.forward(parsed_batch)
        loss = self._loss(logits, parsed_batch["target_tokens"], self.val_celoss)
        mode = self._validation_mode(dataloader_idx)
        self.log(
            f"{self.TASK_NAME}_{mode}_loss", loss, on_step=True, on_epoch=True,
            sync_dist=True, batch_size=batch_size, add_dataloader_idx=False, prog_bar=dataloader_idx == 0,
        )
        if self.calculate_precision and not self.cheap_val:
            self._log_sequence_metrics(mode, parsed_batch, batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch, Eval=True)
        logits = self.forward(parsed_batch)
        loss = self._loss(logits, parsed_batch["target_tokens"], self.val_celoss)
        self.log(f"{self.TASK_NAME}_test_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, add_dataloader_idx=False)
        beam = self.beam_search_decode(parsed_batch["spectra"], parsed_batch["precursors"], parsed_batch["spectrum_padding_mask"])
        peptides_true = self.tokenizer.detokenize(parsed_batch["target_tokens"], pad_token_idx=self.pad_token, EOS_token_idx=self.stop_token, exclude_stop=True)
        peptides_pred = []
        predictions = [] if self.log_predictions else None
        if self.log_predictions:
            peak_files = batch.get("peak_file", ["unknown"] * len(peptides_true))
            scan_ids = batch.get("scan_id", [-1] * len(peptides_true))
            titles = batch.get("title", ["unknown"] * len(peptides_true))
            canonical_indices = batch.get("index", [-1] * len(peptides_true))
            species = batch.get("species", [""] * len(peptides_true))
            splits = batch.get("split", [""] * len(peptides_true))
            source_row_indices = batch.get("source_row_index", [""] * len(peptides_true))
            isolation_lows = batch.get("isolation_low", [""] * len(peptides_true))
            isolation_highs = batch.get("isolation_high", [""] * len(peptides_true))
        for index, spectrum_predictions in enumerate(beam):
            metadata = None
            if self.log_predictions:
                scan_id = scan_ids[index]
                if isinstance(scan_id, torch.Tensor):
                    scan_id = int(scan_id.item())
                canonical_index = canonical_indices[index]
                if isinstance(canonical_index, torch.Tensor):
                    canonical_index = int(canonical_index.item())
                def _metadata_scalar(value):
                    return value.item() if isinstance(value, torch.Tensor) else value

                metadata = {
                    "canonical_index": canonical_index, "species": species[index],
                    "peak_file": peak_files[index], "scan_id": scan_id, "title": titles[index],
                    "split": _metadata_scalar(splits[index]), "source_row_index": _metadata_scalar(source_row_indices[index]),
                    "isolation_low": _metadata_scalar(isolation_lows[index]), "isolation_high": _metadata_scalar(isolation_highs[index]),
                    "precursor_charge": float(parsed_batch["precursors"][index, 1]),
                    "precursor_mz": float(parsed_batch["precursors"][index, 2]),
                }
            if spectrum_predictions:
                score, aa_scores, peptide = spectrum_predictions[0]
                peptide = [aa for aa in peptide if aa != "$"]
                if self.log_predictions:
                    predictions.append({**metadata, "peptide": peptide, "peptide_score": score, "aa_scores": aa_scores})
                peptides_pred.append(peptide)
            else:
                if self.log_predictions:
                    predictions.append({**metadata, "peptide": [], "peptide_score": "null", "aa_scores": "null"})
                peptides_pred.append([])
        aa_matches_batch, n_aa_true, n_aa_pred = evaluate.aa_match_batch(
            peptides_true, peptides_pred, self.residues
        )
        aa_precision, aa_recall, pep_precision = evaluate.aa_match_metrics(
            aa_matches_batch, n_aa_true, n_aa_pred
        )
        log_args = dict(on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, add_dataloader_idx=False)
        self.log(f"{self.TASK_NAME}_test_aa_prec", aa_precision, **log_args)
        self.log(f"{self.TASK_NAME}_test_aa_recall", aa_recall, **log_args)
        self.log(f"{self.TASK_NAME}_test_pep_prec", pep_precision, **log_args)
        if self.log_species_metrics:
            self._accumulate_test_species_metrics(
                batch.get("species"), aa_matches_batch, peptides_true, peptides_pred
            )
        if self.log_predictions:
            return {"predictions": predictions, "peptides_true": peptides_true}
        return None

    def on_test_epoch_start(self):
        self._test_species_counts = {}

    def _accumulate_test_species_metrics(self, species, aa_matches_batch, peptides_true, peptides_pred):
        if species is None:
            raise KeyError(
                "log_species_metrics requires a 'species' metadata column in the downstream dataset."
            )
        if len(species) != len(aa_matches_batch):
            raise ValueError("Species metadata and de novo predictions have different batch sizes.")
        for species_name, (aa_matches, peptide_correct), peptide_true, peptide_pred in zip(
            species, aa_matches_batch, peptides_true, peptides_pred, strict=True
        ):
            counts = self._test_species_counts.setdefault(str(species_name), [0, 0, 0, 0, 0])
            counts[0] += 1
            counts[1] += int(peptide_correct)
            counts[2] += int(aa_matches.sum())
            counts[3] += len(peptide_true)
            counts[4] += len(peptide_pred)

    @staticmethod
    def _species_metric_values(counts):
        n_spectra, n_peptide_correct, n_aa_correct, n_aa_true, n_aa_pred = counts
        return {
            "n_spectra": float(n_spectra),
            "pep_prec": n_peptide_correct / (n_spectra + 1e-8),
            "aa_prec": n_aa_correct / (n_aa_pred + 1e-8),
            "aa_recall": n_aa_correct / (n_aa_true + 1e-8),
        }

    def on_test_epoch_end(self):
        if not self.log_species_metrics:
            return
        gathered = [self._test_species_counts]
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * self.trainer.world_size
            dist.all_gather_object(gathered, self._test_species_counts)
        if not self.trainer.is_global_zero:
            return
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
        for rank_counts in gathered:
            for species_name, counts in rank_counts.items():
                totals[species_name] = [left + right for left, right in zip(totals[species_name], counts, strict=True)]
        for species_name, counts in sorted(totals.items()):
            prefix = f"{self.TASK_NAME}_test_species_{self._sanitize_metric_name(species_name)}"
            for suffix, value in self._species_metric_values(counts).items():
                self.log(f"{prefix}_{suffix}", value, on_step=False, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        parameters = self.decoder.parameters() if self.freeze_encoder else self.parameters()
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, self.warmup_steps, self.cosine_period_steps)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}



class SpectralQualityAssessment(BaseDownstreamWrapper):
    def __init__(
        self,
        encoder,
        classifier_head,
        global_args,
        collate_fn=None,
        task_dict=None,
        **kwargs,
    ):
        super().__init__(
            encoder,
            classifier_head,
            global_args,
            collate_fn=collate_fn,
            task_dict=task_dict,
        )
        self.TASK_NAME = "sqa"

        self.global_args = global_args
        self.freeze_encoder = global_args.freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            print("Froze encoder weights for SQA task")

        self.embedding_dir = getattr(global_args, "embedding_dir", None)
        self._embedding_cache = {
            "train": {"by_index": {}},  # {idx: feature_tensor_cpu}
            "val": {"by_index": {}},
            "test": {"by_index": {}},
        }
        self._cache_built = {"train": False, "val": False, "test": False}

        self._maybe_load_embeddings_from_disk(global_args)

        # manual collectors for AUC
        self._train_preds, self._train_targets = [], []
        self._val_preds, self._val_targets = [], []
        self._test_preds, self._test_targets = [], []

        self.auroc = BinaryAUROC()

    def configure_optimizers(self):
        """Use a conservative encoder LR while retaining a faster SQA head LR."""
        head_lr = float(self.task_dict.get("head_lr", self.lr))
        encoder_lr = float(self.task_dict.get("encoder_lr", head_lr))
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "weight_decay": self.weight_decay,
        }

        if self.freeze_encoder:
            return torch.optim.Adam(
                self.decoder.parameters(), lr=head_lr, **optimizer_kwargs
            )

        if head_lr <= 0 or encoder_lr <= 0:
            raise ValueError("SQA head_lr and encoder_lr must be positive.")

        encoder_opt = torch.optim.Adam(
            self.encoder.parameters(), lr=encoder_lr, **optimizer_kwargs
        )
        # CosineAnnealLRCallback schedules against head_lr and respects this
        # multiplier, preserving the encoder/head LR ratio throughout training.
        for group in encoder_opt.param_groups:
            group["lr_scale"] = encoder_lr / head_lr

        head_opt = torch.optim.Adam(
            self.decoder.parameters(), lr=head_lr, **optimizer_kwargs
        )
        return [encoder_opt, head_opt]

    def _get_dataset_hash(self, path: str) -> str:
        return hashlib.md5(path.encode("utf-8")).hexdigest()[:8]

    def _get_embedding_key_dir(self, global_args):
        external_cache = getattr(global_args, "external_embedding_cache", "")
        if external_cache:
            return os.path.realpath(external_cache)
        if self.embedding_dir is None:
            return None
        enc_name = global_args.encoder_model
        if global_args.encoder_weights:
            checkpoint_path = os.path.realpath(global_args.encoder_weights)
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            checkpoint_hash = hashlib.sha256(checkpoint_path.encode("utf-8")).hexdigest()[:12]
            enc_w = f"{checkpoint_name}__path{checkpoint_hash}"
        else:
            enc_w = "none"
        peaks = str(global_args.max_peaks)
        dataset_hash = self._get_dataset_hash(global_args.downstream_root_dir)
        precursor_conditioning = getattr(
            global_args, "precursor_conditioning", "conditioned"
        )
        key = (
            f"{enc_name}__{enc_w}__maxpeaks{peaks}__"
            f"prec{precursor_conditioning}__data{dataset_hash}"
        )
        return os.path.join(self.embedding_dir, key)

    def _maybe_write_dataset_meta(self, key_dir: str):
        if getattr(self.global_args, "external_embedding_cache", ""):
            return
        meta_path = os.path.join(key_dir, "dataset_path.txt")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                f.write(self.global_args.downstream_root_dir)

    def _maybe_load_embeddings_from_disk(self, global_args):
        key_dir = self._get_embedding_key_dir(global_args)
        if key_dir is None or not os.path.isdir(key_dir):
            print(
                f"No pregenerated embeddings found at {key_dir}. Will generate during first epoch."
            )
            return

        loaded_any = False
        dataset_path_str = ""
        meta_path = os.path.join(key_dir, "dataset_path.txt")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                dataset_path_str = f.read().strip()

        for split in ["train", "val", "test"]:
            path = os.path.join(key_dir, f"{split}.pt")
            if os.path.exists(path):
                payload = torch.load(path, map_location="cpu")
                idxs = payload["index"].tolist()
                feats = payload["features"]  # shape [N, D]
                by_index = {int(i): feats[j].cpu() for j, i in enumerate(idxs)}
                self._embedding_cache[split]["by_index"] = by_index
                self._cache_built[split] = True
                loaded_any = True

        if loaded_any:
            print(
                f"Found pregenerated embeddings under {key_dir} (dataset: {dataset_path_str}). Using them."
            )
        if getattr(global_args, "external_embedding_cache", ""):
            # A validation-only fit must not force materialization of the held-out
            # test split. Missing split files are rejected lazily when Lightning
            # actually enters that split through _get_cached_batch.
            required = ["train", "val"]
            missing = [split for split in required if not self._cache_built[split]]
            if missing:
                raise FileNotFoundError(
                    f"External embedding cache {key_dir} is incomplete; missing "
                    f"{', '.join(missing)}.pt required for fitting."
                )

    def _save_embeddings_to_disk(self, split: str):
        if getattr(self.global_args, "external_embedding_cache", ""):
            return
        key_dir = self._get_embedding_key_dir(self.global_args)
        if key_dir is None:
            return
        os.makedirs(key_dir, exist_ok=True)
        self._maybe_write_dataset_meta(key_dir)

        by_index = self._embedding_cache[split]["by_index"]
        if not by_index:
            return  # nothing to save

        # Save as (sorted indices, stacked features) for compactness
        idxs = sorted(by_index.keys())
        feats = torch.stack([by_index[i] for i in idxs], dim=0)
        payload = {"index": torch.tensor(idxs, dtype=torch.long), "features": feats}
        path = os.path.join(key_dir, f"{split}.pt")
        torch.save(payload, path)

    # --------- batch parsing / caching ---------

    def _parse_batch(self, batch, Eval=False):
        mzab = self._mzab_array(batch)
        parsed_batch = {
            "mz_ab": mzab,
            "mass": batch["precursor_mass"],
            "charge": batch["precursor_charge"],
            "peak_lengths": batch["peak_lengths"],
            "quality": batch["quality"],
        }
        parsed_batch["mass"], parsed_batch["charge"] = condition_precursor_inputs(
            parsed_batch["mass"],
            parsed_batch["charge"],
            getattr(self.global_args, "precursor_conditioning", "conditioned"),
        )
        # pass through dataset indices for caching (supports either key)
        if "index" in batch:
            parsed_batch["indices"] = batch["index"]
        return parsed_batch, mzab.shape[0]

    def _encode_and_cache(self, parsed_batch, split: str):
        key_padding_mask = self._get_padding_mask(
            parsed_batch["mz_ab"], parsed_batch["peak_lengths"]
        )
        feats = self.encoder(
            parsed_batch["mz_ab"],
            mass=parsed_batch["mass"],
            charge=parsed_batch["charge"],
            key_padding_mask=key_padding_mask,
        )
        # cache per-sample using dataset indices
        idxs = parsed_batch.get("indices", None)
        if idxs is not None:
            by_index = self._embedding_cache[split]["by_index"]
            for f, i in zip(feats, idxs):
                by_index[int(i)] = f.detach().cpu()
        return feats

    def _get_cached_batch(self, split: str, parsed_batch):
        idxs = parsed_batch.get("indices", None)
        assert (
            idxs is not None
        ), "Indices missing in parsed_batch; cannot fetch cached embeddings."
        idxs_list = [int(i) for i in idxs.tolist()]
        by_index = self._embedding_cache[split]["by_index"]

        # find any not-yet-cached ids
        missing = [i for i in idxs_list if i not in by_index]
        if missing:
            if getattr(self.global_args, "external_embedding_cache", ""):
                preview = ", ".join(str(value) for value in missing[:5])
                raise KeyError(
                    f"External embedding cache has no {split} features for {len(missing)} "
                    f"dataset indices (first: {preview})."
                )
            # encode whole batch once
            key_padding_mask = self._get_padding_mask(
                parsed_batch["mz_ab"], parsed_batch["peak_lengths"]
            )
            feats_full = self.encoder(
                parsed_batch["mz_ab"],
                mass=parsed_batch["mass"],
                charge=parsed_batch["charge"],
                key_padding_mask=key_padding_mask,
            )
            # stash only the missing ones
            for i, f in zip(idxs_list, feats_full):
                if i not in by_index:
                    by_index[i] = f.detach().cpu()

        feats = torch.stack([by_index[int(i)] for i in idxs.tolist()], dim=0)
        return feats.to(self.device)

    # --------- forward / training hooks ---------

    def forward(self, parsed_batch, batch_idx, **kwargs):
        if not self.freeze_encoder:
            key_padding_mask = self._get_padding_mask(
                parsed_batch["mz_ab"], parsed_batch["peak_lengths"]
            )
            features = self.encoder(
                parsed_batch["mz_ab"],
                mass=parsed_batch["mass"],
                charge=parsed_batch["charge"],
                key_padding_mask=key_padding_mask,
            )
            logits = self.decoder(features)
            return logits

        if self.trainer.training:
            split = "train"
        elif self.trainer.validating:
            split = "val"
        elif self.trainer.testing:
            split = "test"
        else:
            split = None  # sanity check path

        if split is None:
            if getattr(self.global_args, "external_embedding_cache", ""):
                # Lightning sanity validation is a validation batch even though it
                # does not set trainer.validating yet.
                features = self._get_cached_batch("val", parsed_batch)
                logits = self.decoder(features)
                return logits
            key_padding_mask = self._get_padding_mask(
                parsed_batch["mz_ab"], parsed_batch["peak_lengths"]
            )
            features = self.encoder(
                parsed_batch["mz_ab"],
                mass=parsed_batch["mass"],
                charge=parsed_batch["charge"],
                key_padding_mask=key_padding_mask,
            )
        elif self._cache_built[split]:
            features = self._get_cached_batch(split, parsed_batch)
        else:
            features = self._encode_and_cache(parsed_batch, split)

        logits = self.decoder(features)
        return logits

    def _get_train_stats(self, logits, parsed_batch):
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), parsed_batch["quality"].float()
        )
        metrics = _calc_binary_metrics(logits, parsed_batch["quality"])
        stats = {"loss": loss.detach().item(), **metrics}

        preds = torch.sigmoid(logits.squeeze(-1)).detach().cpu()
        labels = parsed_batch["quality"].int().detach().cpu()
        self._train_preds.append(preds)
        self._train_targets.append(labels)

        return loss, stats

    def _get_eval_stats(self, logits, parsed_batch, split=None):
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), parsed_batch["quality"].float()
        )
        metrics = _calc_binary_metrics(logits, parsed_batch["quality"])
        stats = {"loss": loss.detach().item(), **metrics}

        if self.trainer is None or self.trainer.sanity_checking:
            return stats

        preds = torch.sigmoid(logits.squeeze(-1)).detach().cpu()
        labels = parsed_batch["quality"].int().detach().cpu()

        if split == "val":
            self._val_preds.append(preds)
            self._val_targets.append(labels)
        elif split == "test":
            self._test_preds.append(preds)
            self._test_targets.append(labels)

        return stats

    def _mark_cache_built(self, split: str):
        if self.freeze_encoder and not self._cache_built[split]:
            self._cache_built[split] = True
            self._save_embeddings_to_disk(split)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if not self.trainer.sanity_checking:
            if self.freeze_encoder:
                self._mark_cache_built("train")

            self.auroc.reset()
            preds = torch.cat(self._train_preds)
            targets = torch.cat(self._train_targets)
            self.auroc(preds, targets)
            self.log("train_auc", self.auroc.compute(), prog_bar=True)
            self.auroc.reset()
            self._train_preds, self._train_targets = [], []

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if not self.trainer.sanity_checking:
            if self.freeze_encoder:
                self._mark_cache_built("val")
            self.auroc.reset()
            preds = torch.cat(self._val_preds)
            targets = torch.cat(self._val_targets)
            self.auroc(preds, targets)
            self.log("val_auc", self.auroc.compute(), prog_bar=True)
            self.auroc.reset()
            self._val_preds, self._val_targets = [], []

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if not self.trainer.sanity_checking:
            if self.freeze_encoder:
                self._mark_cache_built("test")
            self.auroc.reset()
            preds = torch.cat(self._test_preds)
            targets = torch.cat(self._test_targets)
            self.auroc(preds, targets)
            self.log("test_auc", self.auroc.compute(), prog_bar=True)
            self.auroc.reset()
            self._test_preds, self._test_targets = [], []


class SupervisedMetricLearning(BaseDownstreamWrapper):
    """SupCon fine-tuning over a fresh projection of pooled encoder features."""

    def __init__(
        self,
        encoder,
        metric_head,
        global_args,
        collate_fn=None,
        tokenizer=None,
        task_dict=None,
        pooler=None,
        **kwargs,
    ):
        del tokenizer, kwargs
        if pooler is None:
            raise ValueError("Metric learning requires the pretrained wrapper's global pooler.")
        super().__init__(encoder, metric_head, global_args, collate_fn=collate_fn, task_dict=task_dict)
        from src.metric_learning import MetricLearningEmbedder

        self.TASK_NAME = "metric_learning"
        self.automatic_optimization = True
        self.pooler = pooler
        self.temperature = float(task_dict.get("temperature", 0.07))
        self.encoder_precursor_conditioning = task_dict.get(
            "encoder_precursor_conditioning", global_args.precursor_conditioning
        )
        if self.encoder_precursor_conditioning not in VALID_PRECURSOR_CONDITIONING:
            raise ValueError(
                "encoder_precursor_conditioning must be one of "
                f"{sorted(VALID_PRECURSOR_CONDITIONING)}."
            )
        self.freeze_encoder = bool(global_args.freeze_encoder)
        if self.freeze_encoder:
            for module in (self.encoder, self.pooler):
                for parameter in module.parameters():
                    parameter.requires_grad = False
                module.eval()
        self._embedder_type = MetricLearningEmbedder

    def _padding_mask(self, spectra: torch.Tensor, peak_lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(spectra.shape[1], device=spectra.device).unsqueeze(0)
        return positions >= peak_lengths.reshape(-1).to(spectra.device).unsqueeze(1)

    def _parse_batch(self, batch, Eval=False):
        del Eval
        spectra = torch.stack([batch["mz_array"], batch["intensity_array"]], dim=-1)
        mass = batch["precursor_mass"] if self.encoder.use_mass else None
        charge = batch["precursor_charge"] if self.encoder.use_charge else None
        mass, charge = condition_precursor_inputs(
            mass, charge, self.encoder_precursor_conditioning
        )
        return {
            "spectra": spectra,
            "padding_mask": self._padding_mask(spectra, batch["peak_lengths"]),
            "mass": mass,
            "charge": charge,
            "labels": batch["peptide_id"],
        }, spectra.shape[0]

    def forward(self, parsed_batch, **kwargs):
        del kwargs
        encoded = self.encoder(
            parsed_batch["spectra"],
            key_padding_mask=parsed_batch["padding_mask"],
            mass=parsed_batch["mass"],
            charge=parsed_batch["charge"],
        )
        global_representation = self.pooler(encoded["emb"], encoded["mask"])
        return self.decoder(global_representation)

    # BaseDownstreamWrapper's manual-optimization hooks are not used here,
    # but concrete definitions retain the repository's wrapper contract.
    def _get_train_stats(self, returns, parsed_batch):
        return self._loss_and_stats(returns, parsed_batch["labels"])

    def _get_eval_stats(self, returns, parsed_batch, split=None):
        del split
        _, stats = self._loss_and_stats(returns, parsed_batch["labels"])
        return stats

    def _loss_and_stats(self, embeddings, labels):
        from src.metric_learning import supervised_contrastive_loss

        loss, valid_anchor_fraction = supervised_contrastive_loss(
            embeddings, labels, temperature=self.temperature
        )
        return loss, {
            "loss": loss,
            "valid_anchor_fraction": valid_anchor_fraction,
            "embedding_norm": embeddings.norm(dim=-1).mean(),
        }

    def training_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch)
        embeddings = self.forward(parsed_batch)
        loss, stats = self._loss_and_stats(embeddings, parsed_batch["labels"])
        self.log_dict(
            {f"{self.TASK_NAME}_train_{name}": value for name, value in stats.items()},
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        del batch_idx
        parsed_batch, batch_size = self._parse_batch(batch, Eval=True)
        embeddings = self.forward(parsed_batch)
        loss, stats = self._loss_and_stats(embeddings, parsed_batch["labels"])
        suffix = "val" if dataloader_idx == 0 else f"val_{dataloader_idx}"
        self.log_dict(
            {f"{self.TASK_NAME}_{suffix}_{name}_epoch": value for name, value in stats.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
            prog_bar=dataloader_idx == 0,
        )
        return loss

    def test_step(self, batch, batch_idx):
        """Traverse the held-out Lance split without resampling it.

        SupCon requires positive-aware batches, so its batch loss is not a
        meaningful exhaustive per-spectrum test statistic. The final paper
        metric remains the separately materialized retrieval/pair benchmark.
        """
        del batch_idx
        parsed_batch, batch_size = self._parse_batch(batch, Eval=True)
        embeddings = self.forward(parsed_batch)
        embedding_norm = embeddings.norm(dim=-1).mean()
        self.log(
            f"{self.TASK_NAME}_test_embedding_norm",
            embedding_norm,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return {"embedding_norm": embedding_norm}

    def configure_optimizers(self):
        head_lr = float(self.task_dict.get("head_lr", self.lr))
        encoder_lr = float(self.task_dict.get("encoder_lr", head_lr))
        if head_lr <= 0 or encoder_lr <= 0:
            raise ValueError("Metric-learning head_lr and encoder_lr must be positive.")
        groups = [{"params": self.decoder.parameters(), "lr": head_lr}]
        if not self.freeze_encoder:
            encoder_parameters = list(self.encoder.parameters()) + list(self.pooler.parameters())
            groups.append(
                {
                    "params": encoder_parameters,
                    "lr": encoder_lr,
                    "lr_scale": encoder_lr / head_lr,
                }
            )
        return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=self.weight_decay)

    def get_embedder(self, trainable: bool = False):
        if trainable:
            for module in (self.encoder, self.pooler, self.decoder):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        return self._embedder_type(self.encoder, self.pooler, self.decoder)

    def on_train_epoch_start(self) -> None:
        if self.freeze_encoder:
            self.encoder.eval()
            self.pooler.eval()

    def on_train_epoch_end(self) -> None:
        datamodule = self.trainer.datamodule
        diagnostics_fn = getattr(datamodule, "metric_learning_sampler_diagnostics", None)
        if diagnostics_fn is None:
            return
        for name, value in diagnostics_fn().items():
            if math.isfinite(value):
                self.log(
                    f"{self.TASK_NAME}_sampler_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )


class AuxiliaryBinarySpectrumAssessment(SpectralQualityAssessment):
    """SQA trainer reused for a named binary spectrum target from Lance.

    TODO: If time permits after auxiliary-task validation, consider extracting the
    shared binary-spectrum behavior above SQA so SQA is also a thin named task.
    Keep this direction for now to avoid refactoring the established SQA path.
    """

    TASK_NAME_VALUE = "auxiliary_binary"

    def __init__(self, *args, task_dict=None, **kwargs):
        if task_dict is None:
            raise ValueError("Auxiliary binary tasks require a task configuration.")
        self.target_column = str(task_dict.get("target_column", "label"))
        self.evaluation_mask_column = task_dict.get("evaluation_mask_column")
        self._evaluation_masks = {"val": [], "test": []}
        super().__init__(*args, task_dict=task_dict, **kwargs)
        self.TASK_NAME = self.TASK_NAME_VALUE

    def _parse_batch(self, batch, Eval=False):
        adapted = dict(batch)
        adapted["quality"] = batch[self.target_column]
        parsed, batch_size = super()._parse_batch(adapted, Eval=Eval)
        if self.evaluation_mask_column:
            parsed["evaluation_mask"] = batch[self.evaluation_mask_column].bool()
        return parsed, batch_size

    def _get_eval_stats(self, logits, parsed_batch, split=None):
        stats = super()._get_eval_stats(logits, parsed_batch, split=split)
        if (
            split in self._evaluation_masks
            and "evaluation_mask" in parsed_batch
            and self.trainer is not None
            and not self.trainer.sanity_checking
        ):
            self._evaluation_masks[split].append(parsed_batch["evaluation_mask"].detach().cpu())
        return stats

    def _log_masked_auc(self, split: str) -> None:
        if not self.evaluation_mask_column or not self._evaluation_masks[split]:
            return
        predictions = torch.cat(getattr(self, f"_{split}_preds"))
        targets = torch.cat(getattr(self, f"_{split}_targets"))
        mask = torch.cat(self._evaluation_masks[split]).bool()
        if not mask.any() or torch.unique(targets[mask]).numel() < 2:
            return
        metric = BinaryAUROC()
        value = metric(predictions[mask], targets[mask])
        self.log(
            f"{split}_{self.evaluation_mask_column}_auc",
            value,
            prog_bar=False,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        if self.trainer is not None and not self.trainer.sanity_checking:
            self._log_masked_auc("val")
            self._evaluation_masks["val"] = []
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        if self.trainer is not None and not self.trainer.sanity_checking:
            self._log_masked_auc("test")
            self._evaluation_masks["test"] = []
        super().on_test_epoch_end()


class ChimericityAssessment(AuxiliaryBinarySpectrumAssessment):
    """SQA-style binary chimericity assessment."""

    TASK_NAME_VALUE = "chimericity"


class OxidizedMethionineAssessment(AuxiliaryBinarySpectrumAssessment):
    """SQA-style oxidized-methionine assessment with optional matched control."""

    TASK_NAME_VALUE = "oxidized_met"


class RetentionTimeProbe(SpectralQualityAssessment):
    """Frozen linear ordinal or scalar-regression probe over the SQA embedding path."""

    def __init__(self, *args, task_dict=None, **kwargs):
        if task_dict is None:
            raise ValueError("Retention-time probing requires a task configuration.")
        global_args = kwargs.get("global_args")
        if global_args is None and len(args) >= 3:
            global_args = args[2]
        if global_args is None or not bool(global_args.freeze_encoder):
            raise ValueError("Retention-time probing is frozen-encoder only.")
        super().__init__(*args, task_dict=task_dict, **kwargs)
        from src.soft_ordinal import BinSpec

        self.TASK_NAME = "retention_time"
        self.target_column = str(task_dict.get("target_column", "nrt_aligned"))
        self.prediction_mode = str(task_dict.get("prediction_mode", "ordinal"))
        if self.prediction_mode not in {"ordinal", "regression"}:
            raise ValueError("prediction_mode must be ordinal or regression.")
        if not isinstance(self.decoder, torch.nn.Linear):
            raise TypeError("RetentionTimeProbe requires a linear RT head.")
        self.bin_spec = None
        if self.prediction_mode == "ordinal":
            ordinal = task_dict["soft_ordinal"]
            self.bin_spec = BinSpec(
                low=float(ordinal["low"]),
                high=float(ordinal["high"]),
                n_bins=int(ordinal["n_bins"]),
                sigma=float(ordinal["sigma"]),
            )
            if self.decoder.out_features != self.bin_spec.n_bins:
                raise ValueError("Ordinal RT head output width must match soft_ordinal.n_bins.")
        elif self.decoder.out_features != 1:
            raise ValueError("Regression RT head must emit one scalar per spectrum.")
        self._rt_predictions = {"train": [], "val": [], "test": []}
        self._rt_targets = {"train": [], "val": [], "test": []}

    def _parse_batch(self, batch, Eval=False):
        adapted = dict(batch)
        # The parent parses its legacy binary target; RT keeps the real scalar separately.
        adapted["quality"] = torch.zeros_like(batch[self.target_column], dtype=torch.long)
        parsed, batch_size = super()._parse_batch(adapted, Eval=Eval)
        parsed["target"] = batch[self.target_column].float()
        return parsed, batch_size

    def _loss_and_prediction(self, logits, targets):
        if self.prediction_mode == "regression":
            prediction = logits.squeeze(-1)
            return F.mse_loss(prediction, targets), prediction

        from src.soft_ordinal import decode_expectation, soft_cross_entropy, soft_labels

        target_distribution = soft_labels(targets, self.bin_spec)
        return (
            soft_cross_entropy(logits, target_distribution),
            decode_expectation(logits, self.bin_spec),
        )

    def _record(self, split: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        self._rt_predictions[split].append(prediction.detach().float().cpu())
        self._rt_targets[split].append(target.detach().float().cpu())

    def _get_train_stats(self, logits, parsed_batch):
        loss, prediction = self._loss_and_prediction(logits, parsed_batch["target"])
        self._record("train", prediction, parsed_batch["target"])
        return loss, {"loss": loss}

    def _get_eval_stats(self, logits, parsed_batch, split=None):
        loss, prediction = self._loss_and_prediction(logits, parsed_batch["target"])
        if split is not None:
            self._record(split, prediction, parsed_batch["target"])
        return {"loss": loss}

    @staticmethod
    def _rank(values: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.argsort(values)).float()

    @staticmethod
    def _gather_epoch_values(values: list[torch.Tensor]) -> torch.Tensor:
        """Gather variable-size CPU batches before computing non-additive metrics."""
        local = torch.cat(values)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local
        gathered: list[object] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local.numpy())
        return torch.cat([torch.as_tensor(value, dtype=local.dtype) for value in gathered])

    def _log_rt_metrics(self, split: str) -> None:
        if not self._rt_predictions[split]:
            return
        prediction = self._gather_epoch_values(self._rt_predictions[split])
        target = self._gather_epoch_values(self._rt_targets[split])
        error = (prediction - target).abs()
        prediction_centered = prediction - prediction.mean()
        target_centered = target - target.mean()
        denominator = prediction_centered.norm() * target_centered.norm()
        pearson = (
            torch.tensor(float("nan"))
            if float(denominator) == 0.0
            else (prediction_centered * target_centered).sum() / denominator
        )
        residual_sum_squares = (prediction - target).square().sum()
        total_sum_squares = target_centered.square().sum()
        r2 = (
            torch.tensor(float("nan"))
            if float(total_sum_squares) == 0.0
            else 1.0 - residual_sum_squares / total_sum_squares
        )
        prediction_rank = self._rank(prediction)
        target_rank = self._rank(target)
        rank_prediction_centered = prediction_rank - prediction_rank.mean()
        rank_target_centered = target_rank - target_rank.mean()
        rank_denominator = rank_prediction_centered.norm() * rank_target_centered.norm()
        spearman = (
            torch.tensor(float("nan"))
            if float(rank_denominator) == 0.0
            else (rank_prediction_centered * rank_target_centered).sum() / rank_denominator
        )
        for name, value in {
            "mae": error.mean(),
            "delta_t95": torch.quantile(error, 0.95),
            "pearson": pearson,
            "spearman": spearman,
            "r2": r2,
        }.items():
            self.log(
                f"{self.TASK_NAME}_{split}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self._rt_predictions[split] = []
        self._rt_targets[split] = []

    def on_train_epoch_end(self):
        BaseDownstreamWrapper.on_train_epoch_end(self)
        if self.trainer is not None and not self.trainer.sanity_checking:
            self._mark_cache_built("train")
            self._log_rt_metrics("train")

    def on_validation_epoch_end(self):
        BaseDownstreamWrapper.on_validation_epoch_end(self)
        if self.trainer is not None and not self.trainer.sanity_checking:
            self._mark_cache_built("val")
            self._log_rt_metrics("val")

    def on_test_epoch_end(self):
        if self.trainer is not None and not self.trainer.sanity_checking:
            self._mark_cache_built("test")
            self._log_rt_metrics("test")
