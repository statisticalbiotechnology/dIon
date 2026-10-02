"""dIon pretraining wrappers.

The self-distillation machinery is adapted from DINO/DINOv2, and the optional
Gram refinement follows DINOv3:
- https://github.com/facebookresearch/dino
- https://github.com/facebookresearch/dinov2
- https://github.com/facebookresearch/dinov3
"""

from copy import deepcopy
from pathlib import Path
import math
import pickle
import numpy as np
from numpy.core.multiarray import scalar as numpy_scalar
import torch
import torch.distributed as dist
from src.data_augmentation import (
    BatchedIntensityWeightedSelectionAugmentation,
    BatchedRandomSelectionAugmentation,
    IntensityWeightedSelectionAugmentation,
    RandomSelectionAugmentation,
    RandomWindowAugmentation,
    StudentDistractorMixAugmentation,
)
from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation
from src.models.dino.embedder import DINOEmbedder
from src.wrappers.base_wrapper import BasePLWrapper
import torch.nn.functional as F
import torch.nn as nn
from src.models.dino import (
    DINOHead,
    DINOLoss,
    DINOPatchLoss,
    DINOv2MultiCropWrapper,
    KoLeoLoss,
    MultiCropWrapper,
    cosine_scheduler,
)
from src.models.dino.gram import per_spectrum_gram_mse


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _compile_model_enabled(task_dict: dict) -> bool:
    return _as_bool(task_dict.get("compile_model", False))


def _fixed_crop_length(max_peaks: int, scale: float) -> int:
    return max(1, min(max_peaks, int(math.ceil(max_peaks * float(scale)))))


def _augmentation_kwargs(task_dict: dict, global_args) -> dict:
    kwargs = dict(
        global_crops_scale=task_dict["global_crops_scale"],
        local_crops_scale=task_dict["local_crops_scale"],
        num_global_crops=task_dict["num_global_crops"],
        num_local_crops=task_dict["num_local_crops"],
        padding_value=0,
    )
    if _compile_model_enabled(task_dict) and _as_bool(
        task_dict.get("compile_fixed_crop_padding", True)
    ):
        max_peaks = int(
            task_dict.get(
                "fixed_crop_padding_max_peaks",
                getattr(global_args, "max_peaks", 0),
            )
        )
        if max_peaks < 1:
            raise ValueError("fixed_crop_padding_max_peaks must be positive.")
        kwargs.update(
            fixed_global_padded_length=_fixed_crop_length(
                max_peaks,
                task_dict["global_crops_scale"][1],
            ),
            fixed_local_padded_length=_fixed_crop_length(
                max_peaks,
                task_dict["local_crops_scale"][1],
            ),
        )
    return kwargs


def _build_dino_augmentation(task_dict: dict, global_args):
    kwargs = _augmentation_kwargs(task_dict, global_args)
    mode = task_dict["selection_mode"]
    if mode == "window":
        return RandomWindowAugmentation(**kwargs)
    if mode == "random":
        return RandomSelectionAugmentation(**kwargs)
    if mode == "random_batched":
        return BatchedRandomSelectionAugmentation(**kwargs)
    if mode == "random_intensity_weighted":
        return IntensityWeightedSelectionAugmentation(
            **kwargs,
            intensity_alpha=task_dict["intensity_alpha"],
            intensity_eps=task_dict["intensity_eps"],
        )
    if mode == "random_intensity_weighted_batched":
        return BatchedIntensityWeightedSelectionAugmentation(
            **kwargs,
            intensity_alpha=task_dict["intensity_alpha"],
            intensity_eps=task_dict["intensity_eps"],
        )
    raise ValueError("Invalid 'selection_mode' for crops")


def _compile_kwargs(task_dict: dict) -> dict:
    kwargs = {}
    mode = task_dict.get("compile_mode", "default")
    if mode not in {None, "", "none", "None"}:
        kwargs["mode"] = mode
    backend = task_dict.get("compile_backend", None)
    if backend not in {None, "", "none", "None"}:
        kwargs["backend"] = backend
    if "compile_fullgraph" in task_dict:
        kwargs["fullgraph"] = _as_bool(task_dict["compile_fullgraph"])
    if task_dict.get("compile_dynamic", None) is not None:
        kwargs["dynamic"] = _as_bool(task_dict["compile_dynamic"])
    return kwargs


def _compile_module_forward(module: nn.Module | None, task_dict: dict, label: str) -> None:
    if module is None or getattr(module, "_dion_forward_compiled", False):
        return
    if not hasattr(torch, "compile"):
        print(f"torch.compile requested for {label}, but this PyTorch has no torch.compile")
        return
    kwargs = _compile_kwargs(task_dict)
    object.__setattr__(module, "forward", torch.compile(module.forward, **kwargs))
    module._dion_forward_compiled = True
    print(f"torch.compile enabled for {label} with {kwargs}")


def _compile_strategy(task_dict: dict) -> str:
    return str(task_dict.get("compile_strategy", "student_submodules")).strip().lower()


def _compile_whole_module(module: nn.Module, task_dict: dict, label: str) -> nn.Module:
    if getattr(module, "_dion_module_compiled", False):
        return module
    if not hasattr(torch, "compile"):
        print(f"torch.compile requested for {label}, but this PyTorch has no torch.compile")
        return module
    kwargs = _compile_kwargs(task_dict)
    compiled = torch.compile(module, **kwargs)
    compiled._dion_module_compiled = True
    print(f"torch.compile enabled for {label} with {kwargs}")
    return compiled


def _maybe_compile_multicrop_module(
    module: nn.Module,
    task_dict: dict,
    label: str,
    parts: tuple[str, ...] = ("backbone", "pooler", "head", "dino_head", "ibot_head"),
) -> None:
    for part in parts:
        _compile_module_forward(getattr(module, part, None), task_dict, f"{label}.{part}")


def _maybe_compile_dino_models(
    student: nn.Module,
    teacher: nn.Module,
    task_dict: dict,
) -> tuple[nn.Module, nn.Module]:
    if not _compile_model_enabled(task_dict):
        return student, teacher

    strategy = _compile_strategy(task_dict)
    if strategy in {"", "none", "off"}:
        return student, teacher
    if strategy == "student_submodules":
        _maybe_compile_multicrop_module(student, task_dict, "student")
        return student, teacher
    if strategy == "submodules":
        _maybe_compile_multicrop_module(student, task_dict, "student")
        _maybe_compile_multicrop_module(teacher, task_dict, "teacher")
        return student, teacher
    if strategy == "student_backbone":
        _maybe_compile_multicrop_module(
            student,
            task_dict,
            "student",
            parts=("backbone",),
        )
        return student, teacher
    if strategy == "backbone":
        _maybe_compile_multicrop_module(
            student,
            task_dict,
            "student",
            parts=("backbone",),
        )
        _maybe_compile_multicrop_module(
            teacher,
            task_dict,
            "teacher",
            parts=("backbone",),
        )
        return student, teacher
    if strategy == "student":
        return _compile_whole_module(student, task_dict, "student"), teacher
    if strategy == "student_teacher":
        return (
            _compile_whole_module(student, task_dict, "student"),
            _compile_whole_module(teacher, task_dict, "teacher"),
        )
    raise ValueError(
        "compile_strategy must be one of: none, student_submodules, submodules, "
        "student_backbone, backbone, student, student_teacher."
    )


def _get_niter_per_epoch(trainer) -> int:
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is not None and hasattr(datamodule, "num_train_batches"):
        return int(datamodule.num_train_batches())

    train_dataloader = getattr(trainer, "train_dataloader", None)
    if train_dataloader is not None:
        try:
            return len(train_dataloader)
        except TypeError:
            pass

    if datamodule is None:
        raise RuntimeError("Cannot infer train dataloader length without a datamodule.")
    return len(datamodule.train_dataloader())




class _DionPretrainBase(BasePLWrapper):
    def __init__(
        self,
        encoder,
        global_args,
        collate_fn=None,
        task_dict=None,
    ):
        super().__init__(
            encoder, global_args, collate_fn=collate_fn, task_dict=task_dict
        )

        self.TASK_NAME = "dion"

        self.aug = _build_dino_augmentation(task_dict, global_args)

        _head_kwargs = dict(
            in_dim=encoder.running_units,
            out_dim=task_dict["mlp_out_dim"],
            use_bn=task_dict["mlp_use_bn"],
            norm_last_layer=task_dict["mlp_norm_last_layer"],
            nlayers=task_dict["mlp_nlayers"],
            hidden_dim=task_dict["mlp_hidden_dim"],
            bottleneck_dim=task_dict["mlp_bottleneck_dim"],
        )

        self.student = MultiCropWrapper(
            encoder,
            DINOHead(**_head_kwargs),
            pooling=task_dict["pooling"],
        )

        self.teacher = MultiCropWrapper(
            deepcopy(encoder),
            DINOHead(**_head_kwargs),
            pooling=task_dict["pooling"],
        )

        del self.encoder

        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.dino_loss = DINOLoss(
            out_dim=task_dict["mlp_out_dim"],
            num_crops_tot=task_dict["num_global_crops"] + task_dict["num_local_crops"],
            num_global_crops=task_dict["num_global_crops"],
            warmup_teacher_temp=task_dict["warmup_teacher_temp"],
            teacher_temp=task_dict["teacher_temp"],
            warmup_teacher_temp_epochs=task_dict["warmup_teacher_temp_epochs"],
            nepochs=task_dict["epochs"],
            student_temp=task_dict["student_temp"],
            center_momentum=task_dict["center_momentum"],
        )

        self.rand_window_size = task_dict["rand_window_size"]
        self.schedule_mode = task_dict.get("schedule_mode", "cosine")
        if self.schedule_mode not in {"cosine", "indefinite"}:
            raise ValueError(
                "schedule_mode must be either 'cosine' or 'indefinite', "
                f"got {self.schedule_mode!r}."
            )
        self.teacher_momentum_start = task_dict["teacher_momentum_start"]
        self.teacher_momentum_end = task_dict.get(
            "teacher_momentum_end", self.teacher_momentum_start
        )
        self.teacher_momentum_cap = task_dict.get("teacher_momentum_cap")
        if self.teacher_momentum_cap is not None:
            self.teacher_momentum_cap = float(self.teacher_momentum_cap)
            if not (
                self.teacher_momentum_start
                <= self.teacher_momentum_cap
                <= self.teacher_momentum_end
            ):
                raise ValueError(
                    "teacher_momentum_cap must be within the configured teacher momentum range."
                )
        self.num_global_crops = task_dict["num_global_crops"]
        self.num_local_crops = task_dict["num_local_crops"]

        self.mix_aug = None
        self.mix_strength_start = float(task_dict.get("mix_strength_start", 0.0))
        self.mix_strength_end = float(task_dict.get("mix_strength_end", 0.0))
        self.mix_warmup_duration = int(task_dict.get("mix_warmup_duration", 0))
        if task_dict.get("mix_distractor_enabled", False):
            self.mix_aug = StudentDistractorMixAugmentation(
                mix_apply_to=task_dict.get("mix_apply_to", "all"),
                intensity_alpha=task_dict["intensity_alpha"],
                intensity_eps=task_dict["intensity_eps"],
                merge_tol=task_dict.get("mix_merge_tol", 0.001),
                padding_value=0,
            )

        self.use_mass = self.teacher.backbone.use_mass
        self.use_charge = self.teacher.backbone.use_charge
        self.student, self.teacher = _maybe_compile_dino_models(
            self.student,
            self.teacher,
            task_dict,
        )

    def on_fit_start(self):
        ret = super().on_fit_start()
        if self.schedule_mode == "cosine":
            niter_per_ep = _get_niter_per_epoch(self.trainer)
            self.momentum_schedule = cosine_scheduler(
                base_value=self.teacher_momentum_start,
                final_value=self.teacher_momentum_end,
                epochs=self.trainer.max_epochs,
                niter_per_ep=niter_per_ep,
            )
        else:
            self.momentum_schedule = None
        return ret

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        self.update_teacher(self._current_teacher_momentum())
        return result

    def _current_teacher_momentum(self):
        if self.schedule_mode == "indefinite":
            momentum = float(self.teacher_momentum_start)
        else:
            step = min(self.trainer.global_step, len(self.momentum_schedule) - 1)
            momentum = float(self.momentum_schedule[step])
        cap = getattr(self, "teacher_momentum_cap", None)
        return min(momentum, cap) if cap is not None else momentum

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        optimizer = self.trainer.optimizers[0]
        self.log(
            "schedule/lr",
            float(optimizer.param_groups[0]["lr"]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "schedule/weight_decay",
            float(optimizer.param_groups[0]["weight_decay"]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "schedule/teacher_momentum",
            self._current_teacher_momentum(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if hasattr(self, "_current_teacher_temp"):
            self.log(
                "schedule/teacher_temperature",
                float(self._current_teacher_temp()),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    @torch.no_grad()
    def update_teacher(self, momentum):
        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    def _current_mix_strength(self):
        if self.mix_aug is None:
            return 0.0

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return self.mix_strength_start

        num_optimizers = max(1, len(getattr(trainer, "optimizers", [])))
        current_step = trainer.global_step // num_optimizers

        if self.mix_warmup_duration <= 0:
            return self.mix_strength_end

        factor = min(current_step / self.mix_warmup_duration, 1.0)
        return self.mix_strength_start + factor * (
            self.mix_strength_end - self.mix_strength_start
        )

    def _parse_batch(self, batch, Eval=None):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        lengths = spectra["peak_lengths"]

        # num_crop_tot x batch_size x L x 2
        crops = self.aug(mzab, lengths, self.rand_window_size)
        mix_strength = 0.0
        student_crops = crops
        if (not Eval) and (self.mix_aug is not None):
            mix_strength = self._current_mix_strength()
            student_crops = self.mix_aug(
                crops,
                spectra=mzab,
                lengths=lengths,
                strength=mix_strength,
                num_global_crops=self.num_global_crops,
            )
        batch_size = mzab.shape[0]

        num_crops_tot = len(student_crops)
        # repeat => num_crops_tot x batch_size
        mass = spectra["precursor_mass"].unsqueeze(0).repeat((num_crops_tot, 1))
        _charge = spectra["precursor_charge"].clamp(0, self.max_charge)
        charge = _charge.unsqueeze(0).repeat((num_crops_tot, 1))

        parsed_batch = {
            "mass": mass,
            "charge": charge,
            "student_crops": student_crops,
            "teacher_crops": crops[: self.num_global_crops],
            "mix_strength": mix_strength,
        }
        return parsed_batch, batch_size

    def forward(self, parsed_batch, **kwargs):
        student_out = self.student(
            parsed_batch["student_crops"],
            mass=parsed_batch["mass"] if self.use_mass else None,
            charge=parsed_batch["charge"] if self.use_charge else None,
        )
        with torch.no_grad():
            crops_t = parsed_batch["teacher_crops"]
            mass_t = (
                parsed_batch["mass"][: self.num_global_crops]
                if self.use_mass
                else None
            )
            charge_t = (
                parsed_batch["charge"][: self.num_global_crops]
                if self.use_charge
                else None
            )
            # get teacher out
            teacher_out = self.teacher(crops_t, mass=mass_t, charge=charge_t)
        return student_out, teacher_out

    def _get_losses(self, student_out, teacher_out):
        loss = self.dino_loss(
            student_out,
            teacher_out,
            epoch=self.trainer.current_epoch,  # TODO/FIXME: probably change anneal by step instead
        )
        if not torch.all(torch.isfinite(loss)):
            print("Loss is NaN")
            raise RuntimeError("Loss is NaN")
        return loss

    def _get_train_stats(self, returns, parsed_batch, **kwargs):
        student_out, teacher_out = returns
        loss = self._get_losses(student_out, teacher_out)
        stats = {"loss": loss, "mix_strength": parsed_batch.get("mix_strength", 0.0)}
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch, **kwargs):
        student_out, teacher_out = returns
        loss = self._get_losses(student_out, teacher_out)
        stats = {"loss": loss}
        diag = self._dino_diagnostics(
            student_out, teacher_out, self.trainer.current_epoch
        )
        self.log_dict(
            {**diag},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return stats

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.student.parameters(),
            betas=(0.9, 0.9999),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _dino_diagnostics(
        self, student_out: torch.Tensor, teacher_out: torch.Tensor, epoch: int
    ):
        """
        Return collapse/lag diagnostics over the *global* crops only,
        so student_out[:Ng, :] and teacher_out both have shape [Ng, D].
        """
        # teacher_out: [Ng, D], student_out: [Nc*batch, D]
        Ng = teacher_out.shape[0]
        student_part = student_out[:Ng]  # pick the global‐crop portion

        # temperatures
        temp_s = self.dino_loss.student_temp
        if hasattr(self, "_current_teacher_temp"):
            temp_t = float(self._current_teacher_temp())
        else:
            temp_t = float(self.dino_loss.teacher_temp_schedule[epoch])

        # center has shape [D]
        centered_teacher = (teacher_out - self.dino_loss.center) / temp_t
        P_t = F.softmax(centered_teacher, dim=-1)
        P_s = F.softmax(student_part / temp_s, dim=-1)

        # H(T)
        H_t_bits = (-P_t * P_t.log()).sum(-1).mean() / math.log(2)
        # KL divergence
        KL_ts = F.kl_div(P_s.log(), P_t, reduction="batchmean")
        # variance‐of‐max prob and norm‐of‐center
        var_max = P_s.var(dim=0).max()
        c_norm = self.dino_loss.center.norm()

        return {
            "diag/H_t_bits": H_t_bits,
            "diag/KL": KL_ts,
            "diag/var_max": var_max,
            "diag/center_norm": c_norm,
            "diag/temp_teacher": temp_t,
            "diag/temp_student": temp_s,
        }

    def get_encoder(
        self, trainable: bool = True
    ):
        """Return the EMA teacher backbone for downstream use."""
        encoder = self.teacher.backbone
        if trainable:
            for parameter in encoder.parameters():
                parameter.requires_grad = True
        return encoder

    def get_embedder(
        self, trainable: bool = True, embedding_readout: str = "backbone"
    ):
        """Return an EMA-teacher DINO embedding readout for downstream use."""
        return DINOEmbedder(
            self,
            trainable=trainable,
            embedding_readout=embedding_readout,
        )


class dIonPretrainWrapper(_DionPretrainBase):
    """
    dIon spectrum adaptation of official DINOv2 ``SSLMetaArch``.

    Mirrors the training structure in
    ``dinov2/train/ssl_meta_arch.py`` in
    https://github.com/facebookresearch/dinov2:
    clean teacher global crops provide DINO class-token and iBOT token targets;
    student global crops are randomly masked for iBOT; student local crops are
    unmasked and contribute only to the global DINO loss. Differences are
    intentional where image-specific code has no spectrum analogue: dIon uses
    peak-subset crops instead of image transforms and random peak masks instead
    of image block masks. Masked peaks are replaced by a learned
    embedding-space token, matching official
    ``vision_transformer.prepare_tokens_with_masks``.
    """

    def __init__(
        self,
        encoder,
        global_args,
        collate_fn=None,
        task_dict=None,
    ):
        BasePLWrapper.__init__(
            self, encoder, global_args, collate_fn=collate_fn, task_dict=task_dict
        )

        self.TASK_NAME = "dion"

        self.aug = _build_dino_augmentation(task_dict, global_args)

        dino_head_kwargs = dict(
            in_dim=encoder.running_units,
            out_dim=task_dict["mlp_out_dim"],
            use_bn=task_dict["mlp_use_bn"],
            norm_last_layer=task_dict["mlp_norm_last_layer"],
            nlayers=task_dict["mlp_nlayers"],
            hidden_dim=task_dict["mlp_hidden_dim"],
            bottleneck_dim=task_dict["mlp_bottleneck_dim"],
        )
        self.dino_loss_weight = float(task_dict.get("dino_loss_weight", 1.0))
        self.ibot_loss_weight = float(task_dict.get("ibot_loss_weight", 1.0))
        self.koleo_loss_weight = float(task_dict.get("koleo_loss_weight", 0.0))
        self.use_ibot = self.ibot_loss_weight > 0.0

        ibot_head_kwargs = None
        if self.use_ibot:
            ibot_head_kwargs = dict(
                in_dim=encoder.running_units,
                out_dim=task_dict.get("ibot_mlp_out_dim", task_dict["mlp_out_dim"]),
                use_bn=task_dict.get("ibot_mlp_use_bn", task_dict["mlp_use_bn"]),
                norm_last_layer=task_dict.get(
                    "ibot_mlp_norm_last_layer", task_dict["mlp_norm_last_layer"]
                ),
                nlayers=task_dict.get("ibot_mlp_nlayers", task_dict["mlp_nlayers"]),
                hidden_dim=task_dict.get(
                    "ibot_mlp_hidden_dim", task_dict["mlp_hidden_dim"]
                ),
                bottleneck_dim=task_dict.get(
                    "ibot_mlp_bottleneck_dim", task_dict["mlp_bottleneck_dim"]
                ),
            )

        self.ibot_rank_embedding_max_peaks = None
        if self.use_ibot:
            self.ibot_rank_embedding_max_peaks = int(
                task_dict.get("ibot_rank_embedding_max_peaks", global_args.max_peaks)
            )
            if self.ibot_rank_embedding_max_peaks < 1:
                raise ValueError("ibot_rank_embedding_max_peaks must be positive.")

        self.student = DINOv2MultiCropWrapper(
            encoder,
            DINOHead(**dino_head_kwargs),
            DINOHead(**ibot_head_kwargs) if self.use_ibot else None,
            pooling=task_dict["pooling"],
            ibot_rank_embedding_max_peaks=self.ibot_rank_embedding_max_peaks,
        )
        self.teacher = DINOv2MultiCropWrapper(
            deepcopy(encoder),
            DINOHead(**dino_head_kwargs),
            DINOHead(**ibot_head_kwargs) if self.use_ibot else None,
            pooling=task_dict["pooling"],
            ibot_rank_embedding_max_peaks=self.ibot_rank_embedding_max_peaks,
        )

        del self.encoder

        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False

        # This entire refinement path is opt-in. Existing DINOv2 configurations
        # do not define gram_enabled and therefore construct neither another
        # backbone nor an extra forward pass.
        self.gram_enabled = _as_bool(task_dict.get("gram_enabled", False))
        self.gram_teacher = None
        if self.gram_enabled:
            gram_teacher_checkpoint = task_dict.get("gram_teacher_checkpoint")
            if not gram_teacher_checkpoint:
                raise ValueError(
                    "gram_enabled requires gram_teacher_checkpoint: an earlier "
                    "dense-good checkpoint distinct from the refinement checkpoint."
                )
            refinement_checkpoint = getattr(global_args, "encoder_weights", None)
            if refinement_checkpoint and Path(gram_teacher_checkpoint).resolve() == Path(
                refinement_checkpoint
            ).resolve():
                raise ValueError(
                    "gram_teacher_checkpoint must differ from encoder_weights: "
                    "Gram needs an earlier dense-good teacher and a separate late "
                    "refinement initialization checkpoint."
                )
            refinement_mode = str(
                task_dict.get("gram_refinement_checkpoint_mode", "")
            ).strip().lower()
            if refinement_mode not in {"resume", "weights_only"}:
                raise ValueError(
                    "gram_enabled requires gram_refinement_checkpoint_mode to be "
                    "either 'resume' (preserve optimizer/scheduler state) or "
                    "'weights_only' (start a deliberate new refinement optimizer)."
                )
            trainer_resume = _as_bool(getattr(global_args, "resume", False))
            if trainer_resume != (refinement_mode == "resume"):
                raise ValueError(
                    "gram_refinement_checkpoint_mode conflicts with --resume: "
                    f"mode={refinement_mode!r}, resume={trainer_resume}."
                )
            if self.use_ibot:
                raise ValueError(
                    "The first Gram refinement mode is intentionally DINO-only; "
                    "set ibot_loss_weight: 0.0."
                )
            self.gram_loss_weight = float(task_dict["gram_loss_weight"])
            if self.gram_loss_weight < 0.0:
                raise ValueError("gram_loss_weight must be non-negative.")
            self.gram_warmup_steps = int(task_dict.get("gram_warmup_steps", 0))
            if self.gram_warmup_steps < 0:
                raise ValueError("gram_warmup_steps must be non-negative.")
            self.gram_warmup_schedule = str(
                task_dict.get("gram_warmup_schedule", "linear")
            ).strip().lower()
            if self.gram_warmup_schedule not in {"linear", "cosine"}:
                raise ValueError(
                    "gram_warmup_schedule must be either 'linear' or 'cosine'."
                )
            self.gram_clean_crop_index = int(task_dict.get("gram_clean_crop_index", 0))
            self.gram_teacher_refresh = _as_bool(
                task_dict.get("gram_teacher_refresh", False)
            )
            self.gram_teacher_refresh_first_step = int(
                task_dict.get("gram_teacher_refresh_first_step", 0)
            )
            self.gram_teacher_refresh_every_n_steps = int(
                task_dict.get("gram_teacher_refresh_every_n_steps", 0)
            )
            self.gram_teacher_refresh_max_updates = int(
                task_dict.get("gram_teacher_refresh_max_updates", 0)
            )
            if self.gram_teacher_refresh:
                if self.gram_teacher_refresh_first_step < 1:
                    raise ValueError(
                        "gram_teacher_refresh_first_step must be at least 1 when "
                        "gram_teacher_refresh is enabled."
                    )
                if self.gram_teacher_refresh_every_n_steps < 1:
                    raise ValueError(
                        "gram_teacher_refresh_every_n_steps must be at least 1 when "
                        "gram_teacher_refresh is enabled."
                    )
                if self.gram_teacher_refresh_max_updates < 1:
                    raise ValueError(
                        "gram_teacher_refresh_max_updates must be at least 1 when "
                        "gram_teacher_refresh is enabled."
                    )
            self.gram_teacher = deepcopy(encoder)
            self._load_gram_teacher_backbone(gram_teacher_checkpoint)
            for param in self.gram_teacher.parameters():
                param.requires_grad = False
            self.gram_teacher.eval()
            # Stored in refinement checkpoints so a resumed refinement retains
            # its relative warmup rather than restarting it on every allocation.
            self.register_buffer(
                "gram_refinement_start_step", torch.tensor(-1, dtype=torch.long)
            )
            # This starts when refresh is enabled, rather than at the original
            # refinement start. A fixed-teacher pilot can therefore be resumed
            # into a later refresh phase without an immediate catch-up copy.
            self.register_buffer(
                "gram_teacher_refresh_start_step", torch.tensor(-1, dtype=torch.long)
            )
            self.register_buffer(
                "gram_teacher_refresh_count", torch.tensor(0, dtype=torch.long)
            )

        centering = task_dict.get("centering", "centering")
        sinkhorn_iterations = task_dict.get("sinkhorn_iterations", 3)
        self.warmup_teacher_temp = float(task_dict["warmup_teacher_temp"])
        self.teacher_temp = float(task_dict["teacher_temp"])
        self.warmup_teacher_temp_epochs = int(task_dict["warmup_teacher_temp_epochs"])
        self.ibot_warmup_teacher_temp = float(
            task_dict.get("ibot_warmup_teacher_temp", task_dict["warmup_teacher_temp"])
        )
        self.ibot_teacher_temp = float(
            task_dict.get("ibot_teacher_temp", task_dict["teacher_temp"])
        )
        self.ibot_warmup_teacher_temp_epochs = int(
            task_dict.get(
                "ibot_warmup_teacher_temp_epochs",
                task_dict["warmup_teacher_temp_epochs"],
            )
        )
        self.teacher_temp_step_schedule = None
        self.ibot_teacher_temp_step_schedule = None

        self.dino_loss = DINOLoss(
            out_dim=task_dict["mlp_out_dim"],
            num_crops_tot=task_dict["num_global_crops"] + task_dict["num_local_crops"],
            num_global_crops=task_dict["num_global_crops"],
            warmup_teacher_temp=self.warmup_teacher_temp,
            teacher_temp=self.teacher_temp,
            warmup_teacher_temp_epochs=self.warmup_teacher_temp_epochs,
            nepochs=task_dict["epochs"],
            student_temp=task_dict["student_temp"],
            center_momentum=task_dict["center_momentum"],
            centering=centering,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        if self.use_ibot:
            self.ibot_loss = DINOPatchLoss(
                out_dim=ibot_head_kwargs["out_dim"],
                warmup_teacher_temp=self.ibot_warmup_teacher_temp,
                teacher_temp=self.ibot_teacher_temp,
                warmup_teacher_temp_epochs=self.ibot_warmup_teacher_temp_epochs,
                nepochs=task_dict["epochs"],
                student_temp=task_dict.get("ibot_student_temp", task_dict["student_temp"]),
                center_momentum=task_dict.get(
                    "ibot_center_momentum", task_dict["center_momentum"]
                ),
                centering=centering,
                sinkhorn_iterations=sinkhorn_iterations,
            )
        else:
            self.ibot_loss = None
        self.koleo_loss = KoLeoLoss()
        mask_ratio_min_max = task_dict.get("ibot_mask_ratio_min_max", None)
        if mask_ratio_min_max is None:
            fixed_mask_ratio = float(task_dict.get("ibot_mask_ratio", 0.4))
            mask_ratio_min_max = [fixed_mask_ratio, fixed_mask_ratio]
        self.ibot_mask_ratio_min = float(mask_ratio_min_max[0])
        self.ibot_mask_ratio_max = float(mask_ratio_min_max[1])
        self.ibot_mask_sample_probability = float(
            task_dict.get("ibot_mask_sample_probability", 1.0)
        )
        if not 0.0 <= self.ibot_mask_ratio_min <= self.ibot_mask_ratio_max <= 1.0:
            raise ValueError(
                "ibot_mask_ratio_min_max must be ordered values in [0, 1]."
            )
        if not 0.0 <= self.ibot_mask_sample_probability <= 1.0:
            raise ValueError("ibot_mask_sample_probability must be in [0, 1].")
        self.rand_window_size = task_dict["rand_window_size"]
        self.schedule_mode = task_dict.get("schedule_mode", "cosine")
        if self.schedule_mode not in {"cosine", "indefinite"}:
            raise ValueError(
                "schedule_mode must be either 'cosine' or 'indefinite', "
                f"got {self.schedule_mode!r}."
            )
        self.teacher_momentum_start = task_dict["teacher_momentum_start"]
        self.teacher_momentum_end = task_dict.get(
            "teacher_momentum_end", self.teacher_momentum_start
        )
        self.teacher_momentum_cap = task_dict.get("teacher_momentum_cap")
        if self.teacher_momentum_cap is not None:
            self.teacher_momentum_cap = float(self.teacher_momentum_cap)
            if not (
                self.teacher_momentum_start
                <= self.teacher_momentum_cap
                <= self.teacher_momentum_end
            ):
                raise ValueError(
                    "teacher_momentum_cap must be within the configured teacher momentum range."
                )
        self.num_global_crops = task_dict["num_global_crops"]
        self.num_local_crops = task_dict["num_local_crops"]

        self.mix_aug = None
        self.mix_strength_start = float(task_dict.get("mix_strength_start", 0.0))
        self.mix_strength_end = float(task_dict.get("mix_strength_end", 0.0))
        self.mix_warmup_duration = int(task_dict.get("mix_warmup_duration", 0))
        if task_dict.get("mix_distractor_enabled", False):
            mix_apply_to = task_dict.get("mix_apply_to", "global_only")
            self.mix_aug = BatchedStudentDistractorMixAugmentation(
                mix_apply_to=mix_apply_to,
                distractor_sampling=task_dict.get(
                    "mix_distractor_sampling", "per_view"
                ),
                condition_separation_ppm=task_dict.get(
                    "mix_condition_separation_ppm", 10.0
                ),
                neutral_mass_separation_ppm=task_dict.get(
                    "mix_neutral_mass_separation_ppm", 10.0
                ),
                merge_ppm=task_dict.get("mix_merge_ppm", 5.0),
                intensity_normalization=task_dict.get(
                    "mix_intensity_normalization", "none"
                ),
                padding_value=0,
            )

        self.use_mass = self.teacher.backbone.use_mass
        self.use_charge = self.teacher.backbone.use_charge
        # The dIon dual objective uses the otherwise unused charge-0 embedding
        # row as an explicit student-only null precursor state. Real batches are
        # validated above and may only contain charges in [1, max_charge].
        self.precursor_null_student_crops = task_dict.get(
            "precursor_null_student_crops", None
        )
        legacy_null_locals = _as_bool(task_dict.get("hybrid_precursor_null_locals", False))
        if legacy_null_locals:
            if self.precursor_null_student_crops not in (None, "local"):
                raise ValueError(
                    "hybrid_precursor_null_locals conflicts with "
                    "precursor_null_student_crops."
                )
            self.precursor_null_student_crops = "local"
        if self.precursor_null_student_crops not in (None, "global", "local"):
            raise ValueError(
                "precursor_null_student_crops must be one of null, 'global', or 'local'."
            )
        self.dino_student_groups = None
        self.dino_student_group_weights = None
        if self.precursor_null_student_crops is not None:
            if not (self.use_mass and self.use_charge):
                raise ValueError(
                    "Precursor-null student crops require both use_mass and use_charge."
                )
            if self.num_local_crops < 1:
                raise ValueError(
                    "Precursor-null student crops require at least one local crop."
                )
            if self.precursor_null_student_crops == "local":
                if self.mix_aug is None or task_dict.get("mix_apply_to") != "global_only":
                    raise ValueError(
                        "Local precursor-null crops require global-only distractor mixing."
                    )
                self.dino_student_groups = (
                    ["conditional_global"] * self.num_global_crops
                    + ["null_local"] * self.num_local_crops
                )
                default_weights = {"conditional_global": 0.5, "null_local": 0.5}
            else:
                if self.mix_aug is not None:
                    raise ValueError(
                        "Global precursor-null crops are only supported by the clean H26 control."
                    )
                self.dino_student_groups = (
                    ["null_global"] * self.num_global_crops
                    + ["conditional_local"] * self.num_local_crops
                )
                default_weights = {"null_global": 0.5, "conditional_local": 0.5}
            self.dino_student_group_weights = task_dict.get(
                "dino_student_group_weights",
                default_weights,
            )

        self.student, self.teacher = _maybe_compile_dino_models(
            self.student,
            self.teacher,
            task_dict,
        )

    def on_fit_start(self):
        ret = _DionPretrainBase.on_fit_start(self)
        niter_per_ep = _get_niter_per_epoch(self.trainer)
        self.teacher_temp_step_schedule = self._build_teacher_temp_step_schedule(
            warmup_teacher_temp=self.warmup_teacher_temp,
            teacher_temp=self.teacher_temp,
            warmup_teacher_temp_epochs=self.warmup_teacher_temp_epochs,
            niter_per_ep=niter_per_ep,
        )
        if self.use_ibot:
            self.ibot_teacher_temp_step_schedule = self._build_teacher_temp_step_schedule(
                warmup_teacher_temp=self.ibot_warmup_teacher_temp,
                teacher_temp=self.ibot_teacher_temp,
                warmup_teacher_temp_epochs=self.ibot_warmup_teacher_temp_epochs,
                niter_per_ep=niter_per_ep,
            )
        return ret

    def on_train_start(self):
        """Start legacy-checkpoint Gram clocks only after Lightning restores state."""
        ret = super().on_train_start()
        if self.gram_enabled and self.gram_refinement_start_step.item() < 0:
            self.gram_refinement_start_step.fill_(int(self.trainer.global_step))
        if (
            self.gram_enabled
            and self.gram_teacher_refresh
            and self.gram_teacher_refresh_start_step.item() < 0
        ):
            self.gram_teacher_refresh_start_step.fill_(int(self.trainer.global_step))
        return ret

    def _load_gram_teacher_backbone(self, checkpoint_path):
        """Load only an EMA/student backbone from a trusted Lightning checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            legacy_numpy_globals = [
                (numpy_scalar, "numpy.core.multiarray.scalar"),
                (np.dtype, "numpy.dtype"),
                type(np.dtype(np.float64)),
            ]
            with torch.serialization.safe_globals(legacy_numpy_globals):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("Gram teacher checkpoint must contain a state_dict mapping.")

        # Prefer the ordinary EMA teacher backbone. Projection heads and centers
        # are deliberately excluded: Gram acts on contextualized peak tokens.
        for prefix in ("teacher.backbone.", "student.backbone.", "encoder."):
            backbone_state = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if backbone_state:
                break
        else:
            raise RuntimeError(
                "Could not find a DINO backbone in gram_teacher_checkpoint; expected "
                "teacher.backbone.*, student.backbone.*, or encoder.* keys."
            )

        missing, unexpected = self.gram_teacher.load_state_dict(backbone_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Gram teacher backbone does not exactly match the refinement "
                f"architecture; missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )
        print(
            f"Loaded frozen Gram-teacher backbone from {checkpoint_path} "
            f"using {prefix} weights."
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Allow a base DINO checkpoint to initialize a new Gram refinement."""
        if self.gram_enabled:
            state_dict = dict(state_dict)
            if not any(key.startswith("gram_teacher.") for key in state_dict):
                state_dict.update(
                    {
                        f"gram_teacher.{key}": value
                        for key, value in self.gram_teacher.state_dict().items()
                    }
                )
            if "gram_refinement_start_step" not in state_dict:
                state_dict["gram_refinement_start_step"] = (
                    self.gram_refinement_start_step.detach().clone()
                )
            if "gram_teacher_refresh_start_step" not in state_dict:
                state_dict["gram_teacher_refresh_start_step"] = (
                    self.gram_teacher_refresh_start_step.detach().clone()
                )
            if "gram_teacher_refresh_count" not in state_dict:
                state_dict["gram_teacher_refresh_count"] = (
                    self.gram_teacher_refresh_count.detach().clone()
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _current_gram_weight(self):
        if not self.gram_enabled:
            return 0.0
        start_step = int(self.gram_refinement_start_step.item())
        if start_step < 0:
            return 0.0
        if self.gram_warmup_steps == 0:
            return self.gram_loss_weight
        completed_steps = max(0, int(self.trainer.global_step) - start_step + 1)
        progress = min(1.0, completed_steps / self.gram_warmup_steps)
        if self.gram_warmup_schedule == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.gram_loss_weight * progress

    @torch.no_grad()
    def _maybe_refresh_gram_teacher(self):
        """Hard-copy the ordinary EMA backbone on the configured relative clock."""
        if (
            not self.gram_enabled
            or not self.gram_teacher_refresh
            or self.gram_teacher_refresh_count.item()
            >= self.gram_teacher_refresh_max_updates
        ):
            return False
        start_step = int(self.gram_teacher_refresh_start_step.item())
        if start_step < 0:
            return False
        completed_steps = max(0, int(self.trainer.global_step) - start_step + 1)
        if completed_steps < self.gram_teacher_refresh_first_step:
            return False
        if (
            completed_steps - self.gram_teacher_refresh_first_step
        ) % self.gram_teacher_refresh_every_n_steps:
            return False
        self.gram_teacher.load_state_dict(self.teacher.backbone.state_dict(), strict=True)
        self.gram_teacher.eval()
        self.gram_teacher_refresh_count.add_(1)
        return True

    def _gram_peak_tokens(self, backbone, crop, mass, charge):
        """Return only physical peak tokens for one clean, aligned crop."""
        sequences, input_padding_mask = crop
        out = backbone(
            sequences,
            key_padding_mask=input_padding_mask,
            mass=mass if self.use_mass else None,
            charge=charge if self.use_charge else None,
        )
        num_prefix_tokens = int(out["num_cem_tokens"])
        peak_tokens = out["emb"][:, num_prefix_tokens:, :]
        output_padding_mask = out["mask"]
        if output_padding_mask is None:
            peak_padding_mask = input_padding_mask
        else:
            peak_padding_mask = output_padding_mask[:, num_prefix_tokens:]
        if peak_tokens.shape[:2] != peak_padding_mask.shape:
            raise RuntimeError("Gram peak tokens and padding mask are misaligned.")
        return peak_tokens, peak_padding_mask

    @staticmethod
    def _build_teacher_temp_step_schedule(
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        niter_per_ep,
    ):
        """
        Build the official-DINOv2-style teacher temperature schedule.

        Matches ``dinov2/train/train.py`` in
        https://github.com/facebookresearch/dinov2:
        ``warmup_teacher_temp_epochs`` is configured in epochs but converted to
        ``warmup_teacher_temp_epochs * epoch_length`` iterations, and
        ``teacher_temp_schedule[iteration]`` is passed into the loss. With equal
        base/final values, this is a per-step linear warmup followed by a fixed
        final temperature.
        """
        warmup_iters = int(warmup_teacher_temp_epochs * niter_per_ep)
        if warmup_iters <= 0:
            return np.array([float(teacher_temp)], dtype=np.float64)
        return np.linspace(
            float(warmup_teacher_temp),
            float(teacher_temp),
            warmup_iters,
            dtype=np.float64,
        )

    def _current_teacher_temp(self):
        if self.teacher_temp_step_schedule is None:
            return self.dino_loss.teacher_temp_schedule[
                min(self.trainer.current_epoch, len(self.dino_loss.teacher_temp_schedule) - 1)
            ]
        step = min(self.trainer.global_step, len(self.teacher_temp_step_schedule) - 1)
        return self.teacher_temp_step_schedule[step]

    def _current_ibot_teacher_temp(self):
        if self.ibot_loss is None:
            return None
        if self.ibot_teacher_temp_step_schedule is None:
            return self.ibot_loss.teacher_temp_schedule[
                min(self.trainer.current_epoch, len(self.ibot_loss.teacher_temp_schedule) - 1)
            ]
        step = min(
            self.trainer.global_step,
            len(self.ibot_teacher_temp_step_schedule) - 1,
        )
        return self.ibot_teacher_temp_step_schedule[step]

    def training_step(self, batch, batch_idx):
        result = BasePLWrapper.training_step(self, batch, batch_idx)
        self.update_teacher(self._current_teacher_momentum())
        self._maybe_refresh_gram_teacher()
        return result

    def _sample_global_mask_ratios(self, batch_size, device):
        """
        Spectrum analogue of official ``collate_data_and_cast`` mask sampling.

        Official DINOv2 samples masks for a fraction of the collated global
        crops using ``mask_sample_probability`` and linearly spaced intervals
        over ``mask_ratio_min_max``. Here the sampled ratio controls random peak
        masking rather than image block masking.
        """
        total_global_crops = self.num_global_crops * batch_size
        ratios = torch.zeros(total_global_crops, device=device)
        n_samples_masked = int(total_global_crops * self.ibot_mask_sample_probability)
        if n_samples_masked <= 0:
            return ratios

        probs = torch.linspace(
            self.ibot_mask_ratio_min,
            self.ibot_mask_ratio_max,
            n_samples_masked + 1,
            device=device,
        )
        sampled = torch.empty(n_samples_masked, device=device).uniform_(0, 1)
        sampled = probs[:-1] + sampled * (probs[1:] - probs[:-1])
        ratios[torch.randperm(total_global_crops, device=device)[:n_samples_masked]] = (
            sampled
        )
        return ratios

    def _sample_dense_mask(self, pad_mask, mask_ratios):
        """Vectorized random peak masking counterpart to official DINOv2 block masks."""
        valid = ~pad_mask
        valid_counts = valid.sum(dim=1)
        num_mask = torch.floor(valid_counts.float() * mask_ratios).long()
        num_mask = torch.where(mask_ratios > 0, num_mask.clamp(min=1), num_mask)
        num_mask = torch.minimum(num_mask, valid_counts)

        random_scores = torch.rand(pad_mask.shape, device=pad_mask.device)
        random_scores = random_scores.masked_fill(~valid, 2.0)
        random_ranks = random_scores.argsort(dim=1).argsort(dim=1)
        return valid & (random_ranks < num_mask.unsqueeze(1))

    def _build_aligned_dense_masks(self, teacher_crops):
        """Sample masks for clean global crops without changing DINO crops."""
        dense_masks = []
        batch_size = teacher_crops[0][0].shape[0]
        mask_ratios = self._sample_global_mask_ratios(
            batch_size=batch_size,
            device=teacher_crops[0][0].device,
        )
        for crop_idx in range(self.num_global_crops):
            _, pad_mask = teacher_crops[crop_idx]
            ratio_slice = mask_ratios[
                crop_idx * batch_size : (crop_idx + 1) * batch_size
            ]
            dense_masks.append(self._sample_dense_mask(pad_mask, ratio_slice))
        return dense_masks

    @staticmethod
    def _validate_real_precursor_charges(
        precursor_charge: torch.Tensor, max_charge: int
    ) -> None:
        """Validate real DINOv2 precursor charges before view construction.

        Charge index zero is reserved for the explicit precursor-null student
        views used by the hybrid distractor objective. It must never enter from
        an observed spectrum. This is deliberately local to DINOv2: legacy
        downstream datasets may retain different charge conventions.
        """
        if precursor_charge.numel() == 0:
            raise ValueError("DINOv2 received an empty precursor_charge tensor.")

        values = precursor_charge.reshape(-1)
        if torch.is_floating_point(values):
            invalid = (~torch.isfinite(values)) | (values != values.round())
        else:
            invalid = torch.zeros_like(values, dtype=torch.bool)
        invalid |= (values < 1) | (values > max_charge)
        if not invalid.any():
            return

        observed = torch.unique(values[invalid]).detach().cpu().tolist()
        raise ValueError(
            "DINOv2 real precursor_charge values must be integral and in "
            f"[1, {max_charge}]; observed invalid values {observed}. "
            "Charge 0 is reserved for explicit precursor-null student views."
        )

    def _parse_batch(self, batch, Eval=None):
        spectra = batch
        self._validate_real_precursor_charges(
            spectra["precursor_charge"], self.max_charge
        )
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        lengths = spectra["peak_lengths"]

        crops = self.aug(mzab, lengths, self.rand_window_size)
        teacher_crops = crops[: self.num_global_crops]
        student_crops = list(crops)
        dino_student_crop_indices = tuple(range(len(student_crops)))
        mix_strength = 0.0
        mix_provenance = None
        if (not Eval) and (self.mix_aug is not None):
            mix_strength = self._current_mix_strength()
            student_crops, mix_provenance = self.mix_aug(
                student_crops,
                spectra=mzab,
                lengths=lengths,
                precursor_mz=spectra["precursor_mz"],
                precursor_charge=spectra["precursor_charge"],
                strength=mix_strength,
                num_global_crops=self.num_global_crops,
                return_provenance=True,
            )
        batch_size = mzab.shape[0]

        num_dino_crops = len(student_crops)
        real_mass = spectra["precursor_mass"]
        mass = real_mass.unsqueeze(0).repeat((num_dino_crops, 1))
        charge_values = spectra["precursor_charge"].to(dtype=torch.long)
        charge = charge_values.unsqueeze(0).repeat((num_dino_crops, 1))
        if self.precursor_null_student_crops == "global":
            # Student-only global views use the learned charge-0 null state plus
            # the deterministic Fourier encoding of mass 0.
            mass[: self.num_global_crops].zero_()
            charge[: self.num_global_crops].zero_()
        elif self.precursor_null_student_crops == "local":
            # Student-only local views use the learned charge-0 null state plus
            # the deterministic Fourier encoding of mass 0.
            mass[self.num_global_crops :].zero_()
            charge[self.num_global_crops :].zero_()

        if self.use_ibot:
            # Clean, conditioned views are appended solely for dense iBOT.
            # The original hybrid crops above retain their existing DINO roles.
            dense_masks = self._build_aligned_dense_masks(teacher_crops)
            ibot_student_crop_indices = tuple(
                range(num_dino_crops, num_dino_crops + self.num_global_crops)
            )
            student_crops.extend(teacher_crops)
            mass = torch.cat(
                [mass, real_mass.unsqueeze(0).repeat((self.num_global_crops, 1))]
            )
            charge = torch.cat(
                [charge, charge_values.unsqueeze(0).repeat((self.num_global_crops, 1))]
            )
            student_token_masks = [None] * num_dino_crops + dense_masks
        else:
            dense_masks = [None] * self.num_global_crops
            ibot_student_crop_indices = ()
            student_token_masks = [None] * num_dino_crops

        parsed_batch = {
            "mass": mass,
            "charge": charge,
            "student_crops": student_crops,
            "teacher_crops": teacher_crops,
            "dense_masks": dense_masks,
            "student_token_masks": student_token_masks,
            "dino_student_crop_indices": dino_student_crop_indices,
            "ibot_student_crop_indices": ibot_student_crop_indices,
            "mix_provenance": mix_provenance,
            "mix_strength": mix_strength,
            "real_precursor_mass": real_mass,
            "real_precursor_charge": charge_values,
        }
        return parsed_batch, batch_size

    def forward(self, parsed_batch, **kwargs):
        student_features = self.student.forward_features(
            parsed_batch["student_crops"],
            mass=parsed_batch["mass"] if self.use_mass else None,
            charge=parsed_batch["charge"] if self.use_charge else None,
            return_patch_tokens=self.use_ibot,
            patch_token_indices=(
                parsed_batch["ibot_student_crop_indices"] if self.use_ibot else None
            ),
            dino_crop_indices=parsed_batch["dino_student_crop_indices"],
            return_pooled=self.koleo_loss_weight > 0,
            token_masks=parsed_batch["student_token_masks"] if self.use_ibot else None,
            ibot_masks=parsed_batch["student_token_masks"] if self.use_ibot else None,
        )
        with torch.no_grad():
            crops_t = parsed_batch["teacher_crops"]
            mass_t = (
                parsed_batch["mass"][: self.num_global_crops]
                if self.use_mass
                else None
            )
            charge_t = (
                parsed_batch["charge"][: self.num_global_crops]
                if self.use_charge
                else None
            )
            teacher_features = self.teacher.forward_features(
                crops_t,
                mass=mass_t,
                charge=charge_t,
                return_patch_tokens=self.use_ibot,
                patch_token_indices=range(self.num_global_crops) if self.use_ibot else None,
                dino_crop_indices=range(self.num_global_crops),
                ibot_masks=parsed_batch["dense_masks"] if self.use_ibot else None,
            )
        student_pooled = student_features.get("pooled_output")
        if student_pooled is not None:
            batch_size = parsed_batch["student_crops"][0][0].shape[0]
            student_pooled = student_pooled[: self.num_global_crops * batch_size]

        features = {
            "student_out": student_features["dino_output"],
            "teacher_out": teacher_features["dino_output"],
            "student_pooled": student_pooled,
        }
        if self.gram_enabled:
            if not 0 <= self.gram_clean_crop_index < self.num_global_crops:
                raise RuntimeError(
                    "gram_clean_crop_index must select one of the clean global crops."
                )
            clean_crop = parsed_batch["teacher_crops"][self.gram_clean_crop_index]
            student_gram_tokens, gram_padding_mask = self._gram_peak_tokens(
                self.student.backbone,
                clean_crop,
                parsed_batch["real_precursor_mass"],
                parsed_batch["real_precursor_charge"],
            )
            with torch.no_grad():
                teacher_gram_tokens, teacher_padding_mask = self._gram_peak_tokens(
                    self.gram_teacher,
                    clean_crop,
                    parsed_batch["real_precursor_mass"],
                    parsed_batch["real_precursor_charge"],
                )
            if not torch.equal(gram_padding_mask, teacher_padding_mask):
                raise RuntimeError("Student and Gram-teacher peak padding masks differ.")
            features.update(
                gram_student_tokens=student_gram_tokens,
                gram_teacher_tokens=teacher_gram_tokens,
                gram_padding_mask=gram_padding_mask,
            )
        if self.use_ibot:
            features["student_patch_out"] = [
                student_features["patch_outputs"][index]
                for index in parsed_batch["ibot_student_crop_indices"]
            ]
            features["teacher_patch_out"] = teacher_features["patch_outputs"][
                : self.num_global_crops
            ]
        return features

    def _get_losses(self, returns, parsed_batch):
        teacher_temp = self._current_teacher_temp()
        ibot_teacher_temp = self._current_ibot_teacher_temp()
        dino_loss, dino_group_losses = self.dino_loss(
            returns["student_out"],
            returns["teacher_out"],
            epoch=self.trainer.current_epoch,
            teacher_temp=teacher_temp,
            student_groups=self.dino_student_groups,
            student_group_weights=self.dino_student_group_weights,
            return_group_losses=True,
        )
        if self.use_ibot:
            ibot_loss = self.ibot_loss(
                returns["student_patch_out"],
                returns["teacher_patch_out"],
                parsed_batch["dense_masks"],
                epoch=self.trainer.current_epoch,
                teacher_temp=ibot_teacher_temp,
            )
        else:
            ibot_loss = torch.zeros_like(dino_loss)
        if returns["student_pooled"] is None:
            koleo_loss = torch.zeros_like(dino_loss)
        else:
            # Official DINOv2 applies KoLeo separately to each global-crop
            # chunk to avoid nearest-neighbor matching between two views of
            # the same image/spectrum.
            koleo_loss = sum(
                self.koleo_loss(p)
                for p in returns["student_pooled"].chunk(self.num_global_crops)
            )

        loss = (
            self.dino_loss_weight * dino_loss
            + self.ibot_loss_weight * ibot_loss
            + self.koleo_loss_weight * koleo_loss
        )
        if self.gram_enabled:
            gram_loss = per_spectrum_gram_mse(
                returns["gram_student_tokens"],
                returns["gram_teacher_tokens"],
                returns["gram_padding_mask"],
            )
            gram_weight = self._current_gram_weight()
            loss = loss + gram_weight * gram_loss
        else:
            gram_loss = torch.zeros_like(dino_loss)
            gram_weight = 0.0
        if not torch.all(torch.isfinite(loss)):
            print("Loss is NaN")
            raise RuntimeError("Loss is NaN")
        return (
            loss,
            dino_loss,
            ibot_loss,
            koleo_loss,
            gram_loss,
            gram_weight,
            dino_group_losses,
        )

    @staticmethod
    def _cross_view_ce(student_output, teacher_targets, student_temp, num_crops):
        student_chunks = (student_output / student_temp).chunk(num_crops)
        teacher_chunks = teacher_targets.chunk(num_crops)
        terms = []
        for teacher_idx, target in enumerate(teacher_chunks):
            for student_idx, logits in enumerate(student_chunks):
                if student_idx == teacher_idx:
                    continue
                terms.append(
                    torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()
                )
        if not terms:
            raise RuntimeError("Global DINO diagnostic has no cross-view terms.")
        return torch.stack(terms).mean()

    @torch.no_grad()
    def _dual_condition_global_diagnostics(self, returns, parsed_batch, teacher_temp):
        if not (self.use_mass and self.use_charge):
            return {}

        global_crops = parsed_batch["teacher_crops"]
        num_globals = self.num_global_crops
        batch_size = global_crops[0][0].shape[0]
        num_global_rows = num_globals * batch_size
        existing_global_output = returns["student_out"][:num_global_rows]

        conditioned_mass = parsed_batch["real_precursor_mass"].unsqueeze(0).repeat(
            (num_globals, 1)
        )
        conditioned_charge = parsed_batch["real_precursor_charge"].unsqueeze(0).repeat(
            (num_globals, 1)
        )
        if self.precursor_null_student_crops == "global":
            null_output = existing_global_output
            conditioned_output = self.student.forward_features(
                global_crops,
                mass=conditioned_mass,
                charge=conditioned_charge,
            )["dino_output"]
        else:
            conditioned_output = existing_global_output
            null_output = self.student.forward_features(
                global_crops,
                mass=torch.zeros_like(conditioned_mass),
                charge=torch.zeros_like(conditioned_charge),
            )["dino_output"]

        teacher_output = returns["teacher_out"]
        teacher_targets = self.dino_loss._teacher_targets(
            teacher_output, teacher_temp
        ).detach()
        conditioned_ce = self._cross_view_ce(
            conditioned_output,
            teacher_targets,
            self.dino_loss.student_temp,
            num_globals,
        )
        null_ce = self._cross_view_ce(
            null_output,
            teacher_targets,
            self.dino_loss.student_temp,
            num_globals,
        )

        raw_teacher = F.softmax(
            (teacher_output - self.dino_loss.center) / teacher_temp,
            dim=-1,
        )
        conditioned_raw = F.log_softmax(
            conditioned_output / self.dino_loss.student_temp, dim=-1
        )
        null_raw = F.log_softmax(
            null_output / self.dino_loss.student_temp, dim=-1
        )
        target_entropy = -torch.sum(
            teacher_targets * teacher_targets.clamp_min(1e-12).log(), dim=-1
        ).mean() / math.log(2)

        target_marginal = teacher_targets.sum(dim=0)
        target_count = teacher_targets.new_tensor(float(teacher_targets.shape[0]))
        if dist.is_initialized():
            dist.all_reduce(target_marginal)
            dist.all_reduce(target_count)
        target_marginal = target_marginal / target_count
        marginal_entropy = -torch.sum(
            target_marginal * target_marginal.clamp_min(1e-12).log()
        ) / math.log(2)

        return {
            "diag/sinkhorn_ce/conditioned_global_clean": conditioned_ce,
            "diag/sinkhorn_ce/null_global_clean": null_ce,
            "diag/raw_softmax_kl/conditioned_global_clean_aligned": F.kl_div(
                conditioned_raw, raw_teacher, reduction="batchmean"
            ),
            "diag/raw_softmax_kl/null_global_clean_aligned": F.kl_div(
                null_raw, raw_teacher, reduction="batchmean"
            ),
            "diag/sinkhorn_teacher_entropy_bits": target_entropy,
            "diag/sinkhorn_teacher_marginal_entropy_bits": marginal_entropy,
        }

    def _get_train_stats(self, returns, parsed_batch, **kwargs):
        (
            loss,
            dino_loss,
            ibot_loss,
            koleo_loss,
            gram_loss,
            gram_weight,
            group_losses,
        ) = self._get_losses(returns, parsed_batch)
        stats = {
            "loss": loss,
            "dino_loss": dino_loss,
            "ibot_loss": ibot_loss,
            "koleo_loss": koleo_loss,
            "gram_loss": gram_loss,
            "gram_loss_weight": gram_weight,
            "mix_strength": parsed_batch.get("mix_strength", 0.0),
        }
        stats.update(
            {f"dino_group_{name}_loss": value for name, value in group_losses.items()}
        )
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch, **kwargs):
        (
            loss,
            dino_loss,
            ibot_loss,
            koleo_loss,
            gram_loss,
            gram_weight,
            group_losses,
        ) = self._get_losses(returns, parsed_batch)
        stats = {
            "loss": loss,
            "dino_loss": dino_loss,
            "ibot_loss": ibot_loss,
            "koleo_loss": koleo_loss,
            "gram_loss": gram_loss,
            "gram_loss_weight": gram_weight,
        }
        stats.update(
            {f"dino_group_{name}_loss": value for name, value in group_losses.items()}
        )
        diag = self._dino_diagnostics(
            returns["student_out"], returns["teacher_out"], self.trainer.current_epoch
        )
        diag.update(
            self._dual_condition_global_diagnostics(
                returns, parsed_batch, float(self._current_teacher_temp())
            )
        )
        self.log_dict(
            {**diag},
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return stats

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.gram_enabled:
            self.log(
                "schedule/gram_loss_weight",
                self._current_gram_weight(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            if self.gram_teacher_refresh:
                self.log(
                    "schedule/gram_teacher_refresh_count",
                    float(self.gram_teacher_refresh_count.item()),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.student.parameters(),
            betas=(0.9, 0.9999),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def train(self, mode=True):
        ret = super().train(mode)
        # Official DINOv2 keeps the EMA teacher in eval mode while training.
        self.teacher.eval()
        if self.gram_teacher is not None:
            self.gram_teacher.eval()
        return ret
