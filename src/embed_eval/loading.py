"""Standalone checkpoint loading for dIon models exposing ``get_embedder``."""

from __future__ import annotations

import torch

import src.models.binned_encoder as binned_encoder
import src.models.casanovo.encoder_interface as casanovo_encoders
import src.models.custom.encoder as custom_encoders
import src.models.dc_models.dc_encoder as dc_encoders
from src.metric_learning import MetricLearningEmbedder, MetricProjectionHead
from src.wrappers.dummy_wrapper import DummyEmbedderWrapper
from src.wrappers.pretrain_wrappers import (
    dIonPretrainWrapper,
)


ENCODER_DICT = {
    **custom_encoders.__dict__,
    **dc_encoders.__dict__,
    **casanovo_encoders.__dict__,
    **binned_encoder.__dict__,
}
PRETRAIN_TASK_DICT = {
    "dion": dIonPretrainWrapper,
    "dummy": DummyEmbedderWrapper,
}


def load_checkpoint_embedder(global_args, pretrain_config: dict):
    """Load any dIon pretraining wrapper that implements ``get_embedder``."""
    if not global_args.encoder_weights:
        raise ValueError("--encoder_weights is required for standalone embedding evaluation.")
    task_name = global_args.pretraining_task
    if task_name not in PRETRAIN_TASK_DICT:
        raise ValueError(f"No pretraining wrapper registered for {task_name!r}.")
    if task_name not in pretrain_config:
        raise ValueError(f"Missing {task_name!r} section in pretraining config.")
    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
        dropout=pretrain_config[task_name].get("dropout", 0),
        cls_token=global_args.cls_token,
        max_charge=global_args.max_charge,
    )
    wrapper_class = PRETRAIN_TASK_DICT[task_name]
    checkpoint = torch.load(global_args.encoder_weights, map_location="cpu")
    wrapper = wrapper_class(
        encoder=encoder,
        global_args=global_args,
        task_dict=pretrain_config[task_name],
    )
    missing, unexpected = wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_unexpected = {"encoder.charge_emb.weight", "encoder.charge_emb.bias"}
    unexpected = set(unexpected) - allowed_unexpected
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch. missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    embedding_readout = getattr(global_args, "embedding_readout", "backbone")
    if task_name not in {"dion"} and embedding_readout != "backbone":
        raise ValueError(
            "--embedding_readout is currently supported only for dIon "
            "checkpoints."
        )
    try:
        if task_name in {"dion"}:
            embedder = wrapper.get_embedder(
                trainable=False,
                embedding_readout=embedding_readout,
            )
        else:
            embedder = wrapper.get_embedder(trainable=False)
    except NotImplementedError as exc:
        raise ValueError(
            f"{wrapper_class.__name__} does not expose an embedding wrapper."
        ) from exc
    return wrapper, embedder


def _load_prefixed_module_state(module, state_dict: dict, prefix: str) -> None:
    module_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not module_state:
        raise ValueError(f"Checkpoint contains no {prefix!r} state entries.")
    missing, unexpected = module.load_state_dict(module_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Metric-learning checkpoint mismatch for {prefix!r}. "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def load_metric_learning_checkpoint_embedder(
    global_args, pretrain_config: dict, downstream_config: dict
) -> MetricLearningEmbedder:
    """Load the trained encoder, DINO pooler, and normalized SupCon metric head.

    Metric-learning checkpoints are downstream Lightning checkpoints, rather
    than DINO wrapper checkpoints. Reconstructing only their encoder would
    silently discard the learned global pooler/head, so standalone retrieval
    and pair evaluation must use this composite embedder.
    """
    checkpoint_path = getattr(global_args, "downstream_weights", None)
    if not checkpoint_path:
        raise ValueError(
            "--downstream_weights is required to evaluate a metric-learning checkpoint."
        )
    task_name = global_args.pretraining_task
    if task_name not in {"dion"}:
        raise ValueError("metric-learning standalone evaluation requires dIon.")
    if task_name not in pretrain_config:
        raise ValueError(f"Missing {task_name!r} section in pretraining config.")
    if global_args.encoder_model not in ENCODER_DICT:
        raise ValueError(f"Unknown encoder model {global_args.encoder_model!r}.")

    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
        dropout=pretrain_config[task_name].get("dropout", 0),
        cls_token=global_args.cls_token,
        max_charge=global_args.max_charge,
    )
    pretrain_wrapper = PRETRAIN_TASK_DICT[task_name](
        encoder=encoder,
        global_args=global_args,
        task_dict=pretrain_config[task_name],
    )
    metric_encoder = pretrain_wrapper.get_encoder(trainable=False)
    pooler = getattr(getattr(pretrain_wrapper, "teacher", None), "pooler", None)
    if pooler is None:
        raise TypeError("metric-learning checkpoint requires a DINO teacher pooler.")
    metric_head = MetricProjectionHead(
        input_dim=metric_encoder.running_units,
        hidden_dim=downstream_config.get("metric_hidden_dim"),
        output_dim=int(downstream_config.get("metric_embedding_dim", 128)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Metric-learning checkpoint has no state_dict mapping.")
    _load_prefixed_module_state(metric_encoder, state_dict, "encoder.")
    _load_prefixed_module_state(pooler, state_dict, "pooler.")
    _load_prefixed_module_state(metric_head, state_dict, "decoder.")
    for module in (metric_encoder, pooler, metric_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return MetricLearningEmbedder(metric_encoder, pooler, metric_head)


def load_peak_only_binned_spectrum_embedder(global_args):
    """Instantiate the fixed 1,024-D metadata-free binned spectrum baseline."""
    if bool(global_args.use_mass) or bool(global_args.use_charge):
        raise ValueError(
            "binned_spectrum requires --use_mass 0 and --use_charge 0; "
            "it must contain peak information only."
        )
    encoder = binned_encoder.encoder_binned_baseline_1024d(
        max_mz=float(global_args.max_mz),
        max_charge=int(global_args.max_charge),
        use_mass=False,
        use_charge=False,
    )
    wrapper = DummyEmbedderWrapper(encoder)
    return wrapper, wrapper.get_embedder(trainable=False)


def load_checkpoint_encoder(global_args, pretrain_config: dict):
    """Load a checkpoint and return its dense encoder backbone.

    dIon wrappers return the EMA teacher backbone, matching downstream
    fine-tuning and avoiding the pooled ``DINOEmbedder`` readout.
    """
    wrapper, _ = load_checkpoint_embedder(global_args, pretrain_config)
    try:
        encoder = wrapper.get_encoder(trainable=False)
    except (AttributeError, NotImplementedError) as exc:
        raise ValueError(
            f"{type(wrapper).__name__} does not expose a dense encoder backbone."
        ) from exc
    return wrapper, encoder


def load_checkpoint_teacher_backbone(global_args, pretrain_config: dict):
    """Load only the ordinary EMA teacher backbone from a dIon checkpoint.

    This bypasses wrapper state so dense-token comparisons are compatible with
    Gram-refinement checkpoints, whose auxiliary Gram teacher is not part of
    the EMA representation being evaluated.
    """
    if not global_args.encoder_weights:
        raise ValueError("--encoder_weights is required for dense-token evaluation.")
    task_name = global_args.pretraining_task
    if task_name not in {"dion"}:
        raise ValueError("Dense EMA-backbone loading requires a dIon checkpoint.")
    if task_name not in pretrain_config:
        raise ValueError(f"Missing {task_name!r} section in pretraining config.")
    if global_args.encoder_model not in ENCODER_DICT:
        raise ValueError(f"Unknown encoder model {global_args.encoder_model!r}.")
    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
        dropout=pretrain_config[task_name].get("dropout", 0),
        cls_token=global_args.cls_token,
        max_charge=global_args.max_charge,
    )
    checkpoint = torch.load(global_args.encoder_weights, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a state_dict mapping.")
    backbone_state = {
        key.removeprefix("teacher.backbone."): value
        for key, value in state_dict.items()
        if key.startswith("teacher.backbone.")
    }
    if not backbone_state:
        raise RuntimeError("Checkpoint contains no ordinary EMA teacher backbone state.")
    missing, unexpected = encoder.load_state_dict(backbone_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "EMA teacher backbone does not exactly match the requested encoder; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder

def load_random_checkpoint_embedder(global_args, pretrain_config: dict):
    """Instantiate a seeded random wrapper and expose its normal global readout."""
    task_name = global_args.pretraining_task
    if task_name not in PRETRAIN_TASK_DICT:
        raise ValueError(f"No pretraining wrapper registered for {task_name!r}.")
    if task_name not in pretrain_config:
        raise ValueError(f"Missing {task_name!r} section in pretraining config.")
    if global_args.encoder_model not in ENCODER_DICT:
        raise ValueError(f"Unknown encoder model {global_args.encoder_model!r}.")
    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
        dropout=pretrain_config[task_name].get("dropout", 0),
        cls_token=global_args.cls_token,
        max_charge=global_args.max_charge,
    )
    wrapper = PRETRAIN_TASK_DICT[task_name](
        encoder=encoder,
        global_args=global_args,
        task_dict=pretrain_config[task_name],
    )
    if task_name not in {"dion"}:
        raise ValueError("Random standalone global evaluation currently requires dIon.")
    embedder = wrapper.get_embedder(
        trainable=False,
        embedding_readout=getattr(global_args, "embedding_readout", "backbone"),
    )
    return wrapper, embedder


def load_random_encoder(global_args, pretrain_config: dict):
    """Instantiate a pretraining wrapper and return its untrained dense encoder."""
    task_name = global_args.pretraining_task
    if task_name not in PRETRAIN_TASK_DICT:
        raise ValueError(f"No pretraining wrapper registered for {task_name!r}.")
    if task_name not in pretrain_config:
        raise ValueError(f"Missing {task_name!r} section in pretraining config.")
    if global_args.encoder_model not in ENCODER_DICT:
        raise ValueError(f"Unknown encoder model {global_args.encoder_model!r}.")
    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
        dropout=pretrain_config[task_name].get("dropout", 0),
        cls_token=global_args.cls_token,
        max_charge=global_args.max_charge,
    )
    wrapper = PRETRAIN_TASK_DICT[task_name](
        encoder=encoder,
        global_args=global_args,
        task_dict=pretrain_config[task_name],
    )
    try:
        dense_encoder = wrapper.get_encoder(trainable=False)
    except (AttributeError, NotImplementedError) as exc:
        raise ValueError(
            f"{type(wrapper).__name__} does not expose a dense encoder backbone."
        ) from exc
    return wrapper, dense_encoder


def load_downstream_encoder(global_args, pretrain_config: dict, checkpoint_path: str):
    """Load only the encoder from a supervised downstream Lightning checkpoint."""
    task_name = global_args.pretraining_task
    if task_name not in pretrain_config:
        raise ValueError(f"Missing {task_name!r} section in pretraining config.")
    if global_args.encoder_model not in ENCODER_DICT:
        raise ValueError(f"Unknown encoder model {global_args.encoder_model!r}.")
    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
        dropout=pretrain_config[task_name].get("dropout", 0),
        cls_token=global_args.cls_token,
        max_charge=global_args.max_charge,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:
        raise ValueError("Downstream checkpoint has no state_dict.")
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError("Downstream checkpoint contains no encoder.* state entries.")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Downstream checkpoint/encoder mismatch. "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder
