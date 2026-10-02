import pickle

import numpy as np
from numpy.core.multiarray import scalar as numpy_scalar
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import numpy as np
import random
from src.callbacks.linprobe_callback import EndAAProbeCallback
from src.callbacks.dense_denovo_probe_callback import DenseDeNovoProbeCallback
from src.callbacks.embedding_evaluation_callback import (
    SpectrumEmbeddingEvaluationCallback,
)
from src.callbacks.pair_evaluation_callback import PeptideIonPairEvaluationCallback
from src.probe_conditioning import get_probe_conditioning_modes
from src.data.lance_data_module import LanceDataModule, lance_callbacks
from src.parse_args import parse_args_and_config, create_output_dirs
import time
import shutil

from src.wrappers.downstream_wrappers import (
    DeNovoTeacherForcing,
    SpectralQualityAssessment,
    ChimericityAssessment,
    OxidizedMethionineAssessment,
    RetentionTimeProbe,
    SupervisedMetricLearning,
)
from src.metric_learning import MetricProjectionHead
from src import utils
from src.wrappers.pretrain_wrappers import (
    dIonPretrainWrapper,
)
from src.wrappers.dummy_wrapper import DummyEmbedderWrapper


import src.utils

import src.models.custom.encoder as encoders
import src.models.dc_models.dc_encoder as dc_encoders
import src.models.dc_models.dc_decoder as dc_decoders
import src.models.casanovo.encoder_interface as casanovo_encoders
import src.models.casanovo.decoder_interface as casanovo_decoders
import src.models.custom.heads as mlp_heads
import src.models.binned_encoder as binned_encoder
from src.external_embeddings import ExternalEmbeddingSource

ENCODER_DICT = {
    **encoders.__dict__,
    **dc_encoders.__dict__,
    **casanovo_encoders.__dict__,
    **binned_encoder.__dict__,
}
DECODER_DICT = {
    **dc_decoders.__dict__,
    **casanovo_decoders.__dict__,
    **mlp_heads.__dict__,
}

PRETRAIN_TASK_DICT = {
    "dion": dIonPretrainWrapper,
    "dummy": DummyEmbedderWrapper,
}

DOWNSTREAM_TASK_DICT = {
    "denovo_tf": DeNovoTeacherForcing,
    "sqa": SpectralQualityAssessment,
    "chimericity": ChimericityAssessment,
    "oxidized_met": OxidizedMethionineAssessment,
    "retention_time": RetentionTimeProbe,
    "metric_learning": SupervisedMetricLearning,
}


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)


def load_checkpoint_safely(path, map_location="cpu"):
    """Load tensor checkpoints without enabling arbitrary pickle execution.

    Older Pairwise Lightning checkpoints serialize NumPy scalar metadata
    under the pre-NumPy-2 module name. The allowlist below is intentionally
    limited to those metadata types; model state remains tensors only.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        legacy_numpy_globals = [
            (numpy_scalar, "numpy.core.multiarray.scalar"),
            (np.dtype, "numpy.dtype"),
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(legacy_numpy_globals):
            return torch.load(path, map_location=map_location, weights_only=True)


def adapt_historical_pairwise_state_dict(state_dict):
    """Map the older Pairwise encoder key name (PwSeq) to pw_seq, if needed."""
    if not any("encoder.PwSeq." in key for key in state_dict):
        return state_dict
    return {key.replace("encoder.PwSeq.", "encoder.pw_seq."): value for key, value in state_dict.items()}


def safe_load_pretrain_wrapper(klass, ckpt_path, **kwargs):
    ckpt = load_checkpoint_safely(ckpt_path, map_location="cpu")
    module = klass(**kwargs)

    missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)

    # Allowed unexpected keys only. Gram-refinement checkpoints contain a
    # separate frozen anchor encoder and refinement bookkeeping which are not
    # part of a downstream encoder. The regular EMA teacher remains required.
    allowed_unexpected = {
        "encoder.charge_emb.weight",
        "encoder.charge_emb.bias",
    }
    allowed_refinement_metadata = {
        "gram_refinement_start_step",
        "gram_teacher_refresh_start_step",
        "gram_teacher_refresh_count",
    }

    ignored_unexpected = {
        key
        for key in unexpected
        if key in allowed_unexpected
        or key in allowed_refinement_metadata
        or key.startswith("gram_teacher.")
    }
    unexpected_filtered = set(unexpected) - ignored_unexpected
    if unexpected_filtered:
        raise RuntimeError(f" Unexpected keys: {unexpected_filtered}")

    # Missing keys must be empty
    if missing:
        raise RuntimeError(f" Missing keys: {missing}")

    ignored = sorted(ignored_unexpected)
    if ignored:
        print(
            f"Loaded checkpoint; ignored {len(ignored)} non-downstream wrapper keys "
            f"(first: {ignored[:3]})"
        )
    else:
        print("Loaded checkpoint")
    return module


def metric_learning_evaluation_callbacks(probing_config, global_args):
    """Reuse canonical online embedding/pair callbacks for metric fine-tuning."""
    callbacks = []
    if probing_config is None:
        return callbacks
    for task_name, callback_class in (
        ("embedding_evaluation", SpectrumEmbeddingEvaluationCallback),
        ("pair_discrimination_evaluation", PeptideIonPairEvaluationCallback),
    ):
        if task_name not in probing_config:
            continue
        modes = get_probe_conditioning_modes(
            probing_config, task_name, global_args.precursor_conditioning
        )
        for mode in modes:
            callbacks.append(
                callback_class(
                    probing_config[task_name],
                    global_args,
                    precursor_conditioning=mode,
                    metric_prefix=mode if len(modes) > 1 else "",
                )
            )
    return callbacks


def main(
    global_args,
    pretrain_config=None,
    ds_config=None,
    probing_config=None,
):
    print(f"Saving checkpoints in {global_args.output_dir}")
    print(f"Saving logs in {global_args.log_dir}")

    config = {
        **vars(global_args),
        "downstream_config": ds_config,
        "pretrain_config": pretrain_config,
    }

    if global_args.subset:
        if global_args.downstream_task != "none":
            config["downstream_config"][global_args.downstream_task][
                "subset"
            ] = global_args.subset
        config["pretrain_config"][global_args.pretraining_task][
            "subset"
        ] = global_args.subset

    # Wandb stuff
    run = None
    logger = None
    if global_args.log_wandb and utils.get_rank() == 0:
        run = wandb.init(
            project=global_args.wandb_project,
            entity=global_args.wandb_entity,
            config=config,
            dir=global_args.log_dir,
            user="allow",
        )
        # this step is for automated hparam sweeping
        # Preserve trainer batch limit semantics: W&B may coerce 1.0 -> 1,
        # which changes "100% of batches" into "exactly 1 batch".
        preserved_runtime_limits = {
            "limit_train_batches": global_args.limit_train_batches,
            "limit_val_batches": global_args.limit_val_batches,
            "limit_test_batches": global_args.limit_test_batches,
        }
        update_args(global_args, dict(run.config))
        update_args(global_args, preserved_runtime_limits)
        config = dict(run.config)
        config.update(preserved_runtime_limits)
        logger = WandbLogger(experiment=run)

    external_embedding_cache = str(
        getattr(global_args, "external_embedding_cache", "") or ""
    ).strip()
    if external_embedding_cache:
        if global_args.downstream_task not in {
            "sqa", "chimericity", "oxidized_met", "retention_time"
        }:
            raise ValueError(
                "--external_embedding_cache is supported only by SQA-derived "
                "frozen downstream tasks."
            )
        if not global_args.freeze_encoder:
            raise ValueError(
                "--external_embedding_cache is frozen-only; external encoders "
                "cannot be fine-tuned through dIon."
            )
        if global_args.pretrain or global_args.encoder_weights:
            raise ValueError(
                "--external_embedding_cache replaces a dIon encoder checkpoint; "
                "do not also set --pretrain or --encoder_weights."
            )
        encoder = ExternalEmbeddingSource(external_embedding_cache)
        pl_encoder = None
        print(f"Using frozen external embedding cache: {encoder.cache_dir}")
    else:
        # Define encoder model
        encoder = ENCODER_DICT[global_args.encoder_model](
            use_charge=global_args.use_charge,
            use_mass=global_args.use_mass,
            use_energy=global_args.use_energy,
            dropout=config["pretrain_config"][global_args.pretraining_task].get(
                "dropout", 0
            ),
            cls_token=global_args.cls_token,
            max_charge=global_args.max_charge,
        )

    if global_args.pretraining_task not in PRETRAIN_TASK_DICT:
        raise NotImplementedError(
            f"{global_args.pretraining_task} pretraining task not implemented"
        )

    distributed = global_args.num_devices > 1 or global_args.num_nodes > 1
    if global_args.pretrain:
        pretrain_data_module = utils.get_lance_data_module(
            global_args,
            config["pretrain_config"],
            global_args.max_peaks,
            seed=global_args.seed,
            include_test=False,
        )
        pretrain_callbacks = utils.configure_callbacks(
            global_args,
            config["pretrain_config"][global_args.pretraining_task],
            global_args.pretraining_task + "_val_loss_epoch",
        )
        pretrain_callbacks += lance_callbacks(pretrain_data_module)

        if global_args.encoder_weights:
            pl_encoder = PRETRAIN_TASK_DICT[
                global_args.pretraining_task
            ].load_from_checkpoint(
                global_args.encoder_weights,
                global_args=global_args,
                encoder=encoder,
                task_dict=config["pretrain_config"][global_args.pretraining_task],
            )
            print(f"Loading encoder checkpoint: {global_args.encoder_weights}")
        else:
            # Instantiate PL wrapper based on the pretraining task
            pl_encoder = PRETRAIN_TASK_DICT[global_args.pretraining_task](
                encoder,
                global_args=global_args,
                task_dict=config["pretrain_config"][global_args.pretraining_task],
            )

        if probing_config is not None:
            probe_every_n_steps = probing_config.get(
                "online_probe_every_n_steps", global_args.probe_every_n_steps
            )
            for task_name in (
                "end_aa_pred",
                "embedding_evaluation",
                "pair_discrimination_evaluation",
                "dense_denovo_probe",
            ):
                if task_name not in probing_config:
                    continue
                probe_conditioning_modes = get_probe_conditioning_modes(
                    probing_config, task_name, global_args.precursor_conditioning
                )
                multiple_modes = len(probe_conditioning_modes) > 1
                for precursor_conditioning in probe_conditioning_modes:
                    metric_prefix = precursor_conditioning if multiple_modes else ""
                    if task_name == "end_aa_pred":
                        pretrain_callbacks.append(
                            EndAAProbeCallback(
                                probing_config,
                                global_args,
                                embedder_batch_size=pl_encoder.batch_size,
                                precursor_conditioning=precursor_conditioning,
                                probe_every_n_steps=probe_every_n_steps,
                                metric_prefix=metric_prefix,
                            )
                        )
                    elif task_name == "embedding_evaluation":
                        pretrain_callbacks.append(
                            SpectrumEmbeddingEvaluationCallback(
                                probing_config[task_name],
                                global_args,
                                precursor_conditioning=precursor_conditioning,
                                metric_prefix=metric_prefix,
                            )
                        )
                    elif task_name == "pair_discrimination_evaluation":
                        pretrain_callbacks.append(
                            PeptideIonPairEvaluationCallback(
                                probing_config[task_name],
                                global_args,
                                precursor_conditioning=precursor_conditioning,
                                metric_prefix=metric_prefix,
                            )
                        )
                    else:
                        pretrain_callbacks.append(
                            DenseDeNovoProbeCallback(
                                probing_config,
                                global_args,
                                embedder_batch_size=pl_encoder.batch_size,
                                precursor_conditioning=precursor_conditioning,
                                probe_every_n_steps=probe_every_n_steps,
                            )
                        )

        if run is not None and utils.get_rank() == 0:
            if global_args.watch_model:
                run.watch(pl_encoder, log="all")
            run.log(
                {
                    "num_parameters_encoder": utils.get_num_parameters(encoder),
                }
            )

        (
            print(
                f"Starting distributed pretraining using {global_args.num_devices} devices on {global_args.num_nodes} node(s)"
            )
            if distributed
            else print("Starting single-device training")
        )

        # Define trainer
        pretrainer = pl.Trainer(
            # Distributed kwargs
            accelerator=global_args.accelerator,
            devices=(
                [i for i in range(global_args.num_devices)]
                if global_args.accelerator == "gpu"
                else global_args.num_devices
            ),
            num_nodes=global_args.num_nodes,
            strategy=global_args.strategy if distributed else "auto",
            precision=global_args.precision,
            # Training args
            max_epochs=(
                config["pretrain_config"][global_args.pretraining_task]["epochs"]
                if global_args.epochs < 1
                else global_args.epochs
            ),
            gradient_clip_val=None,
            logger=logger,
            callbacks=pretrain_callbacks,
            benchmark=True,
            default_root_dir=global_args.log_dir,
            # profiler="simple",
            barebones=global_args.barebones,
            num_sanity_val_steps=getattr(global_args, "num_sanity_val_steps", 2),
            # detect_anomaly=True,
            limit_train_batches=config["limit_train_batches"],
            limit_val_batches=config["limit_val_batches"],
        )

        if global_args.resume:
            print(
                f"Resuming training from trainer state: {global_args.encoder_weights}"
            )

        start_time = time.time()
        # This is the call to start training the model
        pretrainer.fit(
            pl_encoder,
            datamodule=pretrain_data_module,
            ckpt_path=global_args.encoder_weights if global_args.resume else None,
        )
        end_time = time.time()  # End time measurement
        print(f"Pretraining finished in {end_time - start_time} seconds")

        # # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
        # if (
        #     global_args.save_top_k > 0
        #     and global_args.pretrain
        #     and not global_args.barebones
        # ):
        #     pretrainer.test(datamodule=pretrain_data_module, ckpt_path="best")

    elif external_embedding_cache:
        pass
    elif global_args.encoder_weights:
        pl_encoder = safe_load_pretrain_wrapper(
            PRETRAIN_TASK_DICT[global_args.pretraining_task],
            global_args.encoder_weights,
            global_args=global_args,
            encoder=encoder,
            task_dict=config["pretrain_config"][global_args.pretraining_task],
        )
        print(f"Loading encoder checkpoint: {global_args.encoder_weights}")
    elif global_args.downstream_task in {"sqa", "chimericity", "oxidized_met", "retention_time", "metric_learning"}:
        # Global downstream tasks need pooling modules from the pretrain wrapper.
        pl_encoder = PRETRAIN_TASK_DICT[global_args.pretraining_task](
            encoder,
            global_args=global_args,
            task_dict=config["pretrain_config"][global_args.pretraining_task],
        )
        print("Warning: proceeding with untrained encoder")
    else:
        print("Warning: proceeding with untrained encoder")

    # ----------- Downstream Finetuning -----------
    # ---------------------------------------------

    # Load the selected pretrained encoder only when downstream finetuning follows.
    if global_args.pretrain and global_args.downstream_task != "none":
        ckpt_str = (
            "best_model_path"
            if global_args.downstream_encoder == "best"
            else "last_model_path"
        )
        encoder_path = pretrainer.checkpoint_callback.state_dict()[ckpt_str]
        if not encoder_path:
            raise ValueError(
                "No pretraining checkpoint path is available for downstream loading. "
                "Enable save_top_k/save_last or set downstream_task=none for pure pretraining."
            )
        encoder_ckpt = load_checkpoint_safely(encoder_path)
        pl_encoder.load_state_dict(encoder_ckpt["state_dict"])

    if global_args.downstream_task != "none":
        # Extract pretrained encoder nn.Module
        metric_pooler = None
        if external_embedding_cache:
            # The split-indexed cache is consumed by SpectralQualityAssessment
            # and its auxiliary subclasses; no pretraining wrapper is involved.
            pass
        elif global_args.downstream_task in {"sqa", "chimericity", "oxidized_met", "retention_time"}:
            encoder = pl_encoder.get_embedder(trainable=not global_args.freeze_encoder)
        elif global_args.downstream_task == "metric_learning":
            encoder = pl_encoder.get_encoder(trainable=not global_args.freeze_encoder)
            metric_pooler = getattr(getattr(pl_encoder, "teacher", None), "pooler", None)
            if metric_pooler is None:
                raise TypeError(
                    "metric_learning requires a DINO/DINOv2 wrapper exposing teacher.pooler."
                )
            if global_args.freeze_encoder:
                for parameter in metric_pooler.parameters():
                    parameter.requires_grad = False
            else:
                for parameter in metric_pooler.parameters():
                    parameter.requires_grad = True
        elif global_args.pretrain or global_args.encoder_weights:
            encoder = pl_encoder.get_encoder(trainable=not global_args.freeze_encoder)

        downstream_task_config = config["downstream_config"][
            global_args.downstream_task
        ]
        ds_monitor = downstream_task_config.get(
            "checkpoint_monitor", global_args.downstream_task + "_val_loss_epoch"
        )
        ds_monitor_mode = downstream_task_config.get("checkpoint_mode", "min")
        ds_callbacks = utils.configure_callbacks(
            global_args,
            downstream_task_config,
            ds_monitor,
            metric_mode=ds_monitor_mode,
        )

        _d_name = config["downstream_config"]["dataset_name"]
        # Load downstream dataset
        if global_args.downstream_task == "metric_learning":
            ds_data_module = utils.get_metric_learning_data_module(
                config["downstream_config"], global_args, _d_name, seed=global_args.seed
            )
            tokenizer = None
        elif _d_name in ("massivekb", "bacteria", "kitchensink_v4", "configurable_lance", "ninespecies_updated"):
            ds_data_module, tokenizer = utils.get_lance_peptide_data_module(
                config["downstream_config"],
                global_args,
                _d_name,
                seed=global_args.seed,
            )
        elif _d_name == "ninespecies_v2":
            ds_data_module, tokenizer = utils.get_ninespecies_v2_lance_data_module(
                config["downstream_config"],
                global_args,
                seed=global_args.seed,
            )
        elif _d_name == "sqa":
            ds_data_module = utils.get_sqa_inmem_data_module(
                config["downstream_config"],
                global_args.max_peaks,
                global_args,
                seed=global_args.seed,
            )
            tokenizer = None
        elif _d_name == "auxiliary_lance":
            ds_data_module = utils.get_auxiliary_lance_data_module(
                config["downstream_config"], global_args, seed=global_args.seed
            )
            tokenizer = None
        else:
            raise ValueError(f"Unknown downstream dataset_name: {_d_name!r}")
        ds_callbacks += lance_callbacks(ds_data_module)
        if global_args.downstream_task == "metric_learning":
            ds_callbacks += metric_learning_evaluation_callbacks(probing_config, global_args)

        # Define the task head. Metric learning has a fresh projection head,
        # rather than a sequence decoder or the DINO projection/prototype head.
        assert global_args.decoder_model, "argument decoder_model must be provided when downstream finetuning"
        if global_args.downstream_task == "metric_learning":
            if global_args.decoder_model != "metric_projection_head":
                raise ValueError("metric_learning requires decoder_model: metric_projection_head.")
            decoder = MetricProjectionHead(
                input_dim=encoder.running_units,
                hidden_dim=downstream_task_config.get("metric_hidden_dim"),
                output_dim=int(downstream_task_config.get("metric_embedding_dim", 128)),
            )
        elif global_args.downstream_task == "retention_time":
            prediction_mode = str(downstream_task_config.get("prediction_mode", "ordinal"))
            expected_decoder = (
                "linear_ordinal_head" if prediction_mode == "ordinal" else "linear_regression_head"
            )
            if prediction_mode not in {"ordinal", "regression"}:
                raise ValueError("retention_time prediction_mode must be ordinal or regression.")
            if global_args.decoder_model != expected_decoder:
                raise ValueError(
                    f"retention_time {prediction_mode} mode requires decoder_model: "
                    f"{expected_decoder}."
                )
            decoder = DECODER_DICT[global_args.decoder_model](
                tokenizer,
                d_model=encoder.running_units,
                num_classes=(
                    int(downstream_task_config["soft_ordinal"]["n_bins"])
                    if prediction_mode == "ordinal"
                    else 1
                ),
            )
        else:
            decoder = DECODER_DICT[global_args.decoder_model](
                tokenizer,
                d_model=encoder.running_units,
                dropout=config["downstream_config"][global_args.downstream_task]["decoder_dropout"],
                cross_attend=global_args.cross_attend,
                max_seq_len=int(downstream_task_config.get("max_length", global_args.max_length)) + 1,
                max_charge=global_args.max_charge,
            )

        downstream_kwargs = {
            "global_args": global_args,
            "tokenizer": tokenizer,
            "task_dict": config["downstream_config"][global_args.downstream_task],
        }
        if global_args.downstream_task == "metric_learning":
            downstream_kwargs["pooler"] = metric_pooler
        pl_downstream = DOWNSTREAM_TASK_DICT[global_args.downstream_task](
            encoder, decoder, **downstream_kwargs
        )

        if run is not None and utils.get_rank() == 0:
            if global_args.watch_model:
                run.watch(pl_downstream, log="all")
            run.log({"num_parameters_decoder": utils.get_num_parameters(decoder)})

        if global_args.downstream_weights:
            print(
                f"Loading downstream weights from previous checkpoint: {global_args.downstream_weights}"
            )
            downstream_ckpt = load_checkpoint_safely(
                global_args.downstream_weights, map_location=pl_downstream.device
            )
            downstream_state = adapt_historical_pairwise_state_dict(downstream_ckpt["state_dict"])
            pl_downstream.load_state_dict(downstream_state, strict=True)

        else:
            if global_args.pretrain or global_args.encoder_weights:
                print(
                    "Downstream task weights not provided; fine-tuning from "
                    "pretrained encoder with randomly initialized decoder"
                )
            else:
                print(f"Downstream training from scratch")

        (
            print(
                f"Starting distributed downstream finetuning using {global_args.num_devices} devices on {global_args.num_nodes} node(s)"
            )
            if distributed
            else print("Starting single-device training")
        )
        ds_trainer = pl.Trainer(
            # Distributed kwargs
            accelerator=global_args.accelerator,
            devices=(
                [i for i in range(global_args.num_devices)]
                if global_args.accelerator == "gpu"
                else global_args.num_devices
            ),
            num_nodes=global_args.num_nodes,
            strategy=global_args.strategy if distributed else "auto",
            precision=global_args.precision,
            # Training args
            max_epochs=(
                config["downstream_config"][global_args.downstream_task]["epochs"]
                if global_args.epochs < 1
                else global_args.epochs
            ),
            gradient_clip_val=downstream_task_config.get(
                "gradient_clip_val", global_args.clip_grad
            ),
            logger=logger,
            callbacks=ds_callbacks,
            benchmark=True,
            default_root_dir=global_args.log_dir,
            # profiler="simple",
            # profiler="advanced",
            barebones=global_args.barebones,
            num_sanity_val_steps=0 if global_args.barebones else 2,
            max_steps=global_args.max_steps if global_args.max_steps > 0 else -1,
            limit_train_batches=config["limit_train_batches"],
            limit_val_batches=config["limit_val_batches"],
            limit_test_batches=global_args.limit_test_batches,
            check_val_every_n_epoch=config["validate_every_n_epochs"],
        )

        if global_args.eval_only:
            print("'--eval_only' specified - skipping training")
            pl_downstream.cheap_val = False  # toggle off cheap validation loss
            if global_args.validate_on_end:
                ds_trainer.validate(pl_downstream, datamodule=ds_data_module)
            if global_args.test_on_end:
                ds_trainer.test(pl_downstream, datamodule=ds_data_module)
        else:
            if global_args.resume:
                print(
                    f"Resuming training from trainer state: {global_args.downstream_weights}"
                )
            start_time = time.time()
            # This is the call to start training the model
            ds_trainer.fit(
                pl_downstream,
                datamodule=ds_data_module,
                ckpt_path=(
                    global_args.downstream_weights if global_args.resume else None
                ),
            )
            end_time = time.time()  # End time measurement
            print(f"Downstream finetuning finished in {end_time - start_time} seconds")

            pl_downstream.cheap_val = False  # toggle off cheap validation loss

            # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
            if not global_args.barebones:
                if global_args.validate_on_end:
                    ds_trainer.validate(
                        datamodule=ds_data_module,
                        ckpt_path="best" if global_args.save_top_k > 0 else None,
                    )
                if global_args.test_on_end:
                    ds_trainer.test(
                        datamodule=ds_data_module,
                        ckpt_path="best" if global_args.save_top_k > 0 else None,
                    )

    # Flag the run as finished to the wandb server
    if run is not None and utils.get_rank() == 0:
        wandb.finish()

    if global_args.remove_ckpt:
        shutil.rmtree(global_args.output_dir, ignore_errors=True)


if __name__ == "__main__":
    # parse args
    global_args, pretrain_config, ds_config, probing_config = parse_args_and_config()
    # create output dirs on main process
    create_output_dirs(global_args, is_main_process=utils.get_rank() == 0)
    # A100 specific setting
    if global_args.matmul_precision:
        torch.set_float32_matmul_precision(global_args.matmul_precision)
    if global_args.disable_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("Disabled cuDNN SDPA; PyTorch will select another attention backend")
    # set seed
    if global_args.seed is not None:
        torch.manual_seed(global_args.seed)
        np.random.seed(global_args.seed)
        random.seed(global_args.seed)
        pl.seed_everything(global_args.seed)
    else:
        print("JL - Using a random seed")
    # run
    main(global_args, pretrain_config, ds_config, probing_config)
