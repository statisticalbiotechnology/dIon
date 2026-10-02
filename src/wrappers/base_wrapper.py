import torch
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from collections import deque


class RunningWindowLoss:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.loss_window = deque(maxlen=window_size)

    def update(self, loss):
        self.loss_window.append(loss)

    def get_running_loss(self):
        return (
            sum(self.loss_window) / len(self.loss_window) if self.loss_window else 0.0
        )


class BestMetricTracker:
    def __init__(self):
        self.best_metrics = {}
        self.optimization_direction = (
            {}
        )  # Tracks the optimization direction for each metric

    def update_metric(self, name, value, maximize=True):
        if name not in self.best_metrics or self._is_better(
            value, self.best_metrics[name], maximize
        ):
            self.best_metrics[name] = value
            self.optimization_direction[name] = maximize

    def get_best_metric(self, name):
        return self.best_metrics.get(name, None)

    def _is_better(self, new_value, old_value, maximize):
        if maximize:
            return new_value > old_value
        else:
            return new_value < old_value


class BasePLWrapper(ABC, pl.LightningModule):
    """
    PyTorch Lightning module wrapping the encoder and the head for unsupervised pretraining.

    This wrapper class provides integration with PyTorch Lightning's train, val and test loop.

    Args:
        encoder (pl.LightningModule): The encoder model
        head (pl.LightningModule): The objective-specific projection head for the encoder
        args (argparse.Namespace): Command-line arguments or a configuration namespace
            containing hyperparameters and settings.
        datasets (list): A list containing the train, val, and test datasets.
    """

    def __init__(
        self,
        encoder,
        global_args,
        head=None,
        collate_fn=None,
        task_dict=None,
    ):
        super().__init__()
        self.TASK_NAME = ""
        self.encoder = encoder
        self.head = head
        self.collate_fn = collate_fn
        self.task_dict = task_dict
        self.batch_size = (
            task_dict["batch_size"]
            if global_args.batch_size < 0
            else global_args.batch_size
        )

        self.pin_mem = global_args.pin_mem

        self.weight_decay = task_dict["weight_decay"]

        self.eff_batch_size = (
            self.batch_size
            * global_args.accum_iter
            * global_args.num_devices
            * global_args.num_nodes
        )

        self.scale_lr_by_batchsize = global_args.scale_lr_by_batchsize
        base_lr = task_dict.get("head_lr", task_dict.get("learning_rate", task_dict.get("blr")))
        if base_lr is None:
            raise KeyError("Task configuration must define head_lr, learning_rate, or blr.")
        if self.scale_lr_by_batchsize:
            self.ref_batch_size = task_dict.get("ref_batch_size", 0)
            assert (
                self.ref_batch_size > 0
            ), "'ref_batch_size' must be provided when 'scale_lr_by_batchsize' is True"
            self.lr = base_lr * self.eff_batch_size / self.ref_batch_size
        else:
            self.lr = base_lr
        self.mask_zero_tokens = global_args.mask_zero_tokens
        self.log_wandb = global_args.log_wandb
        self.tracker = BestMetricTracker()
        self.best_metrics_logged = (
            False  # keep track of if the best achieved metrics have been logged
        )
        self.running_train_loss = RunningWindowLoss()
        self.running_val_loss = RunningWindowLoss()
        self.automatic_optimization = False
        self.max_charge = global_args.max_charge
        self.grad_clip_val = global_args.clip_grad

    def _parse_batch(self, batch):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        batch_size = mz_arr.shape[0]
        return (
            mzab,
            {
                "mass": spectra["precursor_mass"] if self.use_mass else None,
                "charge": spectra["precursor_charge"] if self.use_charge else None,
            },
        ), batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, input_dict, target = parsed_batch
        outs = self.encoder(mzab, **input_dict, **kwargs)
        if self.head is not None:
            outs = self.head(outs)
        return outs

    def eval_forward(self, parsed_batch, **kwargs):
        return self.forward(parsed_batch, **kwargs)

    def _get_losses(self, returns, parsed_batch):
        # example
        # preds = returns["preds"]
        # loss = self.loss_fn(preds, labels)
        # return loss
        pass

    @abstractmethod
    def _get_train_stats(self, returns, parsed_batch):
        # define what metrics to log during training steps
        # note: must return the differentiable loss
        stats = {}
        # example

        input, labels = batch
        loss = self._get_losses(returns, labels)
        # stats["loss"] = loss
        # metrics = calc_classification_metrics(returns["preds"], labels)
        # stats = {**stats, **metrics}

        # stats = {"train_" + key: val.detach().item() for key, val in stats.items()}
        return loss, stats

    @abstractmethod
    def _get_eval_stats(self, returns, parsed_batch, split=None):
        # define what metrics to log during val/test steps
        stats = {}
        # example
        # input, labels = batch
        # loss = self._get_losses(returns, labels)
        # stats["loss"] = loss
        # metrics = calc_classification_metrics(returns["preds"], labels)
        # stats = {**stats, **metrics}
        return stats

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        if hasattr(opts, "__iter__"):
            for opt in opts:
                opt.zero_grad()
        else:
            opts.zero_grad()

        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.forward(parsed_batch, batch_idx=batch_idx)
        loss, train_stats = self._get_train_stats(returns, parsed_batch)

        prefix = "train_"
        if self.TASK_NAME:
            prefix = self.TASK_NAME + "_" + prefix

        train_stats = {
            prefix
            + key: val.detach().cpu().item() if isinstance(val, torch.Tensor) else val
            for key, val in train_stats.items()
        }
        self.log_dict(
            {**train_stats, "lr": self.lr},
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.running_train_loss.update(train_stats[prefix + "loss"])
        self.log(
            prefix + "run_loss",
            self.running_train_loss.get_running_loss(),
            prog_bar=True,
            sync_dist=True,
        )

        self.manual_backward(loss)

        # gradient clipping
        if hasattr(opts, "__iter__"):
            for opt in opts:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.grad_clip_val,
                    gradient_clip_algorithm="norm",
                )
                opt.step()
        else:
            self.clip_gradients(
                opts,
                gradient_clip_val=self.grad_clip_val,
                gradient_clip_algorithm="norm",
            )
            opts.step()

        return {"loss": loss, "returns": returns}

    def validation_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch, Eval=True)
        returns = self.eval_forward(parsed_batch, batch_idx=batch_idx)
        val_stats = self._get_eval_stats(returns, parsed_batch, split="val")

        prefix = "val_"
        if self.TASK_NAME:
            prefix = self.TASK_NAME + "_" + prefix

        val_stats = {
            prefix
            + key: val.detach().cpu().item() if isinstance(val, torch.Tensor) else val
            for key, val in val_stats.items()
        }

        self.log_dict(
            {**val_stats},
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        if prefix + "loss" in val_stats:
            self.running_val_loss.update(val_stats[prefix + "loss"])
            self.log(
                "run_val_loss",
                self.running_val_loss.get_running_loss(),
                prog_bar=True,
                sync_dist=True,
            )
        return {"val_stats": val_stats, "returns": returns}

    def test_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch, Eval=True)
        returns = self.eval_forward(parsed_batch, batch_idx=batch_idx)
        test_stats = self._get_eval_stats(returns, parsed_batch, split="test")

        prefix = "test_"
        if self.TASK_NAME:
            prefix = self.TASK_NAME + "_" + prefix

        test_stats = {
            prefix + key: val.detach().item() if isinstance(val, torch.Tensor) else val
            for key, val in test_stats.items()
        }
        self.log_dict(
            {**test_stats}, on_epoch=True, batch_size=batch_size, sync_dist=True
        )
        return {"test_stats": test_stats, "returns": returns}

    def configure_optimizers(self):
        opts = [
            torch.optim.AdamW(
                self.encoder.parameters(),
                lr=self.lr,
                betas=(0.9, 0.9999),
                weight_decay=self.weight_decay,
            ),
        ]
        if self.head:
            opts.append(
                torch.optim.AdamW(
                    self.head.parameters(),
                    lr=self.lr,
                    betas=(0.9, 0.9999),
                    weight_decay=self.weight_decay,
                ),
            )
        return opts

    def _get_padding_mask(self, sequences: torch.Tensor, seq_lengths: torch.Tensor):
        """
        sequences.shape = (batch_size, max_sequence_len, dim)
        seq_lengths.shape = (batch_size, 1)
        """
        assert len(sequences.shape) == 3
        assert len(seq_lengths.shape) == 2
        all_inds = (
            torch.arange(sequences.shape[1], device=sequences.device)
            .unsqueeze(0)
            .repeat((sequences.shape[0], 1))
        )
        return all_inds >= seq_lengths

    def _update_best_metrics(self, metrics, prefix="val_", suffix="_epoch"):
        """Override this to track the best achieved value for additional metrics beyond 'loss'"""
        # validation loss
        self.tracker.update_metric(
            "best_" + prefix + "loss" + suffix,
            metrics[prefix + "loss" + suffix].detach().cpu().item(),
            maximize=False,
        )

    def _get_metric_prefix_suffix(self, prefix="val_", suffix="_epoch"):
        if self.TASK_NAME:
            prefix = self.TASK_NAME + "_" + prefix
        return prefix, suffix

    def on_validation_epoch_end(self):
        # Update the current best achieved value for each val metric
        # Get the per-epoch metric value from the logged metrics
        prefix, suffix = self._get_metric_prefix_suffix()

        cur_epoch = self.trainer.current_epoch
        if self.global_rank == 0:  # Only log on master process
            if cur_epoch > 0:
                metrics = self.trainer.logged_metrics

                self._update_best_metrics(metrics, prefix, suffix)

            # TODO: verify: don't think this part is needed bc of the "on_train_end"
            # at the last epoch, log the best metrics
            if cur_epoch == self.trainer.max_epochs - 1:
                self.log_dict(self.tracker.best_metrics)
                self.best_metrics_logged = True

    def on_train_epoch_end(self):  # log the learning rate
        opts = self.optimizers()
        if hasattr(opts, "__iter__"):
            cur_lr_enc = opts[0].param_groups[0]["lr"]
            self.log("lr_enc", cur_lr_enc, on_step=False, on_epoch=True)
            cur_lr_head = opts[1].param_groups[0]["lr"]
            self.log("lr_head", cur_lr_head, on_step=False, on_epoch=True)
        else:
            cur_lr = opts.param_groups[0]["lr"]
            frozen_downstream_encoder = (
                hasattr(self, "decoder")
                and not any(parameter.requires_grad for parameter in self.encoder.parameters())
            )
            self.log(
                "lr_head" if frozen_downstream_encoder else "lr_enc",
                cur_lr,
                on_step=False,
                on_epoch=True,
            )

    def _wandb_experiment(self):
        if not self.log_wandb or self.logger is None:
            return None
        experiment = getattr(self.logger, "experiment", None)
        if not hasattr(experiment, "log"):
            return None
        return experiment

    def on_train_end(self):
        experiment = self._wandb_experiment()
        if not self.best_metrics_logged and experiment is not None:
            experiment.log(self.tracker.best_metrics)

    def on_fit_start(self):
        experiment = self._wandb_experiment()
        if experiment is not None:
            experiment.log(
                {
                    "eff_batch_size_" + self.TASK_NAME: self.eff_batch_size,
                    "ref_batch_size_"
                    + self.TASK_NAME: (
                        self.ref_batch_size if self.scale_lr_by_batchsize else None
                    ),
                }
            )

    def get_encoder(
        self, trainable: bool = True
    ):
        """Return the encoder for downstream use."""
        if trainable:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        return self.encoder

    def get_embedder(
        self, trainable: bool = True
    ):
        """Return an embedding module for downstream use."""
        del trainable
        raise NotImplementedError(
            "This pretraining wrapper has no 'embedder' defined. "
        )

    def _mzab_array(self, batch):
        mz_arr = batch["mz_array"]
        int_arr = batch["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        return mzab


class BaseDownstreamWrapper(BasePLWrapper):
    def __init__(self, encoder, decoder, global_args, collate_fn=None, task_dict=None):
        super().__init__(encoder, global_args, None, collate_fn, task_dict)
        self.decoder = decoder
        self.layer_decay = task_dict.get("layer_decay", None)
        self.label_smoothing = self.task_dict.get("label_smoothing", 0)

    def configure_optimizers(self):
        encoder_params = list(self.encoder.parameters())
        opts = []

        if len(encoder_params) > 0:
            if self.layer_decay is not None:
                encoder_param_groups = self.create_encoder_param_groups(self.encoder)
            else:
                encoder_param_groups = self.encoder.parameters()
            encoder_opt = torch.optim.Adam(
                encoder_param_groups,
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
            opts.append(encoder_opt)

        opts.append(
            torch.optim.Adam(
                self.decoder.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            ),
        )
        return opts

    def create_encoder_param_groups(self, encoder: torch.nn.Module):
        """Layer-wise learning rate decay following MAE:
        https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py"""
        layer_decay = self.layer_decay if self.layer_decay is not None else 1.0

        num_layers = encoder.n_layers
        layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers)]

        param_group_names = {}
        param_groups = {}

        for n, p in encoder.named_parameters():
            if not p.requires_grad:
                continue

            layer_id = self.encoder.get_layer_id(n)

            if n not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_groups[n] = {
                    "lr_scale": this_scale,
                    "params": [],
                }

            param_groups[n]["params"].append(p)

        return list(param_groups.values())
