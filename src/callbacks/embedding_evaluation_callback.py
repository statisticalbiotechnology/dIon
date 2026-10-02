"""Online peptide retrieval evaluation without changing the linear probe callback."""

from __future__ import annotations

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.embed_eval.evaluator import evaluate_embedder, flatten_report_for_logging


class SpectrumEmbeddingEvaluationCallback(Callback):
    """Run config-driven global embedding metrics at selected probe events."""

    def __init__(
        self,
        evaluation_config: dict,
        global_args,
        precursor_conditioning: str = "conditioned",
        metric_prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = evaluation_config
        online_config = evaluation_config.get("online", {})
        self.enabled = bool(online_config.get("enabled", False))
        self.on_fit_start_enabled = bool(
            online_config.get("on_fit_start", global_args.probe_on_fit_start)
        )
        self.every_n_steps = int(
            online_config.get("every_n_steps", global_args.probe_every_n_steps)
        )
        if self.enabled and self.every_n_steps < 1:
            raise ValueError("embedding evaluation every_n_steps must be >= 1.")
        self.global_args = global_args
        self.precursor_conditioning = precursor_conditioning
        self.metric_prefix = metric_prefix.strip("/")
        self._last_step = None

    def _evaluation_configs(self) -> list[dict]:
        """Return one config or the compact datasets in an online suite."""
        configs = self.config.get("evaluations")
        if configs is None:
            return [self.config]
        if not isinstance(configs, list) or not configs:
            raise ValueError("embedding evaluation suite must contain evaluations.")
        return configs

    @rank_zero_only
    def _run(self, trainer, pl_module) -> None:
        embedder = (
            None
            if self.global_args.embedding_baseline == "precursor_metadata"
            else pl_module.get_embedder(trainable=False)
        )
        reports = [
            evaluate_embedder(
                embedder,
                self.global_args,
                evaluation_config,
                mode="online",
                device=pl_module.device,
                precursor_conditioning=self.precursor_conditioning,
            )
            for evaluation_config in self._evaluation_configs()
        ]
        metrics = {}
        for report in reports:
            metrics.update(flatten_report_for_logging(report, macro_only=True))
        if len(reports) > 1:
            values = [
                float(report["macro"]["map"])
                for report in reports
                if isinstance(report.get("macro", {}).get("map"), float)
            ]
            if values:
                metrics["embed_eval/all_validation/broad_map"] = sum(values) / len(values)
        if self.metric_prefix:
            metrics = {f"{self.metric_prefix}/{name}": value for name, value in metrics.items()}
        trainer.logger.log_metrics(metrics, step=int(trainer.global_step))

    def on_fit_start(self, trainer, pl_module) -> None:
        if not self.enabled or not self.on_fit_start_enabled:
            return
        self._run(trainer, pl_module)
        pl_module.train()
        self._last_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        del outputs, batch, batch_idx
        if not self.enabled:
            return
        step = int(trainer.global_step)
        if step < 1 or step % self.every_n_steps != 0 or self._last_step == step:
            return
        self._run(trainer, pl_module)
        pl_module.train()
        self._last_step = step
