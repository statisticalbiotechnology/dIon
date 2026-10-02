from __future__ import annotations

import csv
import os

import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class PredictionCsvOutputCallback(pl.Callback):
    """Stream canonical-indexed de novo predictions and reject invalid coverage."""

    fields = [
        "canonical_index", "spectrum_id", "peak_file", "scan_id", "title",
        "species", "split", "source_row_index", "precursor_mz", "precursor_charge",
        "isolation_low", "isolation_high", "true_sequence", "predicted_sequence",
        "peptide_confidence", "no_prediction",
    ]

    def __init__(self, log_dir, global_args):
        self.output_file = os.path.join(log_dir, "predictions.csv")
        self.global_args = global_args
        self.handle = None
        self.writer = None
        self.seen = set()

    @rank_zero_only
    def setup(self, trainer, pl_module, stage=None):
        self.handle = open(self.output_file, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        self.writer.writeheader()

    @staticmethod
    def _gather_outputs(outputs, pl_module):
        if pl_module.trainer.world_size == 1:
            return outputs
        gathered = [None] * pl_module.trainer.world_size
        dist.all_gather_object(gathered, outputs)
        return {
            "predictions": [item for rank in gathered for item in rank["predictions"]],
            "peptides_true": [item for rank in gathered for item in rank["peptides_true"]],
        }

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        gathered = self._gather_outputs(outputs, pl_module)
        if not trainer.is_global_zero:
            return
        for prediction, truth in zip(
            gathered["predictions"], gathered["peptides_true"], strict=True
        ):
            index = int(prediction["canonical_index"])
            if index in self.seen:
                raise RuntimeError(f"Duplicate canonical prediction index: {index}")
            self.seen.add(index)
            peptide = prediction["peptide"]
            no_prediction = not peptide
            predicted = "".join(peptide) if isinstance(peptide, list) else str(peptide)
            true = "".join(truth) if isinstance(truth, list) else str(truth)
            peak_file = str(prediction.get("peak_file", ""))
            scan_id = prediction.get("scan_id", "")
            title = str(prediction.get("title", ""))
            self.writer.writerow({
                "canonical_index": index,
                "spectrum_id": title if title and title != "unknown" else f"{peak_file}:{scan_id}",
                "peak_file": peak_file, "scan_id": scan_id, "title": title,
                "species": prediction.get("species", ""),
                "split": prediction.get("split", ""),
                "source_row_index": prediction.get("source_row_index", ""),
                "precursor_mz": prediction.get("precursor_mz", ""),
                "precursor_charge": prediction.get("precursor_charge", ""),
                "isolation_low": prediction.get("isolation_low", ""),
                "isolation_high": prediction.get("isolation_high", ""),
                "true_sequence": true, "predicted_sequence": predicted,
                "peptide_confidence": "" if no_prediction else prediction["peptide_score"],
                "no_prediction": str(no_prediction).lower(),
            })
        self.handle.flush()

    @rank_zero_only
    def teardown(self, trainer, pl_module, stage=None):
        if self.handle is None:
            return
        if len(self.seen) < 1000:
            self.handle.close()
            self.handle = None
            return
        expected = len(trainer.datamodule.test_dataset)
        missing = [index for index in range(expected) if index not in self.seen]
        if missing:
            raise RuntimeError(
                f"Incomplete canonical prediction coverage: {len(missing)} missing; "
                f"first={missing[:10]}"
            )
        self.handle.close()
        self.handle = None
