from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import os.path as path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DistributedSampler

from src.data.lance_datasets import SafeLanceDataset
from src.data.peptide_identity_batch_sampler import PeptideIdentityBatchSampler
from src.data.lance_loader_utils import infer_rank_world_size, make_lance_safe_loader


def _tensorize(value, dtype: torch.dtype = torch.float32):
    """Convert homogeneous Lance list values to tensors before collation."""
    if not isinstance(value, list):
        return value
    try:
        return torch.tensor(value, dtype=dtype)
    except (TypeError, ValueError):
        return value


def _rows_to_batch_dict(rows: List[dict], collate_fn: Callable):
    return collate_fn([{key: _tensorize(value) for key, value in row.items()} for row in rows])


def _num_batches(num_samples: int, batch_size: int, drop_last: bool) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    return num_samples // batch_size if drop_last else (num_samples + batch_size - 1) // batch_size



class ExactDistributedEvalSampler(DistributedSampler):
    """Shard evaluation indices without DistributedSampler padding repeats."""

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        quotient, remainder = divmod(len(self.dataset), self.num_replicas)
        return quotient + int(self.rank < remainder)

class ReplaySampledDataset(Dataset):
    """Primary data followed by a deterministic fresh replay sample each epoch."""

    def __init__(self, primary: Dataset, replay: Dataset, replay_ratio: float = 1.0, seed: int = 0):
        if replay_ratio <= 0:
            raise ValueError("replay_ratio must be > 0 when replay is enabled.")
        if len(replay) == 0:
            raise ValueError("Replay dataset must not be empty.")
        self.primary = primary
        self.replay = replay
        self.replay_ratio = float(replay_ratio)
        self.seed = int(seed)
        self.replay_size = max(1, int(round(len(primary) * self.replay_ratio)))
        self._replay_indices = np.empty(0, dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch))
        self._replay_indices = rng.choice(
            len(self.replay),
            size=self.replay_size,
            replace=self.replay_size > len(self.replay),
        ).astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.primary) + self.replay_size

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        if index < len(self.primary):
            return self.primary[index]
        return self.replay[int(self._replay_indices[index - len(self.primary)])]

    def __getitems__(self, indices: List[int]) -> list[dict]:
        indices = [int(index) for index in indices]
        out = [None] * len(indices)
        primary_positions, primary_indices = [], []
        replay_positions, replay_indices = [], []
        n_primary = len(self.primary)
        for position, index in enumerate(indices):
            if index < n_primary:
                primary_positions.append(position)
                primary_indices.append(index)
            else:
                replay_positions.append(position)
                replay_indices.append(int(self._replay_indices[index - n_primary]))
        if primary_indices:
            for position, item in zip(primary_positions, self.primary.__getitems__(primary_indices)):
                out[position] = item
        if replay_indices:
            for position, item in zip(replay_positions, self.replay.__getitems__(replay_indices)):
                out[position] = item
        return out


class LanceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[Path],
        batch_size: int,
        collate_fn: Callable,
        seed: int = 0,
        eval_batch_size: Optional[int] = None,
        include_test: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = True,
        columns: Optional[List[str]] = None,
        expected_world_size: Optional[int] = None,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        extra_val_paths: Optional[Dict[str, str]] = None,
        replay_train_path: Optional[str] = None,
        replay_ratio: float = 0.0,
        positive_batching: Optional[Dict[str, object]] = None,
        include_index: bool = False,
        test_sequential: bool = False,
        exact_eval_sharding: bool = False,
    ):
        super().__init__()
        root = Path(data_dir) if data_dir is not None else None

        def resolve_path(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            candidate = Path(value)
            if not candidate.is_absolute() and root is not None:
                candidate = root / candidate
            return str(candidate)

        explicit_paths = any(value is not None for value in (train_path, val_path, test_path))
        if explicit_paths:
            if train_path is None or val_path is None:
                raise ValueError("Explicit Lance data requires both train_path and val_path.")
            lance_paths = [resolve_path(train_path), resolve_path(val_path)]
            if include_test and test_path is not None:
                lance_paths.append(resolve_path(test_path))
        elif root is not None:
            splits = ["train", "val", "test"] if include_test else ["train", "val"]
            subdirs = [path.join(str(root), split) for split in splits]
            subdirs_exist = all(path.exists(path.join(subdir, "indexed.lance")) for subdir in subdirs)
            lance_subsets_exist = all(path.exists(subdir + ".lance") for subdir in subdirs)
            if not (subdirs_exist or lance_subsets_exist):
                raise AssertionError(
                    f'Expected split Lance datasets under data_dir: {data_dir}'
                )
            lance_paths = [path.join(subdir, "indexed.lance") for subdir in subdirs] if subdirs_exist else [subdir + ".lance" for subdir in subdirs]
        else:
            raise ValueError("data_dir or explicit train/val Lance paths are required.")

        self.lance_paths = lance_paths
        self.replay_train_path = resolve_path(replay_train_path)
        self.replay_ratio = float(replay_ratio or 0.0)
        self.positive_batching = dict(positive_batching or {})
        if self.positive_batching and self.replay_train_path:
            raise ValueError("Positive-aware batching is incompatible with replay_train_path.")
        if self.positive_batching:
            required = {"label_column", "peptides_per_batch", "spectra_per_peptide"}
            missing = sorted(required - set(self.positive_batching))
            if missing:
                raise ValueError(f"positive_batching is missing required keys: {missing}")
            implied_batch_size = int(self.positive_batching["peptides_per_batch"]) * int(
                self.positive_batching["spectra_per_peptide"]
            )
            if implied_batch_size != int(batch_size):
                raise ValueError(
                    "positive_batching peptides_per_batch * spectra_per_peptide must equal batch_size."
                )
        self.extra_val_paths = []
        for name, value in (extra_val_paths or {}).items():
            resolved = resolve_path(value)
            if resolved is None:
                raise ValueError(f"Extra validation path for {name!r} is empty.")
            self.extra_val_paths.append((str(name), resolved))
        self.val_dataloader_names = ["val"] + [name for name, _ in self.extra_val_paths]
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size) if eval_batch_size is not None else int(batch_size)
        self.collate_fn = partial(_rows_to_batch_dict, collate_fn=collate_fn)
        self.seed = int(seed)
        self.num_workers = max(0, int(num_workers))
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.drop_last = bool(drop_last)
        self.columns = columns
        self.include_index = bool(include_index)
        self.exact_eval_sharding = bool(exact_eval_sharding)
        self.expected_world_size = expected_world_size
        self.test_sequential = bool(test_sequential)
        self.primary_train_dataset: Optional[SafeLanceDataset] = None
        self.replay_dataset: Optional[SafeLanceDataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[SafeLanceDataset] = None
        self.extra_val_datasets: List[SafeLanceDataset] = []
        self.test_dataset: Optional[SafeLanceDataset] = None
        self._train_sampler: Optional[DistributedSampler] = None
        self._positive_train_batch_sampler: Optional[PeptideIdentityBatchSampler] = None
        self._positive_eval_batch_samplers: dict[str, PeptideIdentityBatchSampler] = {}
        self._epoch = 0

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.primary_train_dataset = SafeLanceDataset(self.lance_paths[0], columns=self.columns, include_index=self.include_index)
            if self.replay_train_path and self.replay_ratio > 0:
                self.replay_dataset = SafeLanceDataset(self.replay_train_path, columns=self.columns)
                self.train_dataset = ReplaySampledDataset(
                    self.primary_train_dataset, self.replay_dataset, self.replay_ratio, self.seed
                )
            else:
                self.train_dataset = self.primary_train_dataset
            self.val_dataset = SafeLanceDataset(self.lance_paths[1], columns=self.columns, include_index=self.include_index)
            self.extra_val_datasets = [SafeLanceDataset(value, columns=self.columns, include_index=self.include_index) for _, value in self.extra_val_paths]
        if stage in ("validate", None):
            self.val_dataset = SafeLanceDataset(self.lance_paths[1], columns=self.columns, include_index=self.include_index)
            self.extra_val_datasets = [SafeLanceDataset(value, columns=self.columns, include_index=self.include_index) for _, value in self.extra_val_paths]
        if stage in ("test", None) and len(self.lance_paths) > 2:
            self.test_dataset = SafeLanceDataset(self.lance_paths[2], columns=self.columns, include_index=self.include_index)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(self._epoch)
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self._epoch)
        if self._positive_train_batch_sampler is not None:
            self._positive_train_batch_sampler.set_epoch(self._epoch)
        # Metric validation is a fixed seeded draw. Advancing it with the
        # train epoch would turn validation loss into a moving early-stop and
        # checkpoint-selection target.

    def _positive_batch_sampler(self, dataset: SafeLanceDataset, *, train: bool) -> PeptideIdentityBatchSampler:
        if not self.positive_batching:
            raise RuntimeError("Positive batch sampler requested without positive_batching configuration.")
        cache = self._positive_train_batch_sampler if train else self._positive_eval_batch_samplers.get(dataset.uri)
        if cache is None:
            rank, world_size = infer_rank_world_size(expected_world_size=self.expected_world_size)
            batches_key = "train_batches_per_epoch" if train else "validation_batches_per_epoch"
            cache = PeptideIdentityBatchSampler(
                dataset.uri,
                label_column=str(self.positive_batching["label_column"]),
                peptides_per_batch=int(self.positive_batching["peptides_per_batch"]),
                spectra_per_peptide=int(self.positive_batching["spectra_per_peptide"]),
                seed=self.seed + (0 if train else 10_000_019),
                rank=rank,
                world_size=world_size,
                batches_per_epoch=self.positive_batching.get(batches_key),
                precursor_mz_column=str(self.positive_batching.get("precursor_mz_column", "precursor_mz")),
                precursor_charge_column=str(self.positive_batching.get("precursor_charge_column", "precursor_charge")),
                strict_ppm=float(self.positive_batching.get("strict_ppm", 10.0)),
                relaxed_ppm_windows=self.positive_batching.get("relaxed_ppm_windows", (50.0, 250.0, 1000.0, 5000.0, 20_000.0)),
                diagnostics_reservoir_size=int(self.positive_batching.get("diagnostics_reservoir_size", 100_000)),
            )
            if train:
                self._positive_train_batch_sampler = cache
            else:
                self._positive_eval_batch_samplers[dataset.uri] = cache
        cache.set_epoch(self._epoch)
        return cache

    def _make_train_sampler(self) -> DistributedSampler:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup() first.")
        rank, world_size = infer_rank_world_size(expected_world_size=self.expected_world_size)
        return DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=self.seed, drop_last=self.drop_last,
        )

    def num_train_batches(self) -> int:
        if self.positive_batching:
            if not isinstance(self.primary_train_dataset, SafeLanceDataset):
                raise TypeError("Positive-aware batching requires a SafeLanceDataset.")
            return len(self._positive_batch_sampler(self.primary_train_dataset, train=True))
        return _num_batches(len(self._make_train_sampler()), self.batch_size, self.drop_last)

    def metric_learning_sampler_diagnostics(self) -> dict[str, float]:
        if self._positive_train_batch_sampler is None:
            return {}
        return self._positive_train_batch_sampler.diagnostics()

    def train_dataloader(self):
        if self.positive_batching:
            if not isinstance(self.train_dataset, SafeLanceDataset):
                raise TypeError("Positive-aware batching requires a SafeLanceDataset.")
            return make_lance_safe_loader(
                dataset=self.train_dataset,
                batch_sampler=self._positive_batch_sampler(self.train_dataset, train=True),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                use_safe=True,
            )
        sampler = self._make_train_sampler()
        sampler.set_epoch(0)
        self._train_sampler = sampler
        return make_lance_safe_loader(
            dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            drop_last=self.drop_last, pin_memory=self.pin_memory, sampler=sampler,
            persistent_workers=self.persistent_workers, collate_fn=self.collate_fn,
            use_safe=isinstance(self.train_dataset, SafeLanceDataset),
        )

    def _make_eval_loader(self, dataset: SafeLanceDataset):
        if self.positive_batching:
            return make_lance_safe_loader(
                dataset=dataset,
                batch_sampler=self._positive_batch_sampler(dataset, train=False),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                use_safe=True,
            )
        return self._make_standard_eval_loader(dataset)

    def _make_standard_eval_loader(self, dataset: SafeLanceDataset):
        """Evaluate every row once, without positive-aware SupCon batching."""
        rank, world_size = infer_rank_world_size(expected_world_size=self.expected_world_size)
        sampler = None
        if world_size > 1:
            sampler = (
                ExactDistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
                if self.exact_eval_sharding
                else DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            )
        return make_lance_safe_loader(
            dataset=dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers,
            drop_last=False, pin_memory=self.pin_memory, sampler=sampler,
            persistent_workers=self.persistent_workers, collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup() first.")
        loaders = [self._make_eval_loader(self.val_dataset)]
        loaders.extend(self._make_eval_loader(dataset) for dataset in self.extra_val_datasets)
        return loaders if len(loaders) > 1 else loaders[0]

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized. Call setup() first.")
        if self.test_sequential:
            return self._make_standard_eval_loader(self.test_dataset)
        return self._make_eval_loader(self.test_dataset)


class LanceSamplerEpochCallback(pl.Callback):
    """Advance train/replay sampling exactly once per epoch."""

    def __init__(self, datamodule) -> None:
        self.datamodule = datamodule

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.datamodule.set_epoch(int(trainer.current_epoch))


def lance_callbacks(datamodule):
    return [LanceSamplerEpochCallback(datamodule)]

class MultiSpeciesDataModule(pl.LightningDataModule):
    """Lightning datamodule for map-style datasets with DDP samplers."""

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size: int,
        collate_fn: Callable,
        seed: int = 0,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = True,
        expected_world_size: Optional[int] = None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.eval_batch_size = int(eval_batch_size) if eval_batch_size is not None else int(batch_size)
        self.collate_fn = collate_fn
        self.seed = seed
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.drop_last = bool(drop_last)
        self.expected_world_size = expected_world_size
        self._train_sampler: Optional[DistributedSampler] = None
        self._epoch = 0

    def setup(self, stage=None):
        pass

    def set_epoch(self, epoch: int) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(int(epoch))

    def _make_train_sampler(self) -> DistributedSampler:
        rank, world_size = infer_rank_world_size(
            expected_world_size=self.expected_world_size
        )
        return DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(self.seed),
            drop_last=bool(self.drop_last),
        )

    def num_train_batches(self) -> int:
        """Return train dataloader length without constructing a DataLoader."""
        return _num_batches(
            len(self._make_train_sampler()),
            self.batch_size,
            self.drop_last,
        )

    def train_dataloader(self):
        train_sampler = self._make_train_sampler()
        train_sampler.set_epoch(0)
        self._train_sampler = train_sampler
        return make_lance_safe_loader(
            dataset=self.train_dataset,
            batch_size=int(self.batch_size),
            num_workers=int(self.num_workers),
            drop_last=bool(self.drop_last),
            pin_memory=bool(self.pin_memory),
            sampler=train_sampler,
            persistent_workers=bool(self.persistent_workers),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        rank, world_size = infer_rank_world_size(expected_world_size=self.expected_world_size)
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        return make_lance_safe_loader(
            dataset=self.val_dataset,
            batch_size=int(self.eval_batch_size),
            num_workers=int(self.num_workers),
            drop_last=False,
            pin_memory=bool(self.pin_memory),
            sampler=sampler,
            persistent_workers=bool(self.persistent_workers),
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        rank, world_size = infer_rank_world_size(
            expected_world_size=self.expected_world_size
        )
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )

        return make_lance_safe_loader(
            dataset=self.test_dataset,
            batch_size=int(self.eval_batch_size),
            num_workers=int(self.num_workers),
            drop_last=False,
            pin_memory=bool(self.pin_memory),
            sampler=sampler,
            persistent_workers=bool(self.persistent_workers),
            collate_fn=self.collate_fn,
        )
