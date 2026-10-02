from dataclasses import dataclass
import hashlib
from pathlib import Path

from src.data.unified_tokenizer import PeptideTokenizer
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import ConcatDataset
import os
import torch
from functools import partial

from pytorch_lightning.callbacks import EarlyStopping
from src.data.lance_data_module import LanceDataModule, MultiSpeciesDataModule
from src.data.lance_datasets import SafeLanceDataset
from src.data.sqa_inmem_data_module import SpectrumQualityDataModule
from src.callbacks.prediction_csv_output_callback import PredictionCsvOutputCallback
from src.pl_callbacks import (
    FLOPProfilerCallback,
    CosineAnnealLRCallback,
    WarmupConstantLRCallback,
    StopAfterEpochCallback,
    LinearWarmupLRCallback,
    ExponentialDecayLRCallback,
    MztabOutputCallback,
    AlwaysSaveLastCheckpoint,
)

from src.collate_functions import pad_metric_learning, pad_peaks, pad_peptides



@dataclass(frozen=True)
class PeptideDatasetSpec:
    label_name: str
    precursor_mz_name: str | bool
    precursor_mass_name: str | bool
    explicit_paths: bool = False
    replace_isoleucine_with_leucine: bool = False


PEPTIDE_DATASET_SPECS: dict[str, PeptideDatasetSpec] = {
    "massivekb": PeptideDatasetSpec(
        label_name="sequence",
        precursor_mz_name=False,
        precursor_mass_name="precursor_mz",
    ),
    "bacteria": PeptideDatasetSpec(
        label_name="seq",
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
    ),
    "kitchensink_v4": PeptideDatasetSpec(
        label_name="seq",
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
    ),
    "configurable_lance": PeptideDatasetSpec(
        label_name="seq",
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
        explicit_paths=True,
    ),
    "ninespecies_updated": PeptideDatasetSpec(
        label_name="modified_sequence",
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
        explicit_paths=True,
    ),
    "ninespecies_v2": PeptideDatasetSpec(
        label_name="modified_sequence",
        # In this Lance export, the column named "precursor_mass" contains
        # observed precursor m/z; derive standardized precursor_mass from it.
        precursor_mz_name="precursor_mass",
        precursor_mass_name=False,
    ),
}


def _expected_world_size(global_args) -> int:
    return max(1, int(global_args.num_devices) * int(global_args.num_nodes))


def _stable_species_seed(species: str, base_seed: int) -> int:
    digest = hashlib.md5(species.encode()).hexdigest()[:8]
    return base_seed + int(digest, 16)


def _split_species_dataset(dataset, val_percentage: float, seed: int):
    if not 0.0 < val_percentage < 1.0:
        raise ValueError("val_percentage must be in (0, 1).")
    n = len(dataset)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_count = max(1, int(n * val_percentage))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _species_lance_paths(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Species lance root not found: {root_dir}")
    paths = sorted(root.glob("*.lance"))
    if not paths:
        raise FileNotFoundError(f"No .lance datasets found in {root_dir}")
    return {p.stem: str(p) for p in paths}


def get_lance_data_module(
    global_args,
    pt_config,
    max_peaks=300,
    seed=0,
    include_test=False,
):
    task_config = pt_config[global_args.pretraining_task]
    dataset_name = task_config.get("dataset_name", "bacteria")
    if dataset_name not in PEPTIDE_DATASET_SPECS:
        raise ValueError(f"Unknown pretraining dataset: {dataset_name}")
    spec = PEPTIDE_DATASET_SPECS[dataset_name]

    collate_fn = partial(
        pad_peaks,
        max_peaks=max_peaks,
        precursor_mz_name=spec.precursor_mz_name,
        precursor_mass_name=spec.precursor_mass_name,
        # Peak filter settings
        filter_method=global_args.peak_filter_method,
        intensity_scaling=global_args.intensity_scaling,
        min_mz=global_args.min_mz,
        max_mz=global_args.max_mz,
        min_intensity=global_args.min_intensity,
        remove_precursor_tol=global_args.remove_precursor_tol,
        min_peaks=global_args.min_peaks,
    )

    batch_size = (
        task_config["batch_size"]
        if global_args.batch_size < 0
        else global_args.batch_size
    )
    num_workers = (
        task_config.get("num_workers", 0)
        if global_args.num_workers < 0
        else global_args.num_workers
    )

    source_columns = ["precursor_charge"]
    source_columns.extend(
        name
        for name in (spec.precursor_mz_name, spec.precursor_mass_name)
        if name
    )
    source_columns = list(dict.fromkeys(source_columns))

    return LanceDataModule(
        global_args.data_root_dir,
        batch_size,
        collate_fn,
        seed=seed,
        include_test=include_test,
        columns=source_columns,
        num_workers=num_workers,
        pin_memory=global_args.pin_mem,
        expected_world_size=_expected_world_size(global_args),
    )


def _build_peptide_tokenizer(
    ds_config, task_config, spec: PeptideDatasetSpec
) -> PeptideTokenizer:
    tokenizer_config = ds_config.get("tokenizer", {})
    if tokenizer_config.get("source") == "numeric_mass_delta_manifest":
        return PeptideTokenizer.from_numeric_mass_delta_manifest(
            tokenizer_config["manifest"],
            reverse=task_config.get("reverse", False),
            replace_isoleucine_with_leucine=spec.replace_isoleucine_with_leucine,
            **tokenizer_config.get("special_tokens", {}),
        )
    return PeptideTokenizer.unified(
        reverse=task_config.get("reverse", False),
        replace_isoleucine_with_leucine=spec.replace_isoleucine_with_leucine,
    )


def _optional_path(ds_config, global_args, config_key: str, arg_key: str):
    explicit_arg = getattr(global_args, arg_key, None)
    return explicit_arg or ds_config.get(config_key)


def get_lance_peptide_data_module(
    ds_config,
    global_args,
    dataset_name: str,
    seed=0,
):
    if dataset_name not in PEPTIDE_DATASET_SPECS:
        raise ValueError(f"Unknown peptide dataset: {dataset_name}")

    spec = PEPTIDE_DATASET_SPECS[dataset_name]
    task_config = ds_config[global_args.downstream_task]
    tokenizer = _build_peptide_tokenizer(ds_config, task_config, spec)
    label_name = ds_config.get("label_name", spec.label_name)
    precursor_mz_name = ds_config.get("precursor_mz_name", spec.precursor_mz_name)
    precursor_mass_name = ds_config.get("precursor_mass_name", spec.precursor_mass_name)
    collate_fn = partial(
        pad_peptides,
        max_peaks=global_args.max_peaks,
        max_length=ds_config["pep_length"][1],
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        label_name=label_name,
        precursor_mz_name=precursor_mz_name,
        precursor_mass_name=precursor_mass_name,
        filter_method=global_args.peak_filter_method,
        intensity_scaling=global_args.intensity_scaling,
        min_mz=global_args.min_mz,
        max_mz=global_args.max_mz,
        min_intensity=global_args.min_intensity,
        remove_precursor_tol=global_args.remove_precursor_tol,
    )

    batch_size = task_config["batch_size"] if global_args.batch_size < 0 else global_args.batch_size
    eval_batch_size = task_config.get("eval_batch_size", batch_size)
    num_workers = ds_config.get("num_workers", 0) if global_args.num_workers < 0 else global_args.num_workers
    columns = [label_name, "precursor_charge"]
    columns.extend(name for name in (precursor_mz_name, precursor_mass_name) if name)
    columns.extend(ds_config.get("metadata_columns", []))
    columns = list(dict.fromkeys(columns))

    train_path = _optional_path(ds_config, global_args, "train_lance", "downstream_train_path")
    val_path = _optional_path(ds_config, global_args, "val_lance", "downstream_val_path")
    test_path = _optional_path(ds_config, global_args, "test_lance", "downstream_test_path")
    explicit_paths = spec.explicit_paths or any(path is not None for path in (train_path, val_path, test_path))
    data_module = LanceDataModule(
        data_dir=global_args.downstream_root_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        collate_fn=collate_fn,
        seed=seed,
        num_workers=num_workers,
        pin_memory=global_args.pin_mem,
        expected_world_size=_expected_world_size(global_args),
        columns=columns,
        train_path=train_path if explicit_paths else None,
        val_path=val_path if explicit_paths else None,
        test_path=test_path if explicit_paths else None,
        extra_val_paths=ds_config.get("extra_val_sets"),
        replay_train_path=ds_config.get("replay_train_lance"),
        replay_ratio=float(ds_config.get("replay_ratio", 0.0)),
        include_index=bool(task_config.get("prediction_table", False)),
        exact_eval_sharding=bool(task_config.get("exact_eval_sharding", False)),
    )
    return data_module, tokenizer


def get_metric_learning_data_module(ds_config, global_args, dataset_name: str, seed=0):
    """Build the normal Lance datamodule with SupCon-positive batch construction."""
    if dataset_name not in PEPTIDE_DATASET_SPECS:
        raise ValueError(f"Unknown metric-learning dataset: {dataset_name}")
    spec = PEPTIDE_DATASET_SPECS[dataset_name]
    task_config = ds_config[global_args.downstream_task]
    label_name = ds_config.get("label_name", spec.label_name)
    precursor_mz_name = ds_config.get("precursor_mz_name", spec.precursor_mz_name)
    precursor_mass_name = ds_config.get("precursor_mass_name", spec.precursor_mass_name)
    collate_fn = partial(
        pad_metric_learning,
        label_name=label_name,
        max_peaks=ds_config["top_peaks"],
        precursor_mz_name=precursor_mz_name,
        precursor_mass_name=precursor_mass_name,
        filter_method=global_args.peak_filter_method,
        intensity_scaling=global_args.intensity_scaling,
        min_mz=global_args.min_mz,
        max_mz=global_args.max_mz,
        min_intensity=global_args.min_intensity,
        remove_precursor_tol=global_args.remove_precursor_tol,
    )
    batch_size = task_config["batch_size"] if global_args.batch_size < 0 else global_args.batch_size
    num_workers = ds_config.get("num_workers", 0) if global_args.num_workers < 0 else global_args.num_workers
    columns = [label_name, "precursor_charge"]
    columns.extend(name for name in (precursor_mz_name, precursor_mass_name) if name)
    columns.extend(ds_config.get("metadata_columns", []))
    columns = list(dict.fromkeys(columns))
    train_path = _optional_path(ds_config, global_args, "train_lance", "downstream_train_path")
    val_path = _optional_path(ds_config, global_args, "val_lance", "downstream_val_path")
    test_path = _optional_path(ds_config, global_args, "test_lance", "downstream_test_path")
    if train_path is None or val_path is None:
        raise ValueError("Metric learning requires explicit train_lance and val_lance paths.")
    positive_batching = {
        "label_column": label_name,
        "peptides_per_batch": int(task_config["peptides_per_batch"]),
        "spectra_per_peptide": int(task_config["spectra_per_peptide"]),
        "precursor_mz_column": precursor_mz_name,
        "precursor_charge_column": "precursor_charge",
    }
    hard_negative_config = task_config.get("hard_negative_sampling", {})
    if not isinstance(hard_negative_config, dict):
        raise TypeError("metric_learning.hard_negative_sampling must be a mapping.")
    for key in ("strict_ppm", "relaxed_ppm_windows", "diagnostics_reservoir_size"):
        if key in hard_negative_config:
            positive_batching[key] = hard_negative_config[key]
    if task_config.get("train_batches_per_epoch") is not None:
        positive_batching["train_batches_per_epoch"] = int(task_config["train_batches_per_epoch"])
    if task_config.get("validation_batches_per_epoch") is not None:
        positive_batching["validation_batches_per_epoch"] = int(task_config["validation_batches_per_epoch"])
    data_module = LanceDataModule(
        data_dir=global_args.downstream_root_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        collate_fn=collate_fn,
        seed=seed,
        num_workers=num_workers,
        pin_memory=global_args.pin_mem,
        expected_world_size=_expected_world_size(global_args),
        columns=columns,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        positive_batching=positive_batching,
        test_sequential=bool(task_config.get("test_full_dataset_pass", False)),
    )
    return data_module


def get_ninespecies_v2_lance_data_module(
    ds_config,
    global_args,
    seed=0,
):
    spec = PEPTIDE_DATASET_SPECS["ninespecies_v2"]
    task_config = ds_config[global_args.downstream_task]
    tokenizer = _build_peptide_tokenizer(ds_config, task_config, spec)
    collate_fn = partial(
        pad_peptides,
        max_peaks=ds_config["top_peaks"],
        max_length=ds_config["pep_length"][1],
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        label_name=spec.label_name,
        precursor_mz_name=spec.precursor_mz_name,
        precursor_mass_name=spec.precursor_mass_name,
        filter_method=global_args.peak_filter_method,
        intensity_scaling=global_args.intensity_scaling,
        min_mz=global_args.min_mz,
        max_mz=global_args.max_mz,
        min_intensity=global_args.min_intensity,
        remove_precursor_tol=global_args.remove_precursor_tol,
    )

    batch_size = (
        ds_config[global_args.downstream_task]["batch_size"]
        if global_args.batch_size < 0
        else global_args.batch_size
    )
    eval_batch_size = ds_config[global_args.downstream_task].get(
        "eval_batch_size", batch_size
    )
    num_workers = (
        ds_config.get("num_workers", 0)
        if global_args.num_workers < 0
        else global_args.num_workers
    )

    test_species = ds_config.get("test_species")
    if not test_species:
        raise ValueError("test_species is required for ninespecies_v2 dataset.")
    val_percentage = float(ds_config.get("val_percentage", 0.1))

    species_paths = _species_lance_paths(global_args.downstream_root_dir)
    if test_species not in species_paths:
        raise ValueError(
            f"test_species '{test_species}' not found in {global_args.downstream_root_dir}"
        )

    train_subsets = []
    val_subsets = []
    for species, path in species_paths.items():
        dataset = SafeLanceDataset(path)
        if species == test_species:
            test_dataset = dataset
            continue
        split_seed = _stable_species_seed(species, seed)
        train_subset, val_subset = _split_species_dataset(
            dataset, val_percentage, split_seed
        )
        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

    if not train_subsets:
        raise ValueError("No training species found after excluding val_species.")

    train_dataset = ConcatDataset(train_subsets)
    val_dataset = ConcatDataset(val_subsets)

    data_module = MultiSpeciesDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        collate_fn=collate_fn,
        seed=seed,
        num_workers=num_workers,
        pin_memory=global_args.pin_mem,
        expected_world_size=_expected_world_size(global_args),
    )

    return data_module, tokenizer


def get_sqa_inmem_data_module(
    ds_config,
    max_peaks,
    global_args,
    seed=0,
):
    batch_size = (
        ds_config[global_args.downstream_task]["batch_size"]
        if global_args.batch_size < 0
        else global_args.batch_size
    )

    collate_fn = partial(
        pad_peaks,
        max_peaks=max_peaks,
        precursor_mz_name="precursor_mz",  # assuming the bacterium dataset
        precursor_mass_name=False,
        # Peak filter settings
        filter_method=global_args.peak_filter_method,
        intensity_scaling=global_args.intensity_scaling,
        min_mz=global_args.min_mz,
        max_mz=global_args.max_mz,
        min_intensity=global_args.min_intensity,
        remove_precursor_tol=global_args.remove_precursor_tol,
    )

    return SpectrumQualityDataModule(
        global_args.downstream_root_dir,
        batch_size,
        collate_fn,
        seed=seed,
        num_workers=global_args.num_workers,
    )


def get_auxiliary_lance_data_module(ds_config, global_args, seed=0):
    """Build the standard Lance loader for spectrum-level auxiliary tasks."""
    task_config = ds_config[global_args.downstream_task]
    target_column = str(task_config["target_column"])
    batch_size = (
        task_config["batch_size"] if global_args.batch_size < 0 else global_args.batch_size
    )
    eval_batch_size = int(task_config.get("eval_batch_size", batch_size))
    evaluation_mask_column = task_config.get("evaluation_mask_column")
    columns = [target_column, "precursor_mz", "precursor_charge"]
    if evaluation_mask_column:
        columns.append(str(evaluation_mask_column))
    columns.extend(ds_config.get("metadata_columns", []))
    columns = list(dict.fromkeys(columns))
    collate_fn = partial(
        pad_peaks,
        max_peaks=ds_config["top_peaks"],
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
    train_path = _optional_path(ds_config, global_args, "train_lance", "downstream_train_path")
    val_path = _optional_path(ds_config, global_args, "val_lance", "downstream_val_path")
    test_path = _optional_path(ds_config, global_args, "test_lance", "downstream_test_path")
    explicit_paths = any(path is not None for path in (train_path, val_path, test_path))
    return LanceDataModule(
        data_dir=global_args.downstream_root_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        collate_fn=collate_fn,
        seed=seed,
        num_workers=(ds_config.get("num_workers", 0) if global_args.num_workers < 0 else global_args.num_workers),
        pin_memory=global_args.pin_mem,
        expected_world_size=_expected_world_size(global_args),
        columns=columns,
        train_path=train_path if explicit_paths else None,
        val_path=val_path if explicit_paths else None,
        test_path=test_path if explicit_paths else None,
        include_index=True,
    )


def configure_callbacks(
    global_args, task_config, val_metric_name: str = "val_loss", metric_mode="min"
):
    callbacks = []
    filename = f"{{epoch}}-{{{val_metric_name}:.2f}}"
    # filename = f"{{epoch}}-{{step}}"
    # Checkpoint callback
    if not global_args.barebones:
        callbacks += [
            AlwaysSaveLastCheckpoint(
                dirpath=global_args.output_dir,
                filename=filename,
                monitor=val_metric_name,  # requires that we log something called val_metric_name
                mode=metric_mode,
                save_top_k=global_args.save_top_k,
                save_last=global_args.save_last,
                every_n_epochs=global_args.every_n_epochs,
                # every_n_train_steps=args.every_n_train_steps,
            )
        ]

    batch_size = (
        task_config["batch_size"]
        if global_args.batch_size < 0
        else global_args.batch_size
    )

    base_lr = task_config.get("head_lr", task_config.get("learning_rate", task_config.get("blr")))
    if base_lr is None:
        raise KeyError("Task configuration must define head_lr, learning_rate, or blr.")
    if global_args.scale_lr_by_batchsize:
        ref_batch_size = task_config.get("ref_batch_size", 0)

        eff_batch_size = (
            batch_size
            * global_args.accum_iter
            * global_args.num_devices
            * global_args.num_nodes
        )

        assert (
            ref_batch_size > 0
        ), "'ref_batch_size' must be provided when 'scale_lr_by_batchsize' is True"
        blr = base_lr * eff_batch_size / ref_batch_size
    else:
        blr = base_lr

    if task_config.get("anneal_lr", False):
        schedule_mode = task_config.get("schedule_mode", "cosine")
        if schedule_mode == "cosine":
            callbacks += [
                CosineAnnealLRCallback(
                    lr_start=task_config["lr_start"],
                    blr=blr,
                    lr_end=task_config["lr_end"],
                    warmup_duration=task_config["warmup_duration"],
                    decay_delay=task_config["decay_delay"],
                    decay_duration=task_config["decay_duration"],
                    anneal_per_step=task_config["anneal_per_step"],
                )
            ]
        elif schedule_mode == "indefinite":
            callbacks += [
                WarmupConstantLRCallback(
                    lr_start=task_config["lr_start"],
                    blr=blr,
                    warmup_duration=task_config["warmup_duration"],
                    anneal_per_step=task_config["anneal_per_step"],
                )
            ]
        else:
            raise ValueError(
                "schedule_mode must be either 'cosine' or 'indefinite', "
                f"got {schedule_mode!r}."
            )

    stop_after_epoch = task_config.get("stop_after_epoch")
    if stop_after_epoch is not None:
        callbacks += [StopAfterEpochCallback(stop_after_epoch)]

    if task_config.get("log_predictions", False):
        callbacks += [PredictionCsvOutputCallback(global_args.log_dir, global_args)] if task_config.get("prediction_table", False) else [MztabOutputCallback(global_args.log_dir, global_args)]

    # measure FLOPs on the first train batch
    if global_args.profile_flops:
        callbacks += [FLOPProfilerCallback()]

    if global_args.early_stop > 0:
        callbacks += [
            EarlyStopping(
                monitor=val_metric_name,
                mode=metric_mode,
                patience=global_args.early_stop,
            )
        ]

    return callbacks


def get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_num_parameters(model):
    sum = 0
    for param in list(model.parameters()):
        sum += param.numel()
    return sum


class Scale:
    def __init__(self, tokenizer: PeptideTokenizer):
        self.tokenizer = tokenizer
        self.amod_dict = dict(tokenizer.index)
        int2mass = np.zeros((len(self.amod_dict)))
        for token, integer in self.amod_dict.items():
            int2mass[integer] = tokenizer.residues.get(token, 0.0)

        self.tok2mass = {key: int2mass[self.amod_dict[key]] for key in self.amod_dict}
        self.mp = torch.tensor(int2mass, dtype=torch.float32)

    def intseq2mass(self, intseq):
        return torch.gather(self.mp, 0, intseq).sum(1)

    def modseq2mass(self, modified_sequence):
        tokens = self.tokenizer.preprocess_sequence(modified_sequence)
        return np.sum(self.tok2mass.get(tok, 0.0) for tok in tokens)


if __name__ == "__main__":
    ### Test code
    data_dir = "/Users/anon/Documents/Datasets/instanovo_data_subset"
    lance_dir = "/Users/anon/Documents/Datasets/instanovo_data_subset/indexed.lance"
    mdsaved_dir = "//Users/anon/Documents/Datasets/instanovo_data_subset/mdsaved"

    datasets = get_spectrum_dataset_splits(data_dir)
    print(next(iter(datasets[0])))
    print(len(datasets[0]))
    print(len(datasets[1]))
    print(len(datasets[2]))
