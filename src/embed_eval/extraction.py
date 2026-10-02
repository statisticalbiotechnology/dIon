"""Shared embedding extraction for standalone and online evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from src.precursor_conditioning import condition_precursor_inputs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass(frozen=True)
class CachedEmbeddings:
    """CPU embedding cache with labels aligned to the embedding rows."""

    values: torch.Tensor
    peptide_ids: list[str]
    partition_ids: list[str]
    spectrum_ids: list[str]
    precursor_mz: torch.Tensor
    precursor_charges: torch.Tensor


def extract_precursor_metadata_embeddings(dataset) -> CachedEmbeddings:
    """Create a label-free precursor-only baseline on an evaluation subset.

    The two features are log precursor mass (observed m/z times charge) and
    charge. They are standardized over the selected evaluation spectra, then
    evaluated by the same cosine-distance metrics as model embeddings.
    """
    table = getattr(dataset, "table", None)
    if table is None:
        raise TypeError(
            "The precursor metadata baseline requires a dataset exposing a "
            "selected Arrow table."
        )
    required_columns = {"precursor_mz", "precursor_charge"}
    missing_columns = sorted(required_columns - set(table.column_names))
    if missing_columns:
        raise ValueError(
            "The precursor metadata baseline is missing columns: "
            f"{missing_columns}."
        )

    precursor_mz = torch.tensor(
        table["precursor_mz"].to_pylist(), dtype=torch.float32
    )
    charge = torch.tensor(
        table["precursor_charge"].to_pylist(), dtype=torch.float32
    )
    precursor_mass = precursor_mz * charge
    if not torch.isfinite(precursor_mass).all() or (precursor_mass <= 0).any():
        raise ValueError("Precursor metadata baseline requires finite positive masses.")
    if not torch.isfinite(charge).all():
        raise ValueError("Precursor metadata baseline requires finite charges.")

    features = torch.stack([torch.log(precursor_mass), charge], dim=1)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    values = (features - mean) / std

    peptide_column = getattr(dataset, "peptide_id_column", "peptide_ion_id")
    partition_column = getattr(dataset, "partition_column", "species")
    peptide_ids = [str(value) for value in table[peptide_column].to_pylist()]
    partition_ids = [str(value) for value in table[partition_column].to_pylist()]
    spectrum_column = "spectrum_id"
    spectrum_ids = (
        [str(value) for value in table[spectrum_column].to_pylist()]
        if spectrum_column in table.column_names
        else peptide_ids
    )
    return CachedEmbeddings(
        values=values,
        peptide_ids=peptide_ids,
        partition_ids=partition_ids,
        spectrum_ids=spectrum_ids,
        precursor_mz=precursor_mz,
        precursor_charges=charge,
    )


@contextmanager
def preserve_eval_mode(module: torch.nn.Module):
    """Evaluate a live training module temporarily and restore every mode flag."""
    modules_with_mode = []
    seen = set()
    for child in module.modules():
        if id(child) not in seen:
            seen.add(id(child))
            modules_with_mode.append((child, child.training))
    module.eval()
    try:
        yield
    finally:
        for child, was_training in modules_with_mode:
            child.train(was_training)


def extract_embeddings(
    embedder: torch.nn.Module,
    dataset,
    *,
    collate_fn,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    pin_memory: bool = False,
    show_progress: bool = False,
    progress_desc: str = "Embedding spectra",
    input_intervention=None,
    precursor_conditioning: str = "conditioned",
) -> CachedEmbeddings:
    """Embed a labelled dataset with the model input semantics used by dIon."""
    if batch_size < 1:
        raise ValueError("Embedding evaluation batch_size must be positive.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    values = []
    peptide_ids: list[str] = []
    partition_ids: list[str] = []
    spectrum_ids: list[str] = []
    precursor_mz_values = []
    precursor_charge_values = []
    source_row_indices = []
    embedder = embedder.to(device)

    with preserve_eval_mode(embedder), torch.inference_mode():
        for batch in tqdm(
            loader,
            total=len(loader),
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            disable=not show_progress,
        ):
            mz = batch["mz_array"].to(device)
            intensity = batch["intensity_array"].to(device)
            spectra = torch.stack([mz, intensity], dim=-1)
            batch_size_actual, sequence_length, _ = spectra.shape
            lengths = batch["peak_lengths"].reshape(batch_size_actual).to(device)
            padding_mask = (
                torch.arange(sequence_length, device=device)
                .unsqueeze(0)
                .expand(batch_size_actual, sequence_length)
                .ge(lengths.unsqueeze(1))
            )
            # DINO/DINOv2 training receives the standardized precursor mass
            # generated by pad_peaks: observed precursor m/z times charge.
            mass = (
                batch["precursor_mass"].to(device)
                if embedder.use_mass
                else None
            )
            charge = (
                batch["precursor_charge"].to(device)
                if embedder.use_charge
                else None
            )
            if input_intervention is not None:
                row_indices = batch.get("_embedding_row_index")
                if row_indices is None:
                    raise ValueError(
                        "Input interventions require collated _embedding_row_index metadata."
                    )
                spectra, padding_mask, mass, charge = input_intervention.apply(
                    spectra,
                    padding_mask,
                    mass,
                    charge,
                    row_indices=row_indices.reshape(-1),
                )
            mass, charge = condition_precursor_inputs(
                mass,
                charge,
                precursor_conditioning,
            )
            values.append(embedder(spectra, padding_mask, mass, charge).cpu())
            precursor_mz_values.append(batch["precursor_mz"].reshape(-1).cpu())
            precursor_charge_values.append(
                batch["precursor_charge"].reshape(-1).cpu()
            )
            if hasattr(dataset, "_selection_rank_by_source_index"):
                source_row_indices.extend(
                    int(value)
                    for value in batch["_embedding_row_index"].reshape(-1).tolist()
                )
            peptide_ids.extend(str(value) for value in batch["peptide_id"])
            partition_ids.extend(str(value) for value in batch["partition_id"])
            spectrum_ids.extend(
                str(value)
                for value in batch.get("spectrum_id", batch["peptide_id"])
            )

    if not values:
        raise ValueError("Embedding evaluation dataset produced no batches.")
    values_tensor = torch.cat(values, dim=0)
    precursor_mz_tensor = torch.cat(precursor_mz_values, dim=0)
    precursor_charge_tensor = torch.cat(precursor_charge_values, dim=0)
    if source_row_indices:
        rank_by_source_index = dataset._selection_rank_by_source_index
        order = torch.tensor(
            [rank_by_source_index[index] for index in source_row_indices],
            dtype=torch.long,
        ).argsort()
        values_tensor = values_tensor[order]
        precursor_mz_tensor = precursor_mz_tensor[order]
        precursor_charge_tensor = precursor_charge_tensor[order]
        order_list = order.tolist()
        peptide_ids = [peptide_ids[index] for index in order_list]
        partition_ids = [partition_ids[index] for index in order_list]
        spectrum_ids = [spectrum_ids[index] for index in order_list]
    return CachedEmbeddings(
        values=values_tensor,
        peptide_ids=peptide_ids,
        partition_ids=partition_ids,
        spectrum_ids=spectrum_ids,
        precursor_mz=precursor_mz_tensor,
        precursor_charges=precursor_charge_tensor,
    )
