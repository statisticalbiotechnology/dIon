"""Deterministic test-time interventions for embedding-input reliance tests."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import torch

from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation


def _seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (2**31)


def _dataset_metadata(dataset) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    table = getattr(dataset, "table", None)
    if table is None:
        raise TypeError("Input interventions require a dataset with a selected Arrow table.")
    partition_column = getattr(dataset, "partition_column", "species")
    precursor_mz = torch.tensor(table["precursor_mz"].to_pylist(), dtype=torch.float32)
    charges = torch.tensor(table["precursor_charge"].to_pylist(), dtype=torch.long)
    partitions = [str(value) for value in table[partition_column].to_pylist()]
    return precursor_mz * charges, charges, partitions


def _group_permutation(groups: dict[object, list[int]], seed: int, name: str) -> torch.Tensor:
    size = sum(len(indices) for indices in groups.values())
    permutation = torch.arange(size, dtype=torch.long)
    for group, indices in groups.items():
        if len(indices) < 2:
            continue
        source = torch.tensor(indices, dtype=torch.long)
        generator = torch.Generator().manual_seed(_seed(seed, name, group))
        permutation[source] = source[torch.randperm(len(source), generator=generator)]
    return permutation


class InputIntervention:
    """Prepared deterministic intervention for one selected evaluation dataset."""

    VALID_NAMES = {
        "none",
        "permute_mass_within_species_charge",
        "permute_charge_within_species",
        "permute_mass_and_charge",
        "permute_peaks_within_batch",
        "shuffle_intensities_within_spectrum",
        "inject_distractor_peaks_50pct",
    }

    def __init__(self, name: str, dataset, seed: int) -> None:
        if name not in self.VALID_NAMES:
            raise ValueError(f"Unsupported input intervention: {name!r}.")
        self.name = name
        self.seed = int(seed)
        masses, charges, partitions = _dataset_metadata(dataset)
        self.masses = masses
        self.precursor_mz = masses / charges.clamp_min(1)
        self.charges = charges
        self.distractor_mixer = (
            BatchedStudentDistractorMixAugmentation(
                mix_apply_to="all",
                condition_separation_ppm=10.0,
                neutral_mass_separation_ppm=10.0,
                merge_ppm=5.0,
                intensity_normalization="none",
            )
            if name == "inject_distractor_peaks_50pct"
            else None
        )
        self.mass_permutation = torch.arange(len(masses), dtype=torch.long)
        self.charge_permutation = torch.arange(len(masses), dtype=torch.long)

        by_species_charge: dict[tuple[str, int], list[int]] = defaultdict(list)
        by_species: dict[str, list[int]] = defaultdict(list)
        for index, (partition, charge) in enumerate(
            zip(partitions, charges.tolist(), strict=True)
        ):
            by_species_charge[(partition, int(charge))].append(index)
            by_species[partition].append(index)
        if name in {"permute_mass_within_species_charge", "permute_mass_and_charge"}:
            self.mass_permutation = _group_permutation(
                by_species_charge, self.seed, "mass"
            )
        if name in {"permute_charge_within_species", "permute_mass_and_charge"}:
            self.charge_permutation = _group_permutation(
                by_species, self.seed, "charge"
            )

    def apply(self, spectra, padding_mask, mass, charge, *, row_indices: torch.Tensor):
        """Apply one intervention while preserving the selected corpus distribution."""
        batch_size = spectra.shape[0]
        row_indices = row_indices.detach().to(dtype=torch.long, device="cpu")
        if row_indices.numel() != batch_size:
            raise ValueError("Intervention row_indices must align with the batch.")
        if self.name in {"permute_mass_within_species_charge", "permute_mass_and_charge"} and mass is not None:
            mass = self.masses[self.mass_permutation[row_indices]].to(
                mass.device, dtype=mass.dtype
            )
        if self.name in {"permute_charge_within_species", "permute_mass_and_charge"} and charge is not None:
            charge = self.charges[self.charge_permutation[row_indices]].to(
                charge.device, dtype=charge.dtype
            )
        if self.name == "inject_distractor_peaks_50pct":
            lengths = (~padding_mask).sum(dim=1, keepdim=True)
            cuda_devices = [spectra.device] if spectra.device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(
                    _seed(self.seed, "distractor", *row_indices.tolist())
                )
                mixed = self.distractor_mixer(
                    [(spectra, padding_mask)],
                    spectra,
                    lengths,
                    self.precursor_mz[row_indices].to(spectra.device),
                    self.charges[row_indices].to(spectra.device),
                    strength=0.5,
                    num_global_crops=1,
                )
            spectra, padding_mask = mixed[0]
        if self.name == "permute_peaks_within_batch":
            generator = torch.Generator(device=spectra.device).manual_seed(
                _seed(self.seed, "peaks", *row_indices.tolist())
            )
            permutation = torch.randperm(
                batch_size, generator=generator, device=spectra.device
            )
            spectra = spectra[permutation]
            padding_mask = padding_mask[permutation]
        elif self.name == "shuffle_intensities_within_spectrum":
            spectra = spectra.clone()
            lengths = (~padding_mask).sum(dim=1).tolist()
            for row, length in enumerate(lengths):
                if length < 2:
                    continue
                generator = torch.Generator(device=spectra.device).manual_seed(
                    _seed(self.seed, "intensity", int(row_indices[row].item()))
                )
                permutation = torch.randperm(
                    length, generator=generator, device=spectra.device
                )
                spectra[row, :length, 1] = spectra[row, :length, 1][permutation]
        return spectra, padding_mask, mass, charge
