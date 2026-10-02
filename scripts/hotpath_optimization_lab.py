#!/usr/bin/env python3
"""Isolated hot-path optimization lab for dIon.

This script deliberately does not patch or import into the training codepath.
It imports the current implementations, defines optimized candidates beside
them, and checks candidate outputs against legacy outputs before benchmarking.

The main targets are:

1. DINO random/intensity-weighted selection augmentation
   - Legacy code calls .item() inside the per-sample GPU loop.
   - Candidate preserves the legacy RNG stream and output exactly, while doing
     one CPU transfer for lengths/sizes and vectorizing the final copy.

2. DINO window augmentation
   - Candidate avoids repeat() allocation for the per-row index grid.
   - Candidate preserves the legacy RNG stream and output exactly.

3. pad_peaks collate
   - Candidate avoids mutating sample dictionaries and uses topk for peak
     subsampling.
   - Candidate is asserted against the legacy collate output on synthetic data.

Run from the repo root, preferably inside the dIon Apptainer image:

    python scripts/hotpath_optimization_lab.py --device cuda --batch-size 128
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collate_functions import (  # noqa: E402
    _as_tensor,
    basepeak_scale,
    casanovo_filter_peaks,
    default_filter_peaks,
    minmax_scale,
    pad_peaks as legacy_pad_peaks,
)
from src.data_augmentation import (  # noqa: E402
    IntensityWeightedSelectionAugmentation,
    RandomSelectionAugmentation,
    RandomWindowAugmentation,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)


def clone_batch(batch: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: clone_value(value) for key, value in item.items()} for item in batch]


def assert_nested_equal(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError(f"{path}: tensor/non-tensor mismatch")
        if left.dtype != right.dtype:
            raise AssertionError(f"{path}: dtype mismatch {left.dtype} != {right.dtype}")
        if left.shape != right.shape:
            raise AssertionError(f"{path}: shape mismatch {left.shape} != {right.shape}")
        if torch.is_floating_point(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        elif not torch.equal(left, right):
            raise AssertionError(f"{path}: tensor values differ")
        return

    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"{path}: keys differ {set(left) ^ set(right)}")
        for key in sorted(left):
            assert_nested_equal(left[key], right[key], path=f"{path}.{key}")
        return

    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: length mismatch {len(left)} != {len(right)}")
        for idx, (l_val, r_val) in enumerate(zip(left, right)):
            assert_nested_equal(l_val, r_val, path=f"{path}[{idx}]")
        return

    if left != right:
        raise AssertionError(f"{path}: value mismatch {left!r} != {right!r}")


def assert_crops_equal(left, right, *, label: str) -> None:
    if len(left) != len(right):
        raise AssertionError(f"{label}: crop count mismatch {len(left)} != {len(right)}")
    for crop_idx, ((left_crop, left_mask), (right_crop, right_mask)) in enumerate(
        zip(left, right)
    ):
        assert_nested_equal(left_crop, right_crop, path=f"{label}.crop{crop_idx}")
        assert_nested_equal(left_mask, right_mask, path=f"{label}.mask{crop_idx}")


def assert_selection_invariants(reference, candidate, spectra, lengths, *, label: str) -> None:
    if len(reference) != len(candidate):
        raise AssertionError(
            f"{label}: crop count mismatch {len(reference)} != {len(candidate)}"
        )

    lengths_cpu = lengths.squeeze(-1).detach().cpu().tolist()
    for crop_idx, ((ref_crop, ref_mask), (crop, mask)) in enumerate(
        zip(reference, candidate)
    ):
        if crop.shape != ref_crop.shape:
            raise AssertionError(
                f"{label}.crop{crop_idx}: shape mismatch {crop.shape} != {ref_crop.shape}"
            )
        if mask.shape != ref_mask.shape:
            raise AssertionError(
                f"{label}.mask{crop_idx}: shape mismatch {mask.shape} != {ref_mask.shape}"
            )
        if not torch.equal(mask, ref_mask):
            raise AssertionError(f"{label}.mask{crop_idx}: selected counts differ")

        for sample_idx in range(crop.shape[0]):
            valid = crop[sample_idx][~mask[sample_idx]]
            original = spectra[sample_idx, : int(lengths_cpu[sample_idx])]
            cursor = 0
            for peak in valid:
                matches = torch.nonzero(
                    (original[cursor:] == peak).all(dim=-1), as_tuple=False
                ).flatten()
                if matches.numel() == 0:
                    raise AssertionError(
                        f"{label}.crop{crop_idx}.sample{sample_idx}: "
                        "selected peak is not an ordered valid input peak"
                    )
                cursor += int(matches[0].item()) + 1


def time_call(fn, *, device: torch.device, warmup: int, iters: int) -> tuple[float, float]:
    samples: list[float] = []
    for idx in range(warmup + iters):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        elapsed = time.perf_counter() - start
        if idx >= warmup:
            samples.append(elapsed)
    return statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


class FastRandomWindowAugmentation(RandomWindowAugmentation):
    """Bit-exact candidate for RandomWindowAugmentation with less index memory."""

    def _random_window(self, spectra, lengths, window_sizes, padded_length=None):
        batch_size, seq_len, embed_dim = spectra.shape
        window_sizes = torch.min(lengths, window_sizes)
        max_window_start = lengths - window_sizes

        window_starts = (
            torch.rand(batch_size, device=spectra.device) * max_window_start
        ).long()

        all_indices = torch.arange(seq_len, device=spectra.device, dtype=lengths.dtype)
        all_indices = all_indices.expand(batch_size, -1)

        window_mask = (all_indices >= window_starts.unsqueeze(1)) & (
            all_indices < (window_starts + window_sizes).unsqueeze(1)
        )

        crop_length = (
            int(padded_length) if padded_length is not None else int(window_sizes.max())
        )
        cropped_spectra = spectra.new_full(
            (batch_size, crop_length, embed_dim),
            self.padding_value,
        )
        short_mask = all_indices[:, :crop_length] < window_sizes.unsqueeze(1)
        cropped_spectra[short_mask] = spectra[window_mask]

        return cropped_spectra, ~short_mask


class LegacyRandomSelectionAugmentation(RandomWindowAugmentation):
    """Frozen copy of pre-optimization random selection for equality checks."""

    def _sample_selection_indices(self, spectra, lengths, sample_idx, num_select):
        valid_length = lengths[sample_idx].item()
        return torch.randperm(valid_length, device=spectra.device)[:num_select]

    def _random_selection(self, spectra, lengths, selection_sizes, padded_length=None):
        batch_size, seq_len, embed_dim = spectra.shape
        crop_length = (
            int(padded_length)
            if padded_length is not None
            else int(selection_sizes.max().item())
        )
        selected_spectra = torch.full(
            (batch_size, crop_length, embed_dim),
            self.padding_value,
            dtype=spectra.dtype,
            device=spectra.device,
        )

        padding_mask = torch.ones(
            (batch_size, crop_length),
            dtype=torch.bool,
            device=spectra.device,
        )

        for i in range(batch_size):
            valid_length = lengths[i].item()
            num_select = selection_sizes[i].item()

            selection_mask = torch.zeros(
                valid_length,
                dtype=torch.bool,
                device=spectra.device,
            )
            selection_indices = self._sample_selection_indices(
                spectra, lengths, i, num_select
            )
            selection_mask[selection_indices] = True
            selected_spectra[i, :num_select] = spectra[i, :valid_length][selection_mask]
            padding_mask[i, :num_select] = False

        return selected_spectra, padding_mask

    def __call__(self, spectra, lengths, rand_size=False):
        crops = []
        lengths = lengths.squeeze(-1)
        if any(lengths < 1):
            print("Warning: Found empty spectrum. Can lead to unexpected behaviour. ")

        global_sizes_list = self._sample_crop_sizes(
            lengths,
            self.num_global_crops,
            local=False,
            rand_size=rand_size,
        )
        local_sizes_list = self._sample_crop_sizes(
            lengths,
            self.num_local_crops,
            local=True,
            rand_size=rand_size,
        )

        global_padded_length = self._shared_padded_length(global_sizes_list)
        local_padded_length = self._shared_padded_length(local_sizes_list)

        for global_sizes in global_sizes_list:
            crops.append(
                self._random_selection(
                    spectra,
                    lengths,
                    global_sizes,
                    padded_length=global_padded_length,
                )
            )

        for local_sizes in local_sizes_list:
            crops.append(
                self._random_selection(
                    spectra,
                    lengths,
                    local_sizes,
                    padded_length=local_padded_length,
                )
            )

        return crops


class LegacyIntensityWeightedSelectionAugmentation(LegacyRandomSelectionAugmentation):
    """Frozen copy of pre-optimization intensity-weighted selection."""

    def __init__(
        self,
        global_crops_scale=(0.5, 0.9),
        local_crops_scale=(0.3, 0.5),
        num_global_crops=2,
        num_local_crops=5,
        padding_value=0,
        intensity_alpha=1.0,
        intensity_eps=0.01,
    ):
        super().__init__(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            padding_value=padding_value,
        )
        self.intensity_alpha = intensity_alpha
        self.intensity_eps = intensity_eps

    def _sample_selection_indices(self, spectra, lengths, sample_idx, num_select):
        valid_length = lengths[sample_idx].item()
        intensities = spectra[sample_idx, :valid_length, 1].float().clamp(min=0)
        weights = (intensities + self.intensity_eps).pow(self.intensity_alpha)
        if not torch.any(weights > 0):
            weights = torch.ones_like(weights)
        return torch.multinomial(weights, num_samples=num_select, replacement=False)



class FastRandomSelectionAugmentation(RandomSelectionAugmentation):
    """Bit-exact candidate for random-selection augmentation.

    This preserves legacy random sampling by calling torch.randperm once per
    sample in the same order. The speedup comes from removing per-sample
    .item() synchronizations on CUDA and vectorizing the final selected-peak copy.
    """

    def _sample_selection_indices_fast(
        self,
        spectra: torch.Tensor,
        sample_idx: int,
        valid_length: int,
        num_select: int,
    ) -> torch.Tensor:
        del sample_idx
        return torch.randperm(valid_length, device=spectra.device)[:num_select]

    def _random_selection(self, spectra, lengths, selection_sizes, padded_length=None):
        batch_size, seq_len, embed_dim = spectra.shape
        crop_length = (
            int(padded_length)
            if padded_length is not None
            else int(selection_sizes.max().item())
        )
        selected_spectra = torch.full(
            (batch_size, crop_length, embed_dim),
            self.padding_value,
            dtype=spectra.dtype,
            device=spectra.device,
        )

        valid_lengths = lengths.detach().cpu().tolist()
        num_selects = selection_sizes.detach().cpu().tolist()

        selection_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=spectra.device,
        )

        for sample_idx, (valid_length, num_select) in enumerate(
            zip(valid_lengths, num_selects)
        ):
            valid_length = int(valid_length)
            num_select = int(num_select)
            if num_select <= 0:
                continue
            selection_indices = self._sample_selection_indices_fast(
                spectra,
                sample_idx,
                valid_length,
                num_select,
            )
            selection_mask[sample_idx, selection_indices] = True

        selected_counts = selection_sizes.to(device=spectra.device, dtype=torch.long)
        total_selected = int(selected_counts.sum().item())
        if total_selected:
            selected_values = spectra[selection_mask]
            row_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=spectra.device),
                selected_counts,
            )
            starts = torch.cumsum(selected_counts, dim=0) - selected_counts
            flat_positions = torch.arange(total_selected, device=spectra.device)
            col_ids = flat_positions - torch.repeat_interleave(starts, selected_counts)
            selected_spectra[row_ids, col_ids] = selected_values

        crop_positions = torch.arange(crop_length, device=spectra.device)
        padding_mask = crop_positions.unsqueeze(0) >= selected_counts.unsqueeze(1)
        return selected_spectra, padding_mask


class FastIntensityWeightedSelectionAugmentation(FastRandomSelectionAugmentation):
    """Bit-exact candidate for intensity-weighted random-selection augmentation."""

    def __init__(
        self,
        global_crops_scale=(0.5, 0.9),
        local_crops_scale=(0.3, 0.5),
        num_global_crops=2,
        num_local_crops=5,
        padding_value=0,
        intensity_alpha=1.0,
        intensity_eps=0.01,
    ):
        super().__init__(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            padding_value=padding_value,
        )
        self.intensity_alpha = intensity_alpha
        self.intensity_eps = intensity_eps

    def _sample_selection_indices_fast(
        self,
        spectra: torch.Tensor,
        sample_idx: int,
        valid_length: int,
        num_select: int,
    ) -> torch.Tensor:
        intensities = spectra[sample_idx, :valid_length, 1].float().clamp(min=0)
        weights = (intensities + self.intensity_eps).pow(self.intensity_alpha)
        if not torch.any(weights > 0):
            weights = torch.ones_like(weights)
        return torch.multinomial(weights, num_samples=num_select, replacement=False)


class BatchedRandomSelectionAugmentation(RandomSelectionAugmentation):
    """Distribution-equivalent random subset sampler.

    This is not bit-exact to the legacy randperm-per-sample implementation. It
    samples independent random scores for every valid peak, takes the k lowest
    scores per spectrum, then sorts the selected peak indices back into original
    m/z order. For continuous random scores, this is a uniform subset without
    replacement for each spectrum.
    """

    def _random_selection(self, spectra, lengths, selection_sizes, padded_length=None):
        batch_size, seq_len, embed_dim = spectra.shape
        selection_sizes = torch.min(lengths, selection_sizes).to(
            device=spectra.device, dtype=torch.long
        )
        crop_length = (
            int(padded_length)
            if padded_length is not None
            else int(selection_sizes.max().item())
        )
        selected_spectra = torch.full(
            (batch_size, crop_length, embed_dim),
            self.padding_value,
            dtype=spectra.dtype,
            device=spectra.device,
        )
        if crop_length == 0:
            padding_mask = torch.ones(
                (batch_size, 0), dtype=torch.bool, device=spectra.device
            )
            return selected_spectra, padding_mask

        positions = torch.arange(seq_len, device=spectra.device)
        valid_mask = positions.unsqueeze(0) < lengths.to(spectra.device).unsqueeze(1)
        scores = torch.rand((batch_size, seq_len), device=spectra.device)
        scores = scores.masked_fill(~valid_mask, float("inf"))

        top_indices_by_score = torch.topk(
            scores,
            k=crop_length,
            dim=1,
            largest=False,
            sorted=True,
        ).indices

        keep_positions = torch.arange(crop_length, device=spectra.device)
        rank_mask = keep_positions.unsqueeze(0) < selection_sizes.unsqueeze(1)
        selection_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=spectra.device,
        )
        selection_mask.scatter_(1, top_indices_by_score, rank_mask)

        total_selected = int(selection_sizes.sum().item())
        if total_selected:
            selected_values = spectra[selection_mask]
            row_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=spectra.device),
                selection_sizes,
            )
            starts = torch.cumsum(selection_sizes, dim=0) - selection_sizes
            flat_positions = torch.arange(total_selected, device=spectra.device)
            col_ids = flat_positions - torch.repeat_interleave(starts, selection_sizes)
            selected_spectra[row_ids, col_ids] = selected_values

        padding_mask = keep_positions.unsqueeze(0) >= selection_sizes.unsqueeze(1)
        return selected_spectra, padding_mask


class BatchedIntensityWeightedSelectionAugmentation(IntensityWeightedSelectionAugmentation):
    """Distribution-equivalent weighted subset sampler using Gumbel-top-k."""

    def _random_selection(self, spectra, lengths, selection_sizes, padded_length=None):
        batch_size, seq_len, embed_dim = spectra.shape
        selection_sizes = torch.min(lengths, selection_sizes).to(
            device=spectra.device, dtype=torch.long
        )
        crop_length = (
            int(padded_length)
            if padded_length is not None
            else int(selection_sizes.max().item())
        )
        selected_spectra = torch.full(
            (batch_size, crop_length, embed_dim),
            self.padding_value,
            dtype=spectra.dtype,
            device=spectra.device,
        )
        if crop_length == 0:
            padding_mask = torch.ones(
                (batch_size, 0), dtype=torch.bool, device=spectra.device
            )
            return selected_spectra, padding_mask

        positions = torch.arange(seq_len, device=spectra.device)
        valid_mask = positions.unsqueeze(0) < lengths.to(spectra.device).unsqueeze(1)

        weights = spectra[..., 1].float().clamp(min=0)
        weights = (weights + self.intensity_eps).pow(self.intensity_alpha)
        weights = weights.masked_fill(~valid_mask, 0.0)
        weights = torch.where(
            torch.any(weights > 0, dim=1, keepdim=True),
            weights,
            valid_mask.to(weights.dtype),
        )

        log_weights = torch.full_like(weights, -float("inf"))
        positive_mask = weights > 0
        log_weights[positive_mask] = weights[positive_mask].log()
        exponential_noise = torch.empty_like(weights).exponential_()
        exponential_noise.clamp_min_(torch.finfo(weights.dtype).tiny)
        scores = log_weights - exponential_noise.log()

        top_indices_by_score = torch.topk(
            scores,
            k=crop_length,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        keep_positions = torch.arange(crop_length, device=spectra.device)
        rank_mask = keep_positions.unsqueeze(0) < selection_sizes.unsqueeze(1)
        selection_mask = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=spectra.device,
        )
        selection_mask.scatter_(1, top_indices_by_score, rank_mask)

        total_selected = int(selection_sizes.sum().item())
        if total_selected:
            selected_values = spectra[selection_mask]
            row_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=spectra.device),
                selection_sizes,
            )
            starts = torch.cumsum(selection_sizes, dim=0) - selection_sizes
            flat_positions = torch.arange(total_selected, device=spectra.device)
            col_ids = flat_positions - torch.repeat_interleave(starts, selection_sizes)
            selected_spectra[row_ids, col_ids] = selected_values

        padding_mask = keep_positions.unsqueeze(0) >= selection_sizes.unsqueeze(1)
        return selected_spectra, padding_mask


def fast_subsample_max_peaks(mz_tensor, int_tensor, max_peaks=300):
    """Equivalent to legacy argsort-descending top-k for non-tied intensities."""
    if mz_tensor.numel() <= max_peaks:
        return mz_tensor, int_tensor
    highest_indices = torch.topk(int_tensor, k=max_peaks, largest=True, sorted=False).indices
    highest_indices_original_order, _ = highest_indices.sort()
    return mz_tensor[highest_indices_original_order], int_tensor[highest_indices_original_order]


def fast_default_filter_peaks(
    mz_tensor_b,
    intensity_tensor_b,
    max_peaks,
    intensity_scaling,
    min_mz,
    max_mz,
):
    mz_mask = (mz_tensor_b >= min_mz) & (mz_tensor_b <= max_mz)
    mz_tensor_b = mz_tensor_b[mz_mask]
    intensity_tensor_b = intensity_tensor_b[mz_mask]

    if len(mz_tensor_b) == 0:
        return mz_tensor_b, intensity_tensor_b

    if len(mz_tensor_b) > max_peaks:
        mz_tensor_b, intensity_tensor_b = fast_subsample_max_peaks(
            mz_tensor_b, intensity_tensor_b, max_peaks
        )

    if intensity_scaling == "minmax":
        intensity_tensor_b = minmax_scale(intensity_tensor_b)
    elif intensity_scaling == "basepeak":
        intensity_tensor_b = basepeak_scale(intensity_tensor_b)
    elif intensity_scaling == "none":
        pass
    else:
        raise ValueError(f"Unknown intensity scaling method: '{intensity_scaling}'")

    return mz_tensor_b, intensity_tensor_b


def process_precursor_info_copy(
    item: dict[str, Any],
    precursor_mz_name: str | bool,
    precursor_mass_name: str | bool,
) -> dict[str, Any]:
    out = dict(item)
    if precursor_mz_name and precursor_mz_name != "precursor_mz":
        out["precursor_mz"] = out[precursor_mz_name]
    if precursor_mass_name and precursor_mass_name != "precursor_mass":
        out["precursor_mass"] = out[precursor_mass_name]
    if precursor_mass_name and not precursor_mz_name:
        out["precursor_mz"] = out["precursor_mass"] / out["precursor_charge"]
    if precursor_mz_name and not precursor_mass_name:
        out["precursor_mass"] = out["precursor_mz"] * out["precursor_charge"]
    return out


def fast_pad_peaks(
    batch: Iterable[dict[Any, Any]],
    precision: torch.dtype = torch.float32,
    max_peaks: int = 300,
    precursor_mz_name: str | bool = "precursor_mz",
    precursor_mass_name: str | bool = "precursor_mass",
    filter_method: str = "default",
    intensity_scaling: str = "minmax",
    min_mz: float = 50.0,
    max_mz: float = 2500.0,
    min_intensity: float = 0.01,
    remove_precursor_tol: float = 2,
    min_peaks: int = 0,
) -> dict[str, Any]:
    mz_tensors = []
    int_tensors = []
    kept_items = []
    lengths_list = []

    for item in batch:
        processed = process_precursor_info_copy(
            item,
            precursor_mz_name=precursor_mz_name,
            precursor_mass_name=precursor_mass_name,
        )
        mz_tensor_b = _as_tensor(processed["mz_array"], precision)
        intensity_tensor_b = _as_tensor(processed["intensity_array"], precision)

        if filter_method == "casanovo":
            precursor_mz = processed.get("precursor_mz")
            if precursor_mz is None:
                raise ValueError("Precursor m/z not found in batch item.")
            mz_tensor_b, intensity_tensor_b = casanovo_filter_peaks(
                mz_tensor_b,
                intensity_tensor_b,
                precursor_mz,
                min_mz,
                max_mz,
                min_intensity,
                remove_precursor_tol,
                max_peaks,
            )
        elif filter_method == "default":
            mz_tensor_b, intensity_tensor_b = fast_default_filter_peaks(
                mz_tensor_b,
                intensity_tensor_b,
                max_peaks,
                intensity_scaling,
                min_mz,
                max_mz,
            )
        else:
            raise ValueError(f"Unknown spectrum filtering method: '{filter_method}'")

        if len(mz_tensor_b) < min_peaks:
            continue

        mz_tensors.append(mz_tensor_b)
        int_tensors.append(intensity_tensor_b)
        lengths_list.append(len(mz_tensor_b))
        kept_items.append(
            {
                key: value
                for key, value in processed.items()
                if key not in {"mz_array", "intensity_array"}
            }
        )

    batch_size = len(kept_items)
    if batch_size == 0:
        raise ValueError("fast_pad_peaks received no spectra after filtering.")

    max_len = max(lengths_list)
    mz_array = torch.zeros((batch_size, max_len), dtype=precision)
    intensity_array = torch.zeros((batch_size, max_len), dtype=precision)
    for row_idx, (mz_tensor_b, intensity_tensor_b) in enumerate(
        zip(mz_tensors, int_tensors)
    ):
        n_peaks = len(mz_tensor_b)
        mz_array[row_idx, :n_peaks] = mz_tensor_b
        intensity_array[row_idx, :n_peaks] = intensity_tensor_b

    out = torch.utils.data.default_collate(kept_items)
    out["mz_array"] = mz_array
    out["intensity_array"] = intensity_array
    out["peak_lengths"] = torch.tensor(lengths_list, dtype=torch.int32).unsqueeze(1)

    for key, val in out.items():
        if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
            out[key] = val.type(precision)

    return out


def make_synthetic_spectra(
    batch_size: int,
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    min_len = max(8, seq_len // 3)
    lengths = torch.randint(min_len, seq_len + 1, (batch_size, 1), generator=generator)
    mz = torch.rand((batch_size, seq_len), generator=generator, dtype=dtype)
    mz, _ = mz.sort(dim=1)
    intensity = torch.rand((batch_size, seq_len), generator=generator, dtype=dtype)
    spectra = torch.stack([mz, intensity], dim=-1).to(device=device)
    return spectra, lengths.to(device=device)


def make_synthetic_peak_batch(
    batch_size: int,
    max_raw_peaks: int,
    *,
    dtype: torch.dtype,
    seed: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = []
    for idx in range(batch_size):
        n_peaks = int(
            torch.randint(
                max(16, max_raw_peaks // 2),
                max_raw_peaks + 1,
                (1,),
                generator=generator,
            ).item()
        )
        mz = 2500 * torch.rand(n_peaks, generator=generator, dtype=dtype)
        mz, _ = mz.sort()

        # Add a tiny monotonic term to avoid exact ties in topk-vs-argsort tests.
        intensity = torch.rand(n_peaks, generator=generator, dtype=dtype)
        intensity = intensity + torch.linspace(0, 1e-7, n_peaks, dtype=dtype)

        precursor_mz = torch.tensor(400.0 + idx, dtype=dtype)
        precursor_charge = torch.tensor((idx % 4) + 1, dtype=torch.int64)
        batch.append(
            {
                "mz_array": mz,
                "intensity_array": intensity,
                "precursor_mz": precursor_mz,
                "precursor_charge": precursor_charge,
                "scan_id": torch.tensor(idx, dtype=torch.int64),
            }
        )
    return batch


def benchmark_augmentation(args, device: torch.device) -> None:
    dtype = torch.float32
    spectra, lengths = make_synthetic_spectra(
        args.batch_size,
        args.seq_len,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    kwargs = dict(
        global_crops_scale=(0.95, 0.95),
        local_crops_scale=(0.2, 0.2),
        num_global_crops=args.num_global_crops,
        num_local_crops=args.num_local_crops,
        padding_value=0,
    )

    print("\n[augmentation] exactness checks")

    legacy_window = RandomWindowAugmentation(**kwargs)
    fast_window = FastRandomWindowAugmentation(**kwargs)
    seed_all(args.seed + 1, device)
    old_window = legacy_window(spectra, lengths, rand_size=args.rand_size)
    seed_all(args.seed + 1, device)
    new_window = fast_window(spectra, lengths, rand_size=args.rand_size)
    assert_crops_equal(old_window, new_window, label="window")
    print("  window: exact")

    legacy_random = LegacyRandomSelectionAugmentation(**kwargs)
    fast_random = RandomSelectionAugmentation(**kwargs)
    batched_random = BatchedRandomSelectionAugmentation(**kwargs)
    seed_all(args.seed + 2, device)
    old_random = legacy_random(spectra, lengths, rand_size=args.rand_size)
    seed_all(args.seed + 2, device)
    new_random = fast_random(spectra, lengths, rand_size=args.rand_size)
    assert_crops_equal(old_random, new_random, label="random")
    print("  random selection: exact")

    seed_all(args.seed + 4, device)
    reference_random = fast_random(spectra, lengths, rand_size=args.rand_size)
    seed_all(args.seed + 4, device)
    equivalent_random = batched_random(spectra, lengths, rand_size=args.rand_size)
    assert_selection_invariants(
        reference_random,
        equivalent_random,
        spectra,
        lengths,
        label="random_batched_equivalent",
    )
    print("  random batched-equivalent: invariants ok")

    legacy_weighted = LegacyIntensityWeightedSelectionAugmentation(
        **kwargs,
        intensity_alpha=1.0,
        intensity_eps=0.01,
    )
    fast_weighted = IntensityWeightedSelectionAugmentation(
        **kwargs,
        intensity_alpha=1.0,
        intensity_eps=0.01,
    )
    batched_weighted = BatchedIntensityWeightedSelectionAugmentation(
        **kwargs,
        intensity_alpha=1.0,
        intensity_eps=0.01,
    )
    seed_all(args.seed + 3, device)
    old_weighted = legacy_weighted(spectra, lengths, rand_size=args.rand_size)
    seed_all(args.seed + 3, device)
    new_weighted = fast_weighted(spectra, lengths, rand_size=args.rand_size)
    assert_crops_equal(old_weighted, new_weighted, label="weighted")
    print("  intensity-weighted selection: exact")

    seed_all(args.seed + 5, device)
    reference_weighted = fast_weighted(spectra, lengths, rand_size=args.rand_size)
    seed_all(args.seed + 5, device)
    equivalent_weighted = batched_weighted(spectra, lengths, rand_size=args.rand_size)
    assert_selection_invariants(
        reference_weighted,
        equivalent_weighted,
        spectra,
        lengths,
        label="weighted_batched_equivalent",
    )
    print("  intensity-weighted batched-equivalent: invariants ok")

    print("\n[augmentation] microbenchmarks")
    benches = [
        ("window legacy", lambda: legacy_window(spectra, lengths, rand_size=args.rand_size)),
        ("window fast", lambda: fast_window(spectra, lengths, rand_size=args.rand_size)),
        ("random legacy", lambda: legacy_random(spectra, lengths, rand_size=args.rand_size)),
        ("random fast", lambda: fast_random(spectra, lengths, rand_size=args.rand_size)),
        (
            "random batched-eq",
            lambda: batched_random(spectra, lengths, rand_size=args.rand_size),
        ),
        (
            "weighted legacy",
            lambda: legacy_weighted(spectra, lengths, rand_size=args.rand_size),
        ),
        (
            "weighted fast",
            lambda: fast_weighted(spectra, lengths, rand_size=args.rand_size),
        ),
        (
            "weighted batched-eq",
            lambda: batched_weighted(spectra, lengths, rand_size=args.rand_size),
        ),
    ]
    results = []
    for name, fn in benches:
        mean_s, std_s = time_call(
            fn,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append((name, mean_s, std_s))
        print(f"  {name:16s} {1000 * mean_s:9.3f} ms +/- {1000 * std_s:7.3f}")

    result_map = {name: mean for name, mean, _ in results}
    if "random fast" in result_map:
        speedup = result_map["random legacy"] / result_map["random fast"]
        print(f"  random speedup: {speedup:.2f}x")
    if "random batched-eq" in result_map:
        speedup = result_map["random legacy"] / result_map["random batched-eq"]
        print(f"  random batched-equivalent speedup: {speedup:.2f}x")
    if "weighted fast" in result_map:
        speedup = result_map["weighted legacy"] / result_map["weighted fast"]
        print(f"  weighted speedup: {speedup:.2f}x")
    if "weighted batched-eq" in result_map:
        speedup = result_map["weighted legacy"] / result_map["weighted batched-eq"]
        print(f"  weighted batched-equivalent speedup: {speedup:.2f}x")


def benchmark_collate(args) -> None:
    dtype = torch.float32
    batch = make_synthetic_peak_batch(
        args.batch_size,
        args.raw_peaks,
        dtype=dtype,
        seed=args.seed + 10,
    )
    kwargs = dict(
        precision=dtype,
        max_peaks=args.max_peaks,
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
        filter_method="default",
        intensity_scaling="minmax",
        min_mz=0.0,
        max_mz=2500.0,
        min_intensity=0.0001,
        remove_precursor_tol=2,
        min_peaks=0,
    )

    print("\n[collate] exactness check")
    old_out = legacy_pad_peaks(clone_batch(batch), **kwargs)
    new_out = fast_pad_peaks(clone_batch(batch), **kwargs)
    assert_nested_equal(old_out, new_out, path="pad_peaks")
    print("  pad_peaks: exact on synthetic no-tie data")

    print("\n[collate] microbenchmarks")
    benches = [
        ("pad_peaks legacy", lambda: legacy_pad_peaks(clone_batch(batch), **kwargs)),
        ("pad_peaks fast", lambda: fast_pad_peaks(clone_batch(batch), **kwargs)),
    ]
    results = []
    for name, fn in benches:
        mean_s, std_s = time_call(
            fn,
            device=torch.device("cpu"),
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append((name, mean_s, std_s))
        print(f"  {name:16s} {1000 * mean_s:9.3f} ms +/- {1000 * std_s:7.3f}")

    result_map = {name: mean for name, mean, _ in results}
    speedup = result_map["pad_peaks legacy"] / result_map["pad_peaks fast"]
    print(f"  collate speedup: {speedup:.2f}x")


def resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [move_to_device(val, device) for val in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(val, device) for val in value)
    return value


def load_global_args_from_config(
    config_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_mem: bool,
):
    from src.parse_args import get_args_parser, sanity_checks

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --real-loader") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    conf_parser = argparse.ArgumentParser("Config parser", add_help=False)
    conf_parser.add_argument("--config", type=str, help="config path")
    parser = get_args_parser(conf_parser)
    parser.set_defaults(**config)

    cli_args = [
        "--config",
        str(config_path),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--pin_mem",
        str(int(pin_mem)),
        "--num_devices",
        "1",
        "--num_nodes",
        "1",
        "--log_wandb",
        "0",
        "--downstream_task",
        "none",
    ]
    global_args = parser.parse_args(cli_args)
    sanity_checks(global_args)

    pretrain_path = resolve_repo_path(global_args.pretrain_config)
    if pretrain_path is None:
        raise ValueError("Config does not define pretrain_config")
    with pretrain_path.open("r", encoding="utf-8") as handle:
        pretrain_config = yaml.safe_load(handle)

    return global_args, pretrain_config


def get_config_num_workers(config_path: Path) -> int:
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return int(config.get("num_workers", 0))
    except Exception:
        return 0


def make_aug_pair_from_task(task: dict[str, Any]):
    kwargs = dict(
        global_crops_scale=tuple(task["global_crops_scale"]),
        local_crops_scale=tuple(task["local_crops_scale"]),
        num_global_crops=task["num_global_crops"],
        num_local_crops=task["num_local_crops"],
        padding_value=0,
    )
    mode = task["selection_mode"]
    if mode == "window":
        return (
            mode,
            RandomWindowAugmentation(**kwargs),
            FastRandomWindowAugmentation(**kwargs),
        )
    if mode == "random":
        return (
            mode,
            LegacyRandomSelectionAugmentation(**kwargs),
            RandomSelectionAugmentation(**kwargs),
        )
    if mode == "random_batched":
        return (
            mode,
            LegacyRandomSelectionAugmentation(**kwargs),
            BatchedRandomSelectionAugmentation(**kwargs),
        )
    if mode == "random_intensity_weighted":
        weighted_kwargs = dict(
            intensity_alpha=task["intensity_alpha"],
            intensity_eps=task["intensity_eps"],
        )
        return (
            mode,
            LegacyIntensityWeightedSelectionAugmentation(**kwargs, **weighted_kwargs),
            IntensityWeightedSelectionAugmentation(**kwargs, **weighted_kwargs),
        )
    if mode == "random_intensity_weighted_batched":
        weighted_kwargs = dict(
            intensity_alpha=task["intensity_alpha"],
            intensity_eps=task["intensity_eps"],
        )
        return (
            mode,
            LegacyIntensityWeightedSelectionAugmentation(**kwargs, **weighted_kwargs),
            BatchedIntensityWeightedSelectionAugmentation(**kwargs, **weighted_kwargs),
        )
    raise ValueError(f"Unsupported selection_mode={mode!r}")


def benchmark_real_loader(args, device: torch.device) -> None:
    from src import utils

    config_path = args.config.resolve()
    worker_values = args.loader_workers
    if not worker_values:
        worker_values = [get_config_num_workers(config_path)]

    print("\n[real loader] Lance DataLoader/collate timing")
    for worker_value in worker_values:
        global_args, pretrain_config = load_global_args_from_config(
            config_path,
            batch_size=args.batch_size,
            num_workers=int(worker_value),
            pin_mem=bool(args.loader_pin_mem),
        )
        data_module = utils.get_lance_data_module(
            global_args,
            pretrain_config,
            global_args.max_peaks,
            seed=global_args.seed,
            include_test=False,
        )

        setup_start = time.perf_counter()
        data_module.setup("fit")
        setup_s = time.perf_counter() - setup_start

        loader_start = time.perf_counter()
        loader = data_module.train_dataloader()
        loader_s = time.perf_counter() - loader_start

        iter_start = time.perf_counter()
        iterator = iter(loader)
        iter_s = time.perf_counter() - iter_start

        first_start = time.perf_counter()
        first_batch = next(iterator)
        first_s = time.perf_counter() - first_start
        total_startup_s = setup_s + loader_s + iter_s + first_s

        print(
            f"  workers={int(worker_value):2d} pin_mem={int(args.loader_pin_mem)} "
            f"startup total={total_startup_s:6.3f}s "
            f"setup={setup_s:6.3f}s loader={loader_s:6.3f}s "
            f"iter={iter_s:6.3f}s first={first_s:6.3f}s"
        )

        iterator_state = {"iterator": iterator}
        del first_batch

        def fetch_batch():
            try:
                return next(iterator_state["iterator"])
            except StopIteration:
                iterator_state["iterator"] = iter(loader)
                return next(iterator_state["iterator"])

        mean_s, std_s = time_call(
            fetch_batch,
            device=torch.device("cpu"),
            warmup=args.loader_warmup,
            iters=args.loader_iters,
        )
        print(
            f"  workers={int(worker_value):2d} pin_mem={int(args.loader_pin_mem)} "
            f"steady fetch {1000 * mean_s:7.3f} ms +/- {1000 * std_s:7.3f}"
        )

        if device.type == "cuda":
            def fetch_and_copy_batch():
                return move_to_device(fetch_batch(), device)

            mean_s, std_s = time_call(
                fetch_and_copy_batch,
                device=device,
                warmup=args.loader_warmup,
                iters=args.loader_iters,
            )
            print(
                f"  workers={int(worker_value):2d} pin_mem={int(args.loader_pin_mem)} "
                f"fetch+copy {1000 * mean_s:4.3f} ms +/- {1000 * std_s:7.3f}"
            )

    if device.type != "cuda":
        print("  real-batch augmentation skipped because --device is not cuda")
        return

    last_workers = int(worker_values[-1])
    global_args, pretrain_config = load_global_args_from_config(
        config_path,
        batch_size=args.batch_size,
        num_workers=last_workers,
        pin_mem=bool(args.loader_pin_mem),
    )
    task = pretrain_config[global_args.pretraining_task]
    data_module = utils.get_lance_data_module(
        global_args,
        pretrain_config,
        global_args.max_peaks,
        seed=global_args.seed,
        include_test=False,
    )
    data_module.setup("fit")
    real_batch = next(iter(data_module.train_dataloader()))
    real_batch = move_to_device(real_batch, device)

    mzab = torch.stack([real_batch["mz_array"], real_batch["intensity_array"]], dim=-1)
    lengths = real_batch["peak_lengths"]
    mode, legacy_aug, fast_aug = make_aug_pair_from_task(task)

    seed_all(args.seed + 20, device)
    old_crops = legacy_aug(mzab, lengths, rand_size=task.get("rand_window_size", False))
    seed_all(args.seed + 20, device)
    new_crops = fast_aug(mzab, lengths, rand_size=task.get("rand_window_size", False))
    if mode in {"random_batched", "random_intensity_weighted_batched"}:
        assert_selection_invariants(
            old_crops,
            new_crops,
            mzab,
            lengths,
            label=f"real_batch_{mode}",
        )
    else:
        assert_crops_equal(old_crops, new_crops, label=f"real_batch_{mode}")

    print(f"\n[real batch augmentation] selection_mode={mode}")
    benches = [
        (
            f"{mode} legacy",
            lambda: legacy_aug(mzab, lengths, rand_size=task.get("rand_window_size", False)),
        ),
        (
            f"{mode} fast",
            lambda: fast_aug(mzab, lengths, rand_size=task.get("rand_window_size", False)),
        ),
    ]
    if mode == "random":
        batched_aug = BatchedRandomSelectionAugmentation(
            global_crops_scale=tuple(task["global_crops_scale"]),
            local_crops_scale=tuple(task["local_crops_scale"]),
            num_global_crops=task["num_global_crops"],
            num_local_crops=task["num_local_crops"],
            padding_value=0,
        )
        seed_all(args.seed + 21, device)
        reference_crops = fast_aug(
            mzab, lengths, rand_size=task.get("rand_window_size", False)
        )
        seed_all(args.seed + 21, device)
        batched_crops = batched_aug(
            mzab, lengths, rand_size=task.get("rand_window_size", False)
        )
        assert_selection_invariants(
            reference_crops,
            batched_crops,
            mzab,
            lengths,
            label="real_batch_random_batched_equivalent",
        )
        benches.append(
            (
                f"{mode} batched-eq",
                lambda: batched_aug(
                    mzab, lengths, rand_size=task.get("rand_window_size", False)
                ),
            )
        )
    if mode == "random_intensity_weighted":
        weighted_kwargs = dict(
            intensity_alpha=task["intensity_alpha"],
            intensity_eps=task["intensity_eps"],
        )
        batched_aug = BatchedIntensityWeightedSelectionAugmentation(
            global_crops_scale=tuple(task["global_crops_scale"]),
            local_crops_scale=tuple(task["local_crops_scale"]),
            num_global_crops=task["num_global_crops"],
            num_local_crops=task["num_local_crops"],
            padding_value=0,
            **weighted_kwargs,
        )
        seed_all(args.seed + 22, device)
        reference_crops = fast_aug(
            mzab, lengths, rand_size=task.get("rand_window_size", False)
        )
        seed_all(args.seed + 22, device)
        batched_crops = batched_aug(
            mzab, lengths, rand_size=task.get("rand_window_size", False)
        )
        assert_selection_invariants(
            reference_crops,
            batched_crops,
            mzab,
            lengths,
            label="real_batch_weighted_batched_equivalent",
        )
        benches.append(
            (
                f"{mode} batched-eq",
                lambda: batched_aug(
                    mzab, lengths, rand_size=task.get("rand_window_size", False)
                ),
            )
        )
    results = []
    for name, fn in benches:
        mean_s, std_s = time_call(
            fn,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append((name, mean_s, std_s))
        print(f"  {name:16s} {1000 * mean_s:9.3f} ms +/- {1000 * std_s:7.3f}")
    speedup = results[0][1] / results[1][1]
    print(f"  real-batch augmentation speedup: {speedup:.2f}x")


def print_config_diagnostics(config_path: Path | None) -> None:
    if config_path is None:
        return
    try:
        import yaml
    except ImportError:
        print("\n[config] PyYAML not installed; skipping config diagnostics")
        return

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    print(f"\n[config] {config_path}")
    for key in ("num_workers", "batch_size", "precision", "pin_mem", "log_wandb"):
        if key in config:
            print(f"  {key}: {config[key]}")

    pretrain_config = config.get("pretrain_config")
    if pretrain_config:
        pretrain_path = (config_path.parent.parent / pretrain_config).resolve()
        if not pretrain_path.exists():
            pretrain_path = (REPO_ROOT / pretrain_config).resolve()
        if pretrain_path.exists():
            with pretrain_path.open("r", encoding="utf-8") as handle:
                pt_config = yaml.safe_load(handle)
            task_name = config.get("pretraining_task", "dion")
            task = pt_config.get(task_name, {})
            print(f"  pretrain_config: {pretrain_path}")
            for key in (
                "selection_mode",
                "num_global_crops",
                "num_local_crops",
                "mix_distractor_enabled",
                "batch_size",
                "num_workers",
            ):
                if key in task:
                    print(f"  {task_name}.{key}: {task[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--raw-peaks", type=int, default=700)
    parser.add_argument("--max-peaks", type=int, default=300)
    parser.add_argument("--num-global-crops", type=int, default=2)
    parser.add_argument("--num-local-crops", type=int, default=5)
    parser.add_argument("--rand-size", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config_cluster_b" / "master_dion.yaml",
    )
    parser.add_argument("--skip-augmentation", action="store_true")
    parser.add_argument("--skip-collate", action="store_true")
    parser.add_argument(
        "--real-loader",
        action="store_true",
        help="Also time the real ClusterB Lance train loader from --config.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        nargs="*",
        default=None,
        help="Worker counts to sweep for --real-loader. Defaults to config num_workers.",
    )
    parser.add_argument("--loader-pin-mem", type=int, default=0)
    parser.add_argument("--loader-warmup", type=int, default=5)
    parser.add_argument("--loader-iters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[env] torch={torch.__version__}")
    print(f"[env] device={device}")
    if device.type == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(device)}")

    print_config_diagnostics(args.config)

    if not args.skip_augmentation:
        benchmark_augmentation(args, device)
    if not args.skip_collate:
        benchmark_collate(args)
    if args.real_loader:
        benchmark_real_loader(args, device)

    print("\n[notes]")
    print("  Exact candidates preserve legacy outputs under a fixed seed.")
    print("  Random/weighted selection still call one RNG op per sample to preserve exactness.")
    print("  The batched-equivalent sampler preserves random-subset semantics,")
    print("  but is not bit-exact to the current randperm RNG stream.")


if __name__ == "__main__":
    main()
