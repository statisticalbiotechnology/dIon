import math
import warnings

import torch


class RandomWindowAugmentation:
    def __init__(
        self,
        global_crops_scale=(0.5, 0.9),
        local_crops_scale=(0.3, 0.5),
        num_global_crops=2,
        num_local_crops=5,
        padding_value=0,
        fixed_global_padded_length=None,
        fixed_local_padded_length=None,
    ):
        """
        Usage:
            aug = RandomWindowAugmentation()

            aug_spectra = augmentation(spectra, lengths)

        Args:
            global_crops_scale (tuple): Scale range for global crops.
            local_crops_scale (tuple): Scale range for local crops.
            num_global_crops (int): Number of global crops.
            num_local_crops (int): Number of local crops.
            padding_value (int): Value to use for padding.
            fixed_global_padded_length (int, optional): Fixed output length for global crops.
            fixed_local_padded_length (int, optional): Fixed output length for local crops.
        """
        self.global_high, self.global_low = global_crops_scale[1], global_crops_scale[0]
        self.global_crop_interval = self.global_high - self.global_low

        self.local_high, self.local_low = local_crops_scale[1], local_crops_scale[0]
        self.local_crop_interval = self.local_high - self.local_low

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.padding_value = padding_value
        self.fixed_global_padded_length = fixed_global_padded_length
        self.fixed_local_padded_length = fixed_local_padded_length

    def _random_window(self, spectra, lengths, window_sizes, padded_length=None):
        """
        Perform random cropping on spectra based on specified crop sizes.

        Args:
            spectra (torch.Tensor): Input spectra of shape (batch_size, seq_len, embed_dim).
            lengths (torch.Tensor): Lengths of spectra in the batch.
            window_sizes (torch.Tensor): Crop sizes for each sequence in the batch.

        Returns:
            tuple: Cropped spectra and corresponding padding masks.
        """
        batch_size, seq_len, embed_dim = spectra.shape
        # Find max allowed start index
        window_sizes = torch.min(lengths, window_sizes)
        max_window_start = lengths - window_sizes

        # Set start indices
        window_starts = (
            torch.rand(batch_size, device=spectra.device) * max_window_start
        ).long()

        all_indices = torch.arange(seq_len).type_as(lengths).repeat((batch_size, 1))

        # Construct bool mask to select windows
        window_mask_start = all_indices >= window_starts.unsqueeze(1)
        window_mask_end = all_indices < (window_starts + window_sizes).unsqueeze(1)
        window_mask = window_mask_start & window_mask_end

        crop_length = (
            int(padded_length) if padded_length is not None else int(window_sizes.max())
        )
        cropped_spectra = (
            torch.ones((batch_size, crop_length, embed_dim)).type_as(spectra)
            * self.padding_value
        )
        crop_indices = torch.arange(
            crop_length,
            device=spectra.device,
            dtype=lengths.dtype,
        ).unsqueeze(0)
        short_mask = crop_indices < window_sizes.unsqueeze(1)
        cropped_spectra[short_mask] = spectra[window_mask]

        padding_mask = ~short_mask
        return (cropped_spectra, padding_mask)

    def _random_window_size(self, lengths, local=True):
        interval = self.local_crop_interval if local else self.global_crop_interval
        low = self.local_low if local else self.global_low
        return (
            (torch.rand(lengths.shape[0], device=lengths.device) * interval + low)
            * lengths
        ).long()

    def _get_window_sizes(self, lengths, local=True, random=False):
        if random:
            _window_sizes = self._random_window_size(lengths, local)
        else:
            scale = self.local_high if local else self.global_high
            _window_sizes = (lengths * scale).long()
        return _window_sizes.clamp(min=1)

    def _sample_crop_sizes(self, lengths, num_crops, local=True, rand_size=False):
        return [
            self._get_window_sizes(lengths, local=local, random=rand_size)
            for _ in range(num_crops)
        ]

    @staticmethod
    def _shared_padded_length(size_list):
        if not size_list:
            return None
        return int(max(sizes.max().item() for sizes in size_list))

    def _padded_length(self, size_list, fixed_padded_length=None):
        shared = self._shared_padded_length(size_list)
        if fixed_padded_length is None:
            return shared
        fixed = int(fixed_padded_length)
        if fixed < 1:
            raise ValueError("Fixed crop padded lengths must be positive.")
        if shared is not None and shared > fixed:
            raise ValueError(
                f"Fixed crop padded length {fixed} is smaller than sampled crop "
                f"length {shared}. Increase fixed_crop_padding_max_peaks or crop scale."
            )
        return fixed

    def __call__(self, spectra, lengths, rand_size=False):
        """
        Perform data augmentation by generating global and local crops.

        Args:
            spectra (torch.Tensor): Input spectra of shape (batch_size, seq_len, embed_dim).
            lengths (torch.Tensor): Lengths of spectra in the batch.
            rand_size (bool): Whether to use random window sizes or fixed maximum window sizes.

        Returns:
            list: List of tuples containing cropped spectra and corresponding padding masks.
        """
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

        global_padded_length = self._padded_length(
            global_sizes_list,
            self.fixed_global_padded_length,
        )
        local_padded_length = self._padded_length(
            local_sizes_list,
            self.fixed_local_padded_length,
        )

        # Keep all global crops at one padded length and all local crops at one
        # padded length so MultiCropWrapper can batch them together.
        for global_sizes in global_sizes_list:
            crops.append(
                self._random_window(
                    spectra,
                    lengths,
                    global_sizes,
                    padded_length=global_padded_length,
                )
            )

        for local_sizes in local_sizes_list:
            crops.append(
                self._random_window(
                    spectra,
                    lengths,
                    local_sizes,
                    padded_length=local_padded_length,
                )
            )

        return crops


class RandomSelectionAugmentation(RandomWindowAugmentation):
    def _sample_selection_indices(self, spectra, lengths, sample_idx, num_select):
        valid_length = lengths[sample_idx].item()
        return torch.randperm(valid_length, device=spectra.device)[:num_select]

    def _sample_selection_indices_fast(
        self,
        spectra,
        sample_idx,
        valid_length,
        num_select,
    ):
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

        global_padded_length = self._padded_length(
            global_sizes_list,
            self.fixed_global_padded_length,
        )
        local_padded_length = self._padded_length(
            local_sizes_list,
            self.fixed_local_padded_length,
        )

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


class BatchedRandomSelectionAugmentation(RandomSelectionAugmentation):
    """Fast distribution-equivalent random subset sampler.

    Unlike RandomSelectionAugmentation, this does not preserve the exact
    randperm-per-sample RNG stream. It assigns one random score per valid peak,
    keeps the k lowest scores for each spectrum, and returns selected peaks in
    their original order. That is still a uniform subset without replacement.
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

        topk_count = min(crop_length, seq_len)
        top_indices_by_score = torch.topk(
            scores,
            k=topk_count,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        topk_positions = torch.arange(topk_count, device=spectra.device)
        rank_mask = topk_positions.unsqueeze(0) < selection_sizes.unsqueeze(1)
        rank_positions = torch.arange(crop_length, device=spectra.device)

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

        padding_mask = rank_positions.unsqueeze(0) >= selection_sizes.unsqueeze(1)
        return selected_spectra, padding_mask


class IntensityWeightedSelectionAugmentation(RandomSelectionAugmentation):
    def __init__(
        self,
        global_crops_scale=(0.5, 0.9),
        local_crops_scale=(0.3, 0.5),
        num_global_crops=2,
        num_local_crops=5,
        padding_value=0,
        intensity_alpha=1.0,
        intensity_eps=0.01,
        fixed_global_padded_length=None,
        fixed_local_padded_length=None,
    ):
        super().__init__(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            num_global_crops=num_global_crops,
            num_local_crops=num_local_crops,
            padding_value=padding_value,
            fixed_global_padded_length=fixed_global_padded_length,
            fixed_local_padded_length=fixed_local_padded_length,
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

    def _sample_selection_indices_fast(
        self,
        spectra,
        sample_idx,
        valid_length,
        num_select,
    ):
        intensities = spectra[sample_idx, :valid_length, 1].float().clamp(min=0)
        weights = (intensities + self.intensity_eps).pow(self.intensity_alpha)
        if not torch.any(weights > 0):
            weights = torch.ones_like(weights)
        return torch.multinomial(weights, num_samples=num_select, replacement=False)


class BatchedIntensityWeightedSelectionAugmentation(IntensityWeightedSelectionAugmentation):
    """Fast distribution-equivalent intensity-weighted subset sampler.

    This uses the Gumbel-top-k trick to sample a weighted subset without
    replacement for every spectrum in the batch at once. The selected peaks are
    returned in original m/z order, matching the augmentation contract, but the
    RNG stream is not bit-exact to torch.multinomial-per-sample.
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

        weights = spectra[..., 1].float().clamp(min=0)
        weights = (weights + self.intensity_eps).pow(self.intensity_alpha)
        weights = weights.masked_fill(~valid_mask, 0.0)
        fallback_weights = valid_mask.to(weights.dtype)
        weights = torch.where(
            torch.any(weights > 0, dim=1, keepdim=True),
            weights,
            fallback_weights,
        )

        log_weights = torch.full_like(weights, -float("inf"))
        positive_mask = weights > 0
        log_weights[positive_mask] = weights[positive_mask].log()
        exponential_noise = torch.empty_like(weights).exponential_()
        exponential_noise.clamp_min_(torch.finfo(weights.dtype).tiny)
        scores = log_weights - exponential_noise.log()

        topk_count = min(crop_length, seq_len)
        top_indices_by_score = torch.topk(
            scores,
            k=topk_count,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        topk_positions = torch.arange(topk_count, device=spectra.device)
        rank_mask = topk_positions.unsqueeze(0) < selection_sizes.unsqueeze(1)
        rank_positions = torch.arange(crop_length, device=spectra.device)

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

        padding_mask = rank_positions.unsqueeze(0) >= selection_sizes.unsqueeze(1)
        return selected_spectra, padding_mask


class StudentDistractorMixAugmentation:
    def __init__(
        self,
        mix_apply_to="all",
        intensity_alpha=1.0,
        intensity_eps=0.01,
        merge_tol=0.001,
        padding_value=0,
    ):
        valid_modes = {"all", "local_only"}
        if mix_apply_to not in valid_modes:
            raise ValueError(
                f"mix_apply_to={mix_apply_to!r} not in supported options: {sorted(valid_modes)}"
            )
        self.mix_apply_to = mix_apply_to
        self.intensity_alpha = intensity_alpha
        self.intensity_eps = intensity_eps
        self.merge_tol = merge_tol
        self.padding_value = padding_value
        self._warned_bs1_noop = False

    @staticmethod
    def _group_padded_length(peaks_by_view):
        if not peaks_by_view:
            return 0
        return max(
            peaks.shape[0]
            for per_sample_peaks in peaks_by_view
            for peaks in per_sample_peaks
        )

    @staticmethod
    def _pad_peaks(peaks_by_sample, padded_length, padding_value=0):
        batch_size = len(peaks_by_sample)
        if batch_size == 0:
            raise ValueError("Cannot pad empty peaks_by_sample list")

        embed_dim = peaks_by_sample[0].shape[-1]
        device = peaks_by_sample[0].device
        dtype = peaks_by_sample[0].dtype
        padded = torch.full(
            (batch_size, padded_length, embed_dim),
            padding_value,
            dtype=dtype,
            device=device,
        )
        padding_mask = torch.ones(
            (batch_size, padded_length),
            dtype=torch.bool,
            device=device,
        )

        for sample_idx, peaks in enumerate(peaks_by_sample):
            n_peaks = peaks.shape[0]
            if n_peaks > 0:
                padded[sample_idx, :n_peaks] = peaks
                padding_mask[sample_idx, :n_peaks] = False

        return padded, padding_mask

    def _sample_distractor_indices(self, spectra, lengths, sample_idx, num_select):
        valid_length = lengths[sample_idx].item()
        intensities = spectra[sample_idx, :valid_length, 1].float().clamp(min=0)
        weights = (intensities + self.intensity_eps).pow(self.intensity_alpha)
        if not torch.any(weights > 0):
            weights = torch.ones_like(weights)
        return torch.multinomial(weights, num_samples=num_select, replacement=False)

    @staticmethod
    def _choose_distractor_sample(batch_size, anchor_idx, device):
        if batch_size <= 1:
            return None
        distractor_idx = torch.randint(batch_size - 1, (1,), device=device).item()
        if distractor_idx >= anchor_idx:
            distractor_idx += 1
        return distractor_idx

    def _merge_close_peaks(self, peaks):
        if peaks.shape[0] <= 1:
            return peaks

        peaks = peaks[torch.argsort(peaks[:, 0], stable=True)]
        merged = [peaks[0].clone()]

        for peak in peaks[1:]:
            current = merged[-1]
            if (peak[0] - current[0]).item() < self.merge_tol:
                total_intensity = current[1] + peak[1]
                if total_intensity.item() > 0:
                    current[0] = (
                        current[0] * current[1] + peak[0] * peak[1]
                    ) / total_intensity
                else:
                    current[0] = 0.5 * (current[0] + peak[0])
                current[1] = total_intensity
            else:
                merged.append(peak.clone())

        return torch.stack(merged, dim=0)

    def _mix_one_sample(self, anchor_peaks, spectra, lengths, anchor_idx, strength):
        if strength <= 0:
            return anchor_peaks

        distractor_idx = self._choose_distractor_sample(
            spectra.shape[0], anchor_idx, spectra.device
        )
        if distractor_idx is None:
            return anchor_peaks

        distractor_length = lengths[distractor_idx].item()
        if distractor_length < 1:
            return anchor_peaks

        num_select = min(
            distractor_length,
            max(1, int(math.ceil(strength * distractor_length))),
        )
        distractor_indices = self._sample_distractor_indices(
            spectra,
            lengths,
            distractor_idx,
            num_select,
        )
        distractor_peaks = spectra[distractor_idx, :distractor_length][
            distractor_indices
        ]

        mixed = torch.cat([anchor_peaks, distractor_peaks], dim=0)
        return self._merge_close_peaks(mixed)

    def __call__(self, crops, spectra, lengths, strength, num_global_crops):
        if strength <= 0:
            return crops

        lengths = lengths.squeeze(-1)
        batch_size = spectra.shape[0]
        if batch_size <= 1:
            if not self._warned_bs1_noop:
                warnings.warn(
                    "StudentDistractorMixAugmentation is a no-op for batch_size=1.",
                    stacklevel=2,
                )
                self._warned_bs1_noop = True
            return crops

        mixed_crops = list(crops)
        if self.mix_apply_to == "all":
            target_indices = list(range(len(crops)))
        else:
            target_indices = list(range(num_global_crops, len(crops)))

        global_indices = [idx for idx in target_indices if idx < num_global_crops]
        local_indices = [idx for idx in target_indices if idx >= num_global_crops]

        for group_indices in (global_indices, local_indices):
            if not group_indices:
                continue

            mixed_peaks_by_view = []
            for crop_idx in group_indices:
                crop, pad_mask = crops[crop_idx]
                mixed_peaks_by_sample = []
                for sample_idx in range(batch_size):
                    anchor_peaks = crop[sample_idx][~pad_mask[sample_idx]]
                    mixed_peaks = self._mix_one_sample(
                        anchor_peaks,
                        spectra,
                        lengths,
                        sample_idx,
                        strength,
                    )
                    mixed_peaks_by_sample.append(mixed_peaks)
                mixed_peaks_by_view.append(mixed_peaks_by_sample)

            padded_length = self._group_padded_length(mixed_peaks_by_view)
            for crop_idx, mixed_peaks_by_sample in zip(
                group_indices, mixed_peaks_by_view
            ):
                mixed_crops[crop_idx] = self._pad_peaks(
                    mixed_peaks_by_sample,
                    padded_length=padded_length,
                    padding_value=self.padding_value,
                )

        return mixed_crops


if __name__ == "__main__":
    SEED = 4

    def print_valid_peaks(title, spectra, lengths):
        print(f"\n{title}")
        for i, length in enumerate(lengths.tolist()):
            print(f"sample {i} valid peaks:")
            print(spectra[i, :length])

    def print_crops(title, crops, num_global_crops):
        print(f"\n{title}")
        for crop_idx, (crop, pad_mask) in enumerate(crops):
            crop_kind = "global" if crop_idx < num_global_crops else "local"
            print(f"\n{crop_kind} crop {crop_idx}:")
            for sample_idx in range(crop.shape[0]):
                valid = crop[sample_idx][~pad_mask[sample_idx]]
                print(
                    f"sample {sample_idx} pad_mask={pad_mask[sample_idx].int().tolist()}"
                )
                print(valid)

    def print_crop_shapes(title, crops, num_global_crops):
        print(f"\n{title}")
        for crop_idx, (crop, _) in enumerate(crops):
            crop_kind = "global" if crop_idx < num_global_crops else "local"
            print(f"{crop_kind} crop {crop_idx} shape={tuple(crop.shape)}")

    def print_sample_views(title, crops, sample_idx, num_global_crops):
        print(f"\n{title}")
        for crop_idx, (crop, pad_mask) in enumerate(crops):
            crop_kind = "global" if crop_idx < num_global_crops else "local"
            valid = crop[sample_idx][~pad_mask[sample_idx]]
            print(f"\n{crop_kind} crop {crop_idx}, sample {sample_idx}:")
            print(valid)

    # Hand-crafted toy spectra:
    # - first column is a peak id, so contiguity/order is visually obvious
    # - second column is a sample-specific marker
    spectra = torch.tensor(
        [
            [
                [10, 100],
                [11, 100],
                [12, 100],
                [13, 1000],
                [14, 100],
                [15, 100],
            ],
            [
                [20, 200],
                [21, 200],
                [22, 2000],
                [23, 200],
                [0, 0],
                [0, 0],
            ],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([6, 4], dtype=torch.long)

    print_valid_peaks("Original spectra", spectra, lengths)

    # Fixed sizes make the behavior easy to check by eye:
    # - global keeps 3 / 2 peaks
    # - local keeps 2 / 1 peaks
    window_aug = RandomWindowAugmentation(
        global_crops_scale=(0.5, 0.5),
        local_crops_scale=(0.34, 0.34),
        num_global_crops=1,
        num_local_crops=1,
    )
    selection_aug = RandomSelectionAugmentation(
        global_crops_scale=(0.5, 0.5),
        local_crops_scale=(0.34, 0.34),
        num_global_crops=1,
        num_local_crops=1,
    )
    intensity_weighted_aug = IntensityWeightedSelectionAugmentation(
        global_crops_scale=(0.4, 0.9),
        local_crops_scale=(0.2, 0.6),
        num_global_crops=1,
        num_local_crops=1,
        intensity_alpha=2.0,
        intensity_eps=0.01,
    )
    local_only_intensity_weighted_aug = IntensityWeightedSelectionAugmentation(
        global_crops_scale=(0.4, 0.9),
        local_crops_scale=(0.2, 0.6),
        num_global_crops=0,
        num_local_crops=4,
        intensity_alpha=2.0,
        intensity_eps=0.01,
    )
    distractor_mix_aug = StudentDistractorMixAugmentation(
        mix_apply_to="all",
        intensity_alpha=2.0,
        intensity_eps=0.01,
        merge_tol=0.001,
    )

    # torch.manual_seed(SEED)
    window_crops = window_aug(spectra, lengths)
    # torch.manual_seed(SEED)
    selection_crops = selection_aug(spectra, lengths)
    # torch.manual_seed(SEED)
    intensity_weighted_crops = intensity_weighted_aug(spectra, lengths)
    # torch.manual_seed(SEED)
    local_only_intensity_weighted_crops = local_only_intensity_weighted_aug(
        spectra, lengths
    )
    torch.manual_seed(SEED)
    mixed_local_views_weak = distractor_mix_aug(
        local_only_intensity_weighted_crops,
        spectra=spectra,
        lengths=lengths,
        strength=0.5,
        num_global_crops=local_only_intensity_weighted_aug.num_global_crops,
    )
    torch.manual_seed(SEED)
    mixed_local_views_full = distractor_mix_aug(
        local_only_intensity_weighted_crops,
        spectra=spectra,
        lengths=lengths,
        strength=1.0,
        num_global_crops=local_only_intensity_weighted_aug.num_global_crops,
    )

    # print_crops(
    #     "RandomWindowAugmentation: valid peaks should stay contiguous in original order",
    #     window_crops,
    #     window_aug.num_global_crops,
    # )
    # print_crops(
    #     "RandomSelectionAugmentation: valid peaks should be arbitrary subsets",
    #     selection_crops,
    #     selection_aug.num_global_crops,
    # )
    print_crops(
        "IntensityWeightedSelectionAugmentation: valid peaks should be biased toward larger intensities",
        intensity_weighted_crops,
        intensity_weighted_aug.num_global_crops,
    )
    print_crop_shapes(
        "IntensityWeightedSelectionAugmentation with rand_size=True: all globals should share one padded length and all locals another",
        intensity_weighted_crops,
        intensity_weighted_aug.num_global_crops,
    )
    print_sample_views(
        "IntensityWeightedSelectionAugmentation: multiple local-only clean views of sample 0",
        local_only_intensity_weighted_crops,
        sample_idx=0,
        num_global_crops=local_only_intensity_weighted_aug.num_global_crops,
    )
    print_sample_views(
        "StudentDistractorMixAugmentation: same local-only views of sample 0 with strength=0.5",
        mixed_local_views_weak,
        sample_idx=0,
        num_global_crops=local_only_intensity_weighted_aug.num_global_crops,
    )
    print_sample_views(
        "StudentDistractorMixAugmentation: same local-only views of sample 0 with strength=1.0",
        mixed_local_views_full,
        sample_idx=0,
        num_global_crops=local_only_intensity_weighted_aug.num_global_crops,
    )
    print_crop_shapes(
        "StudentDistractorMixAugmentation: mixed local views should still share one padded shape",
        mixed_local_views_full,
        num_global_crops=local_only_intensity_weighted_aug.num_global_crops,
    )
