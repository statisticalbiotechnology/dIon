"""Batched precursor-conditioned distractor mixing for DINO student views."""

from __future__ import annotations

import warnings

import torch


class BatchedStudentDistractorMixAugmentation:
    """Mix random in-batch distractor peaks into selected student crop groups.

    Distractor identities and peak subsets are sampled independently per view.
    Provenance codes are 0 for pure anchor, 1 for pure distractor, 2 for a
    cross-component merge, and -1 for padding.
    """

    PROTON_MASS_DA = 1.007276466621
    ANCHOR = 0
    DISTRACTOR = 1
    MIXED = 2
    PADDING = -1

    def __init__(
        self,
        mix_apply_to="global_only",
        distractor_sampling="per_view",
        condition_separation_ppm=10.0,
        neutral_mass_separation_ppm=10.0,
        merge_ppm=5.0,
        intensity_normalization="none",
        padding_value=0,
    ):
        valid_modes = {"all", "global_only", "local_only"}
        if mix_apply_to not in valid_modes:
            raise ValueError(
                f"mix_apply_to={mix_apply_to!r} not in supported options: "
                f"{sorted(valid_modes)}"
            )
        if distractor_sampling != "per_view":
            raise ValueError("Only distractor_sampling='per_view' is supported.")
        valid_normalization = {"none", "final_minmax", "final_basepeak"}
        if intensity_normalization not in valid_normalization:
            raise ValueError(
                "intensity_normalization must be one of: "
                f"{sorted(valid_normalization)}"
            )
        if condition_separation_ppm < 0 or neutral_mass_separation_ppm < 0:
            raise ValueError("Precursor separation tolerances must be non-negative.")
        if merge_ppm < 0:
            raise ValueError("merge_ppm must be non-negative.")

        self.mix_apply_to = mix_apply_to
        self.distractor_sampling = distractor_sampling
        self.condition_separation_ppm = float(condition_separation_ppm)
        self.neutral_mass_separation_ppm = float(neutral_mass_separation_ppm)
        self.merge_ppm = float(merge_ppm)
        self.intensity_normalization = intensity_normalization
        self.padding_value = padding_value
        self._warned_bs1_noop = False
        self._warned_no_eligible = False

    @staticmethod
    def _gather_rows(values, row_indices):
        gather_indices = row_indices.reshape(
            (values.shape[0],) + (1,) * (values.ndim - 1)
        ).expand(-1, *values.shape[1:])
        return torch.gather(values, 0, gather_indices)

    def eligible_distractors(self, precursor_mz, precursor_charge):
        """Return the directed anchor-by-candidate eligibility matrix."""
        precursor_mz = precursor_mz.reshape(-1).float()
        precursor_charge = precursor_charge.reshape(-1).float()
        batch_size = precursor_mz.numel()
        device = precursor_mz.device

        nonself = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        same_charge = precursor_charge[:, None] == precursor_charge[None, :]
        mz_delta = (precursor_mz[:, None] - precursor_mz[None, :]).abs()
        mz_midpoint = 0.5 * (precursor_mz[:, None] + precursor_mz[None, :])
        condition_collision = same_charge & (
            mz_delta < self.condition_separation_ppm * 1e-6 * mz_midpoint
        )

        neutral_mass = precursor_charge * (
            precursor_mz - self.PROTON_MASS_DA
        )
        neutral_delta = (neutral_mass[:, None] - neutral_mass[None, :]).abs()
        neutral_midpoint = 0.5 * (
            neutral_mass[:, None] + neutral_mass[None, :]
        )
        neutral_collision = neutral_delta < (
            self.neutral_mass_separation_ppm * 1e-6 * neutral_midpoint
        )
        return nonself & ~(condition_collision | neutral_collision)

    @staticmethod
    def _sample_distractors(eligible):
        has_eligible = eligible.any(dim=1)
        scores = torch.rand(eligible.shape, device=eligible.device)
        scores.masked_fill_(~eligible, float("inf"))
        selected = scores.argmin(dim=1)
        anchors = torch.arange(eligible.shape[0], device=eligible.device)
        return torch.where(has_eligible, selected, anchors), has_eligible

    def _random_subset(self, spectra, lengths, selection_sizes):
        batch_size, seq_len, embed_dim = spectra.shape
        lengths = lengths.to(device=spectra.device, dtype=torch.long)
        selection_sizes = torch.minimum(
            selection_sizes.to(device=spectra.device, dtype=torch.long), lengths
        ).clamp(min=0)
        crop_length = int(selection_sizes.max().item())
        if crop_length == 0:
            return (
                spectra.new_full(
                    (batch_size, 0, embed_dim), self.padding_value
                ),
                torch.ones(
                    (batch_size, 0), dtype=torch.bool, device=spectra.device
                ),
            )

        positions = torch.arange(seq_len, device=spectra.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        scores = torch.rand((batch_size, seq_len), device=spectra.device)
        scores.masked_fill_(~valid, float("inf"))
        selected_by_score = torch.topk(
            scores,
            k=min(crop_length, seq_len),
            dim=1,
            largest=False,
            sorted=False,
        ).indices
        keep_rank = (
            torch.arange(selected_by_score.shape[1], device=spectra.device)
            .unsqueeze(0)
            .lt(selection_sizes.unsqueeze(1))
        )
        selection_mask = torch.zeros_like(valid)
        selection_mask.scatter_(1, selected_by_score, keep_rank)

        selected = spectra.new_full(
            (batch_size, crop_length, embed_dim), self.padding_value
        )
        total_selected = int(selection_sizes.sum().item())
        if total_selected:
            selected_values = spectra[selection_mask]
            row_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=spectra.device), selection_sizes
            )
            starts = torch.cumsum(selection_sizes, dim=0) - selection_sizes
            flat_positions = torch.arange(total_selected, device=spectra.device)
            col_ids = flat_positions - torch.repeat_interleave(
                starts, selection_sizes
            )
            selected[row_ids, col_ids] = selected_values

        padding_mask = (
            torch.arange(crop_length, device=spectra.device).unsqueeze(0)
            >= selection_sizes.unsqueeze(1)
        )
        return selected, padding_mask

    def merge_batched(self, peaks, padding_mask, provenance):
        """Single-linkage merge adjacent m/z peaks within the ppm threshold."""
        batch_size, seq_len, _ = peaks.shape
        if seq_len <= 1 or self.merge_ppm == 0:
            return peaks, padding_mask, provenance

        valid = ~padding_mask
        sortable_mz = peaks[..., 0].masked_fill(~valid, float("inf"))
        order = torch.argsort(sortable_mz, dim=1, stable=True)
        sorted_peaks = torch.gather(
            peaks, 1, order.unsqueeze(-1).expand(-1, -1, peaks.shape[-1])
        )
        sorted_valid = torch.gather(valid, 1, order)
        sorted_provenance = torch.gather(provenance, 1, order)

        mz = sorted_peaks[..., 0]
        intensity = sorted_peaks[..., 1]
        midpoint = 0.5 * (mz[:, 1:] + mz[:, :-1])
        close = (
            sorted_valid[:, 1:]
            & sorted_valid[:, :-1]
            & ((mz[:, 1:] - mz[:, :-1]) < self.merge_ppm * 1e-6 * midpoint)
        )
        starts = torch.zeros_like(sorted_valid)
        starts[:, 0] = sorted_valid[:, 0]
        starts[:, 1:] = sorted_valid[:, 1:] & ~close
        group_ids = starts.cumsum(dim=1) - 1
        safe_group_ids = group_ids.clamp(min=0)
        group_count = starts.sum(dim=1).long()

        output_mz_weighted = peaks.new_zeros((batch_size, seq_len))
        output_intensity = peaks.new_zeros((batch_size, seq_len))
        output_mz_sum = peaks.new_zeros((batch_size, seq_len))
        output_count = peaks.new_zeros((batch_size, seq_len))
        valid_float = sorted_valid.to(peaks.dtype)
        valid_intensity = intensity * valid_float
        output_intensity.scatter_add_(1, safe_group_ids, valid_intensity)
        output_mz_weighted.scatter_add_(
            1,
            safe_group_ids,
            mz.masked_fill(~sorted_valid, 0) * valid_intensity,
        )
        output_mz_sum.scatter_add_(
            1, safe_group_ids, mz.masked_fill(~sorted_valid, 0)
        )
        output_count.scatter_add_(1, safe_group_ids, valid_float)
        output_mz = torch.where(
            output_intensity > 0,
            output_mz_weighted
            / output_intensity.clamp_min(torch.finfo(peaks.dtype).tiny),
            output_mz_sum / output_count.clamp_min(1),
        )

        anchor_presence = torch.zeros(
            (batch_size, seq_len), dtype=torch.int32, device=peaks.device
        )
        distractor_presence = torch.zeros_like(anchor_presence)
        anchor_presence.scatter_add_(
            1,
            safe_group_ids,
            ((sorted_provenance == self.ANCHOR) & sorted_valid).int(),
        )
        distractor_presence.scatter_add_(
            1,
            safe_group_ids,
            ((sorted_provenance == self.DISTRACTOR) & sorted_valid).int(),
        )
        output_provenance = torch.full(
            (batch_size, seq_len),
            self.PADDING,
            dtype=torch.int8,
            device=peaks.device,
        )
        output_provenance[anchor_presence > 0] = self.ANCHOR
        output_provenance[distractor_presence > 0] = self.DISTRACTOR
        output_provenance[(anchor_presence > 0) & (distractor_presence > 0)] = (
            self.MIXED
        )

        output = torch.stack([output_mz, output_intensity], dim=-1)
        output_padding = (
            torch.arange(seq_len, device=peaks.device).unsqueeze(0)
            >= group_count.unsqueeze(1)
        )
        output.masked_fill_(output_padding.unsqueeze(-1), self.padding_value)
        return output, output_padding, output_provenance

    def _normalize_intensity(self, peaks, padding_mask):
        if self.intensity_normalization == "none":
            return peaks

        valid = ~padding_mask
        intensity = peaks[..., 1]
        if self.intensity_normalization == "final_basepeak":
            maximum = intensity.masked_fill(~valid, 0).amax(dim=1, keepdim=True)
            normalized = intensity / maximum.clamp_min(
                torch.finfo(intensity.dtype).tiny
            )
        else:
            minimum = intensity.masked_fill(~valid, float("inf")).amin(
                dim=1, keepdim=True
            )
            maximum = intensity.masked_fill(~valid, -float("inf")).amax(
                dim=1, keepdim=True
            )
            span = maximum - minimum
            normalized = torch.where(
                span > 0,
                (intensity - minimum) / span,
                torch.zeros_like(intensity),
            )
        normalized.masked_fill_(~valid, self.padding_value)
        output = peaks.clone()
        output[..., 1] = normalized
        return output

    def _mix_view(
        self,
        crop,
        spectra,
        lengths,
        distractor_indices,
        has_eligible,
        strength,
    ):
        anchor_peaks, anchor_padding = crop
        anchor_counts = (~anchor_padding).sum(dim=1).long()
        anchor_provenance = torch.full(
            anchor_padding.shape,
            self.ANCHOR,
            dtype=torch.int8,
            device=anchor_peaks.device,
        )
        anchor_provenance.masked_fill_(anchor_padding, self.PADDING)
        anchor_peaks, anchor_padding, anchor_provenance = self.merge_batched(
            anchor_peaks, anchor_padding, anchor_provenance
        )

        distractor_spectra = self._gather_rows(spectra, distractor_indices)
        distractor_lengths = lengths[distractor_indices]
        distractor_counts = torch.floor(
            anchor_counts.float() * float(strength)
        ).long()
        distractor_counts = torch.minimum(distractor_counts, distractor_lengths)
        distractor_counts = torch.where(
            has_eligible & (strength > 0),
            distractor_counts.clamp(min=1),
            torch.zeros_like(distractor_counts),
        )
        distractor_peaks, distractor_padding = self._random_subset(
            distractor_spectra, distractor_lengths, distractor_counts
        )
        distractor_provenance = torch.full(
            distractor_padding.shape,
            self.DISTRACTOR,
            dtype=torch.int8,
            device=anchor_peaks.device,
        )
        distractor_provenance.masked_fill_(distractor_padding, self.PADDING)
        distractor_peaks, distractor_padding, distractor_provenance = (
            self.merge_batched(
                distractor_peaks,
                distractor_padding,
                distractor_provenance,
            )
        )

        mixed_peaks = torch.cat([anchor_peaks, distractor_peaks], dim=1)
        mixed_padding = torch.cat([anchor_padding, distractor_padding], dim=1)
        mixed_provenance = torch.cat(
            [anchor_provenance, distractor_provenance], dim=1
        )
        mixed_peaks, mixed_padding, mixed_provenance = self.merge_batched(
            mixed_peaks, mixed_padding, mixed_provenance
        )
        mixed_peaks = self._normalize_intensity(mixed_peaks, mixed_padding)
        return mixed_peaks, mixed_padding, mixed_provenance

    def mix_fixed_distractors(
        self,
        anchor_peaks,
        anchor_padding,
        distractor_peaks,
        distractor_padding,
        *,
        strength,
        return_provenance=False,
    ):
        """Mix paired anchor/distractor rows with the training merge policy.

        This is the deterministic counterpart to :meth:`__call__` for
        evaluation: row ``i`` of ``distractor_peaks`` is the selected
        distractor for row ``i`` of ``anchor_peaks``. It preserves the exact
        per-anchor distractor count and pre/post-mixing merge behavior used by
        DINO student-view mixing.
        """
        if anchor_peaks.ndim != 3 or distractor_peaks.ndim != 3:
            raise ValueError("Expected [batch, peaks, features] peak tensors.")
        if anchor_peaks.shape[0] != distractor_peaks.shape[0]:
            raise ValueError("Anchor and distractor batches must have equal size.")
        if anchor_peaks.shape[-1] != distractor_peaks.shape[-1]:
            raise ValueError("Anchor and distractor feature dimensions must match.")
        if anchor_padding.shape != anchor_peaks.shape[:2]:
            raise ValueError("Anchor padding shape must match anchor peaks.")
        if distractor_padding.shape != distractor_peaks.shape[:2]:
            raise ValueError("Distractor padding shape must match distractor peaks.")

        max_length = max(anchor_peaks.shape[1], distractor_peaks.shape[1])

        def pad_to_length(peaks, padding):
            if peaks.shape[1] == max_length:
                return peaks, padding
            extra = max_length - peaks.shape[1]
            return (
                torch.nn.functional.pad(
                    peaks, (0, 0, 0, extra), value=self.padding_value
                ),
                torch.nn.functional.pad(padding, (0, extra), value=True),
            )

        anchor_peaks, anchor_padding = pad_to_length(anchor_peaks, anchor_padding)
        distractor_peaks, distractor_padding = pad_to_length(
            distractor_peaks, distractor_padding
        )
        batch_size = anchor_peaks.shape[0]
        distractor_lengths = (~distractor_padding).sum(dim=1).long()
        mixed = self._mix_view(
            (anchor_peaks, anchor_padding),
            distractor_peaks,
            distractor_lengths,
            torch.arange(batch_size, device=anchor_peaks.device),
            torch.ones(batch_size, dtype=torch.bool, device=anchor_peaks.device),
            float(strength),
        )
        if return_provenance:
            return mixed
        return mixed[:2]

    def mix_sampled_distractors(
        self,
        anchor_peaks,
        anchor_padding,
        spectra,
        lengths,
        precursor_mz,
        precursor_charge,
        *,
        strength,
        return_provenance=False,
    ):
        """Mix one sampled in-batch distractor into each anchor crop.

        This exposes the same sampling and merge path used by :meth:`__call__
        for a single DINO student crop.  Downstream target-swapped training
        uses the returned partner indices to construct both `(M, A) -> A` and
        `(M, B) -> B` decoder targets from the identical mixed input `M`.
        """
        if anchor_peaks.ndim != 3 or spectra.ndim != 3:
            raise ValueError("Expected [batch, peaks, features] peak tensors.")
        if anchor_peaks.shape[0] != spectra.shape[0]:
            raise ValueError("Anchor crops and source spectra must share a batch size.")
        if anchor_padding.shape != anchor_peaks.shape[:2]:
            raise ValueError("Anchor padding shape must match anchor peaks.")
        if float(strength) <= 0:
            raise ValueError("Target-swapped distractor mixing requires positive strength.")

        lengths = lengths.reshape(-1).to(device=spectra.device, dtype=torch.long)
        eligible = self.eligible_distractors(precursor_mz, precursor_charge)
        distractor_indices, has_eligible = self._sample_distractors(eligible)
        mixed = self._mix_view(
            (anchor_peaks, anchor_padding),
            spectra,
            lengths,
            distractor_indices,
            has_eligible,
            strength,
        )
        if return_provenance:
            return (*mixed, distractor_indices, has_eligible)
        return mixed[:2], distractor_indices, has_eligible

    def _target_indices(self, num_crops, num_global_crops):
        if self.mix_apply_to == "all":
            return list(range(num_crops))
        if self.mix_apply_to == "global_only":
            return list(range(num_global_crops))
        return list(range(num_global_crops, num_crops))

    def _anchor_provenance(self, crops):
        return [
            torch.where(
                padding,
                torch.full_like(padding, self.PADDING, dtype=torch.int8),
                torch.full_like(padding, self.ANCHOR, dtype=torch.int8),
            )
            for _, padding in crops
        ]

    def __call__(
        self,
        crops,
        spectra,
        lengths,
        precursor_mz,
        precursor_charge,
        strength,
        num_global_crops,
        return_provenance=False,
    ):
        provenance = self._anchor_provenance(crops)
        if strength <= 0:
            return (crops, provenance) if return_provenance else crops

        lengths = lengths.squeeze(-1).long()
        batch_size = spectra.shape[0]
        if batch_size <= 1:
            if not self._warned_bs1_noop:
                warnings.warn(
                    "Distractor mixing is a no-op for batch_size=1.", stacklevel=2
                )
                self._warned_bs1_noop = True
            return (crops, provenance) if return_provenance else crops

        mixed_crops = list(crops)
        target_indices = self._target_indices(len(crops), num_global_crops)
        eligible = self.eligible_distractors(precursor_mz, precursor_charge)
        target_groups = (
            [idx for idx in target_indices if idx < num_global_crops],
            [idx for idx in target_indices if idx >= num_global_crops],
        )
        for group_indices in target_groups:
            if not group_indices:
                continue
            group_results = []
            for crop_idx in group_indices:
                distractor_indices, has_eligible = self._sample_distractors(eligible)
                if not bool(has_eligible.all()) and not self._warned_no_eligible:
                    warnings.warn(
                        "Some anchors have no eligible in-batch distractor; "
                        "those student views remain anchor-only.",
                        stacklevel=2,
                    )
                    self._warned_no_eligible = True
                group_results.append(
                    self._mix_view(
                        crops[crop_idx],
                        spectra,
                        lengths,
                        distractor_indices,
                        has_eligible,
                        strength,
                    )
                )

            shared_length = max(
                int((~result[1]).sum(dim=1).max().item())
                for result in group_results
            )
            for crop_idx, (peaks, padding, origins) in zip(
                group_indices, group_results
            ):
                mixed_crops[crop_idx] = (
                    peaks[:, :shared_length],
                    padding[:, :shared_length],
                )
                provenance[crop_idx] = origins[:, :shared_length]

        if return_provenance:
            return mixed_crops, provenance
        return mixed_crops
