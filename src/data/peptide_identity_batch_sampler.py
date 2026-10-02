"""Precursor-matched positive-aware batch sampling for SupCon metric learning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lance
import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class _PeptideVariant:
    """One exact-peptide, precursor-charge sampling group."""

    label: str
    charge: int
    precursor_mz: float
    indices: np.ndarray


class PeptideIdentityBatchSampler(Sampler[list[int]]):
    """Build SupCon batches from static precursor-matched peptide neighborhoods.

    A batch contains ``P`` distinct exact peptide identities and ``K`` spectra
    from one precursor-charge variant of each identity.  Selection begins with
    different-peptide, same-charge variants within ``strict_ppm`` of an anchor,
    widens through ``relaxed_ppm_windows`` while retaining charge, then falls
    back to same-charge/global variants only when needed.  It is entirely
    metadata-based and independent of encoder embeddings.
    """

    def __init__(
        self,
        lance_path: str | Path,
        *,
        label_column: str,
        peptides_per_batch: int,
        spectra_per_peptide: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        batches_per_epoch: int | None = None,
        precursor_mz_column: str = "precursor_mz",
        precursor_charge_column: str = "precursor_charge",
        strict_ppm: float = 10.0,
        relaxed_ppm_windows: tuple[float, ...] | list[float] = (50.0, 250.0, 1000.0, 5000.0, 20_000.0),
        diagnostics_reservoir_size: int = 100_000,
    ) -> None:
        super().__init__()
        if peptides_per_batch < 2:
            raise ValueError("peptides_per_batch must be at least two for contrastive negatives.")
        if spectra_per_peptide < 2:
            raise ValueError("spectra_per_peptide must be at least two for SupCon positives.")
        if rank < 0 or rank >= world_size:
            raise ValueError("Invalid distributed rank/world size.")
        if strict_ppm <= 0:
            raise ValueError("strict_ppm must be positive.")

        windows = tuple(float(window) for window in relaxed_ppm_windows)
        if any(window <= strict_ppm for window in windows) or list(windows) != sorted(windows):
            raise ValueError("relaxed_ppm_windows must be sorted and strictly greater than strict_ppm.")

        self.lance_path = str(lance_path)
        self.label_column = str(label_column)
        self.peptides_per_batch = int(peptides_per_batch)
        self.spectra_per_peptide = int(spectra_per_peptide)
        self.batch_size = self.peptides_per_batch * self.spectra_per_peptide
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.strict_ppm = float(strict_ppm)
        self.relaxed_ppm_windows = windows
        self.diagnostics_reservoir_size = int(diagnostics_reservoir_size)
        self.epoch = 0

        groups: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
        dataset = lance.dataset(self.lance_path)
        row_index = 0
        columns = [self.label_column, precursor_mz_column, precursor_charge_column]
        for batch in dataset.scanner(columns=columns, batch_size=65_536).to_batches():
            labels = batch.column(self.label_column).to_pylist()
            mzs = batch.column(precursor_mz_column).to_pylist()
            charges = batch.column(precursor_charge_column).to_pylist()
            for offset, (label, mz, charge) in enumerate(zip(labels, mzs, charges)):
                if not isinstance(label, str) or not label:
                    continue
                try:
                    mz = float(mz)
                    charge = int(charge)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(mz) or mz <= 0 or charge <= 0:
                    continue
                groups[(label, charge)].append((row_index + offset, mz))
            row_index += batch.num_rows

        self.variants = []
        for (label, charge), values in groups.items():
            if len(values) < self.spectra_per_peptide:
                continue
            indices, mzs = zip(*values)
            self.variants.append(
                _PeptideVariant(
                    label=label,
                    charge=charge,
                    precursor_mz=float(np.median(np.asarray(mzs, dtype=np.float64))),
                    indices=np.asarray(indices, dtype=np.int64),
                )
            )
        if len(self.variants) < self.peptides_per_batch:
            raise ValueError(
                f"{self.lance_path} has only {len(self.variants)} peptide-charge variants "
                f"with at least {self.spectra_per_peptide} spectra; need {self.peptides_per_batch}."
            )

        self._all_variant_ids = np.arange(len(self.variants), dtype=np.int64)
        self._by_charge: dict[int, np.ndarray] = {}
        self._charge_mzs: dict[int, np.ndarray] = {}
        self._label_to_variants: dict[str, np.ndarray] = {}
        by_charge: dict[int, list[int]] = defaultdict(list)
        by_label: dict[str, list[int]] = defaultdict(list)
        for variant_id, variant in enumerate(self.variants):
            by_charge[variant.charge].append(variant_id)
            by_label[variant.label].append(variant_id)
        for charge, variant_ids in by_charge.items():
            sorted_ids = np.asarray(
                sorted(variant_ids, key=lambda value: self.variants[value].precursor_mz), dtype=np.int64
            )
            self._by_charge[charge] = sorted_ids
            self._charge_mzs[charge] = np.asarray(
                [self.variants[value].precursor_mz for value in sorted_ids], dtype=np.float64
            )
        self._label_to_variants = {
            label: np.asarray(variant_ids, dtype=np.int64) for label, variant_ids in by_label.items()
        }
        self._strict_anchor_ids = np.asarray(
            [
                variant_id
                for variant_id in self._all_variant_ids
                if self._has_different_peptide_neighbor(int(variant_id), self.strict_ppm)
            ],
            dtype=np.int64,
        )
        self.eligible_peptide_identity_count = len(
            {self.variants[variant_id].label for variant_id in self._strict_anchor_ids.tolist()}
        )

        default_global_batches = max(
            1, sum(len(variant.indices) for variant in self.variants) // self.batch_size
        )
        self.batches_per_epoch = (
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else max(1, default_global_batches // self.world_size)
        )
        if self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive.")
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self._observed_batches = 0
        self._fallback_batches = 0
        self._comparison_count = 0
        self._same_charge_comparisons = 0
        self._strict_comparisons = 0
        self._anchor_count = 0
        self._anchors_with_strict_negative = 0
        self._anchors_with_same_charge_negative = 0
        self._strict_negative_counts: list[int] = []
        self._selected_by_tier = {
            "strict": 0,
            "relaxed_same_charge": 0,
            "same_charge_global": 0,
            "global": 0,
        }
        self._negative_ppm_samples: list[float] = []

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self._reset_diagnostics()

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _neighbor_ids(self, variant_id: int, ppm: float) -> np.ndarray:
        anchor = self.variants[variant_id]
        mzs = self._charge_mzs[anchor.charge]
        ids = self._by_charge[anchor.charge]
        delta = anchor.precursor_mz * ppm * 1e-6
        lower = int(np.searchsorted(mzs, anchor.precursor_mz - delta, side="left"))
        upper = int(np.searchsorted(mzs, anchor.precursor_mz + delta, side="right"))
        return ids[lower:upper]

    def _has_different_peptide_neighbor(self, variant_id: int, ppm: float) -> bool:
        label = self.variants[variant_id].label
        return any(self.variants[int(other)].label != label for other in self._neighbor_ids(variant_id, ppm))

    def _add_candidates(
        self,
        selected: list[int],
        candidate_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        if len(selected) >= self.peptides_per_batch:
            return
        selected_labels = {self.variants[variant_id].label for variant_id in selected}
        for variant_id in rng.permutation(candidate_ids):
            variant_id = int(variant_id)
            label = self.variants[variant_id].label
            if label in selected_labels:
                continue
            selected.append(variant_id)
            selected_labels.add(label)
            if len(selected) == self.peptides_per_batch:
                return

    def _add_strict_neighborhoods(
        self, selected: list[int], rng: np.random.Generator
    ) -> None:
        """Prioritize strict pairs from several local precursor neighborhoods."""
        for anchor_id in rng.permutation(self._strict_anchor_ids):
            # A new strict neighborhood needs room for both its anchor and at
            # least one different-peptide partner.
            if self.peptides_per_batch - len(selected) < 2:
                return
            anchor_id = int(anchor_id)
            selected_labels = {self.variants[variant_id].label for variant_id in selected}
            if self.variants[anchor_id].label in selected_labels:
                continue
            neighbors = self._neighbor_ids(anchor_id, self.strict_ppm)
            has_available_partner = any(
                self.variants[int(candidate)].label not in selected_labels
                and self.variants[int(candidate)].label != self.variants[anchor_id].label
                for candidate in neighbors
            )
            if not has_available_partner:
                continue
            selected.append(anchor_id)
            self._add_candidates(selected, neighbors, rng)

    def _select_variants(self, rng: np.random.Generator) -> tuple[list[int], dict[str, int]]:
        anchor_pool = self._strict_anchor_ids if len(self._strict_anchor_ids) else self._all_variant_ids
        anchor_id = int(rng.choice(anchor_pool))
        selected = [anchor_id]
        self._add_candidates(selected, self._neighbor_ids(anchor_id, self.strict_ppm), rng)
        # Before relaxing a sparse anchor neighborhood, add strict local pairs
        # from additional anchors. This keeps available 10-ppm negatives from
        # being overwhelmed by broad fallback identities.
        if len(selected) < self.peptides_per_batch and len(self._strict_anchor_ids):
            self._add_strict_neighborhoods(selected, rng)
        strict_count = len(selected)

        before = len(selected)
        for ppm in self.relaxed_ppm_windows:
            if len(selected) == self.peptides_per_batch:
                break
            self._add_candidates(selected, self._neighbor_ids(anchor_id, ppm), rng)
        relaxed_count = len(selected) - before

        before = len(selected)
        if len(selected) < self.peptides_per_batch:
            self._add_candidates(selected, self._by_charge[self.variants[anchor_id].charge], rng)
        same_charge_global_count = len(selected) - before

        before = len(selected)
        if len(selected) < self.peptides_per_batch:
            self._add_candidates(selected, self._all_variant_ids, rng)
        global_count = len(selected) - before

        if len(selected) != self.peptides_per_batch:
            raise RuntimeError("Unable to assemble a full distinct-peptide metric-learning batch.")
        labels = [self.variants[variant_id].label for variant_id in selected]
        if len(set(labels)) != self.peptides_per_batch:
            raise RuntimeError("Metric-learning batch contains duplicate peptide identities.")
        return selected, {
            "strict": strict_count,
            "relaxed_same_charge": relaxed_count,
            "same_charge_global": same_charge_global_count,
            "global": global_count,
        }

    def _record_batch(
        self, selected: list[int], tier_counts: dict[str, int], rng: np.random.Generator
    ) -> None:
        self._observed_batches += 1
        self._fallback_batches += int(
            sum(count for name, count in tier_counts.items() if name != "strict") > 0
        )
        for name, count in tier_counts.items():
            self._selected_by_tier[name] += count
        variants = [self.variants[variant_id] for variant_id in selected]
        mzs = np.asarray([variant.precursor_mz for variant in variants], dtype=np.float64)
        charges = np.asarray([variant.charge for variant in variants], dtype=np.int64)
        upper = np.triu_indices(len(variants), k=1)
        same_charge = charges[:, None] == charges[None, :]
        ppm = np.abs(mzs[:, None] - mzs[None, :]) / ((mzs[:, None] + mzs[None, :]) / 2.0) * 1e6
        pair_same_charge = same_charge[upper]
        pair_ppm = ppm[upper]
        self._comparison_count += int(pair_ppm.size)
        self._same_charge_comparisons += int(pair_same_charge.sum())
        self._strict_comparisons += int((pair_same_charge & (pair_ppm <= self.strict_ppm)).sum())

        diagonal = np.eye(len(variants), dtype=bool)
        strict_counts = (same_charge & (ppm <= self.strict_ppm) & ~diagonal).sum(axis=1)
        same_charge_counts = (same_charge & ~diagonal).sum(axis=1)
        self._anchor_count += len(variants)
        self._anchors_with_strict_negative += int((strict_counts > 0).sum())
        self._anchors_with_same_charge_negative += int((same_charge_counts > 0).sum())
        self._strict_negative_counts.extend(strict_counts.astype(int).tolist())

        if self.diagnostics_reservoir_size > 0 and pair_ppm.size:
            remaining = self.diagnostics_reservoir_size - len(self._negative_ppm_samples)
            if remaining > 0:
                take = min(remaining, pair_ppm.size, 64)
                self._negative_ppm_samples.extend(rng.choice(pair_ppm, size=take, replace=False).tolist())

    def diagnostics(self) -> dict[str, float]:
        ppm = np.asarray(self._negative_ppm_samples, dtype=np.float64)
        comparisons = max(1, self._comparison_count)
        batches = max(1, self._observed_batches)
        anchors = max(1, self._anchor_count)
        selected = max(1, sum(self._selected_by_tier.values()))
        strict_counts = np.asarray(self._strict_negative_counts, dtype=np.float64)
        return {
            "batches_observed": float(self._observed_batches),
            "fallback_batch_fraction": float(self._fallback_batches / batches),
            "distinct_peptide_identities_per_batch": float(self.peptides_per_batch),
            "spectra_per_peptide": float(self.spectra_per_peptide),
            "anchors_with_same_charge_10ppm_negative_fraction": float(
                self._anchors_with_strict_negative / anchors
            ),
            "strict_negative_identities_per_anchor_mean": (
                float(strict_counts.mean()) if strict_counts.size else float("nan")
            ),
            "strict_negative_identities_per_anchor_median": (
                float(np.median(strict_counts)) if strict_counts.size else float("nan")
            ),
            "anchors_with_same_charge_negative_fraction": float(
                self._anchors_with_same_charge_negative / anchors
            ),
            **{
                f"selected_identity_{name}_tier_fraction": float(count / selected)
                for name, count in self._selected_by_tier.items()
            },
            "cross_peptide_same_charge_fraction": float(self._same_charge_comparisons / comparisons),
            "cross_peptide_same_charge_10ppm_fraction": float(self._strict_comparisons / comparisons),
            "cross_peptide_same_charge_wider_than_10ppm_fraction": float(
                (self._same_charge_comparisons - self._strict_comparisons) / comparisons
            ),
            "cross_peptide_cross_charge_global_fraction": float(
                (self._comparison_count - self._same_charge_comparisons) / comparisons
            ),
            "eligible_strict_neighbor_peptide_identities": float(self.eligible_peptide_identity_count),
            "eligible_strict_neighbor_peptide_fraction": float(
                self.eligible_peptide_identity_count / max(1, len(self._label_to_variants))
            ),
            "negative_precursor_ppm_mean": float(ppm.mean()) if ppm.size else float("nan"),
            "negative_precursor_ppm_p50": float(np.quantile(ppm, 0.50)) if ppm.size else float("nan"),
            "negative_precursor_ppm_p90": float(np.quantile(ppm, 0.90)) if ppm.size else float("nan"),
            "negative_precursor_ppm_p99": float(np.quantile(ppm, 0.99)) if ppm.size else float("nan"),
        }

    def __iter__(self):
        # Rank-specific and epoch-specific only: the sampled supervision is
        # deterministic and never depends on model embeddings.
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch + self.rank)
        for _ in range(self.batches_per_epoch):
            selected, tier_counts = self._select_variants(rng)
            self._record_batch(selected, tier_counts, rng)
            indices: list[int] = []
            for variant_id in selected:
                variant = self.variants[variant_id]
                indices.extend(
                    rng.choice(variant.indices, size=self.spectra_per_peptide, replace=False).tolist()
                )
            rng.shuffle(indices)
            if len(indices) != self.batch_size:
                raise RuntimeError(
                    f"Metric-learning sampler produced {len(indices)} spectra; expected {self.batch_size}."
                )
            yield indices
