"""Peak-token metrics for fixed-mixture precursor interventions."""

from __future__ import annotations

from collections import defaultdict

import torch


def evaluate_dense_precursor_source_selection(
    anchor_query_advantage: torch.Tensor,
    distractor_query_advantage: torch.Tensor,
    anchor_condition_token_distance: torch.Tensor,
    distractor_condition_token_distance: torch.Tensor,
    anchor_matched_peak_count: torch.Tensor,
    distractor_matched_peak_count: torch.Tensor,
    partition_ids: list[str],
) -> dict[str, object]:
    """Summarize source-aligned dense-token changes by synthetic mixture.

    Each row is one fixed A+B mixture. ``anchor_query_advantage`` is the mean
    cosine similarity of A-only mixed tokens to their clean-A token under the
    A precursor, minus that same similarity under the B precursor. The
    distractor quantity is defined symmetrically for B-only tokens. Positive
    values therefore require a precursor-aligned change for the *same peak*,
    rather than merely a separable peak identity.
    """
    tensors = (
        anchor_query_advantage,
        distractor_query_advantage,
        anchor_condition_token_distance,
        distractor_condition_token_distance,
        anchor_matched_peak_count,
        distractor_matched_peak_count,
    )
    count = anchor_query_advantage.numel()
    if count < 1 or any(values.ndim != 1 or values.numel() != count for values in tensors):
        raise ValueError("Dense counterfactual inputs must be matching nonempty vectors.")
    if len(partition_ids) != count:
        raise ValueError("Partition IDs must align with dense counterfactual rows.")
    if (anchor_matched_peak_count <= 0).any() or (distractor_matched_peak_count <= 0).any():
        raise ValueError("Every evaluated mixture must include matched A-only and B-only peaks.")

    anchor_advantage = anchor_query_advantage.float()
    distractor_advantage = distractor_query_advantage.float()
    values = {
        "anchor_query_advantage": anchor_advantage,
        "distractor_query_advantage": distractor_advantage,
        "mean_query_advantage": 0.5 * (anchor_advantage + distractor_advantage),
        "anchor_query_alignment": anchor_advantage.gt(0).float(),
        "distractor_query_alignment": distractor_advantage.gt(0).float(),
        "paired_query_alignment": (
            anchor_advantage.gt(0) & distractor_advantage.gt(0)
        ).float(),
        "anchor_condition_token_cosine_distance": anchor_condition_token_distance.float(),
        "distractor_condition_token_cosine_distance": distractor_condition_token_distance.float(),
        "mean_condition_token_cosine_distance": 0.5
        * (anchor_condition_token_distance.float() + distractor_condition_token_distance.float()),
    }

    def summarize(indices: torch.Tensor) -> dict[str, float | int]:
        summary: dict[str, float | int] = {
            name: float(value[indices].mean().item()) for name, value in values.items()
        }
        summary["matched_anchor_peaks"] = int(anchor_matched_peak_count[indices].sum().item())
        summary["matched_distractor_peaks"] = int(distractor_matched_peak_count[indices].sum().item())
        summary["pairs"] = int(indices.numel())
        return summary

    partition_rows: dict[str, list[int]] = defaultdict(list)
    for index, partition in enumerate(partition_ids):
        partition_rows[str(partition)].append(index)
    partitions = {
        partition: summarize(torch.tensor(indices, dtype=torch.long))
        for partition, indices in sorted(partition_rows.items())
    }
    macro: dict[str, float | int] = {
        name: sum(float(report[name]) for report in partitions.values()) / len(partitions)
        for name in values
    }
    macro["pairs"] = count
    macro["matched_anchor_peaks"] = int(anchor_matched_peak_count.sum().item())
    macro["matched_distractor_peaks"] = int(distractor_matched_peak_count.sum().item())
    pooled = summarize(torch.arange(count, dtype=torch.long))
    return {"partitions": partitions, "macro": macro, "pooled": pooled}
