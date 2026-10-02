"""Metrics and pair selection for precursor-conditioned mixture evaluation."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation


@dataclass(frozen=True)
class CounterfactualPair:
    """A directed synthetic mixture with its clean A and B source rows."""

    anchor_index: int
    distractor_index: int
    partition_id: str


def _stable_int(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _coprime_stride(size: int, seed: int) -> int:
    if size < 2:
        return 1
    stride = 1 + seed % (size - 1)
    while math.gcd(stride, size) != 1:
        stride = stride % (size - 1) + 1
    return stride


def select_counterfactual_pairs(
    dataset,
    *,
    mixer: BatchedStudentDistractorMixAugmentation,
    seed: int,
    max_pairs_per_partition: int | None,
) -> tuple[list[CounterfactualPair], dict[str, dict[str, int]]]:
    """Select deterministic different-peptide A/B pairs under training exclusions.

    Peptide identities are used only to exclude ambiguous A/B pairs whose clean
    targets are identical. The precursor collision eligibility itself exactly
    follows the distractor training mixer.
    """
    table = dataset.table
    precursor_mz = torch.tensor(table["precursor_mz"].to_pylist(), dtype=torch.float32)
    precursor_charge = torch.tensor(
        table["precursor_charge"].to_pylist(), dtype=torch.long
    )
    peptide_ids = [str(value) for value in table[dataset.peptide_id_column].to_pylist()]
    partitions = [str(value) for value in table[dataset.partition_column].to_pylist()]
    eligible = mixer.eligible_distractors(precursor_mz, precursor_charge).cpu()
    by_partition: dict[str, list[int]] = defaultdict(list)
    for index, partition in enumerate(partitions):
        by_partition[partition].append(index)

    records: list[CounterfactualPair] = []
    summary: dict[str, dict[str, int]] = {}
    total = len(peptide_ids)
    for partition, indices in sorted(by_partition.items()):
        ordered_anchors = sorted(
            indices, key=lambda index: _stable_int(seed, "anchor", partition, index)
        )
        selected = 0
        no_eligible_different_peptide = 0
        for anchor in ordered_anchors:
            if max_pairs_per_partition is not None and selected >= max_pairs_per_partition:
                break
            start = _stable_int(seed, "distractor-start", anchor) % total
            stride = _coprime_stride(total, _stable_int(seed, "distractor-step", anchor))
            distractor = None
            for offset in range(total):
                candidate = (start + offset * stride) % total
                if eligible[anchor, candidate] and peptide_ids[candidate] != peptide_ids[anchor]:
                    distractor = candidate
                    break
            if distractor is None:
                no_eligible_different_peptide += 1
                continue
            records.append(CounterfactualPair(anchor, distractor, partition))
            selected += 1
        summary[partition] = {
            "candidate_anchors": len(indices),
            "selected_pairs": selected,
            "no_eligible_different_peptide": no_eligible_different_peptide,
        }
    if not records:
        raise ValueError("No eligible different-peptide counterfactual pairs were selected.")
    return records, summary


def evaluate_null_local_source_consistency(
    clean_source_values: torch.Tensor,
    clean_negative_values: torch.Tensor,
    local_view_one_values: torch.Tensor,
    local_view_two_values: torch.Tensor,
    partition_ids: list[str],
) -> dict[str, object]:
    """Evaluate two null-conditioned local views against the same clean source."""
    tensors = (
        clean_source_values,
        clean_negative_values,
        local_view_one_values,
        local_view_two_values,
    )
    count = clean_source_values.shape[0]
    if count < 1 or any(values.shape != clean_source_values.shape for values in tensors):
        raise ValueError("All null-local tensors must have matching nonempty shapes.")
    if len(partition_ids) != count:
        raise ValueError("Partition IDs must align with null-local representations.")

    clean_source, clean_negative, local_one, local_two = [
        F.normalize(values.float(), dim=1) for values in tensors
    ]

    def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 1.0 - (left * right).sum(dim=1)

    one_to_source = cosine_distance(local_one, clean_source)
    one_to_negative = cosine_distance(local_one, clean_negative)
    two_to_source = cosine_distance(local_two, clean_source)
    two_to_negative = cosine_distance(local_two, clean_negative)
    one_correct = one_to_source.lt(one_to_negative).float()
    two_correct = two_to_source.lt(two_to_negative).float()
    one_margin = one_to_negative - one_to_source
    two_margin = two_to_negative - two_to_source
    local_distance = cosine_distance(local_one, local_two)
    clean_distance = cosine_distance(clean_source, clean_negative)

    values = {
        "local_view_1_source_accuracy": one_correct,
        "local_view_2_source_accuracy": two_correct,
        "paired_source_accuracy": (one_correct.bool() & two_correct.bool()).float(),
        "mean_source_accuracy": 0.5 * (one_correct + two_correct),
        "local_view_1_margin": one_margin,
        "local_view_2_margin": two_margin,
        "mean_source_margin": 0.5 * (one_margin + two_margin),
        "local_view_cosine_distance": local_distance,
        "clean_source_cosine_distance": clean_distance,
        "local_to_clean_distance_ratio": local_distance / clean_distance.clamp_min(1e-8),
    }

    def summarize(indices: torch.Tensor) -> dict[str, float]:
        summary = {name: float(value[indices].mean().item()) for name, value in values.items()}
        summary["pairs"] = int(indices.numel())
        return summary

    partition_rows: dict[str, list[int]] = defaultdict(list)
    for index, partition in enumerate(partition_ids):
        partition_rows[str(partition)].append(index)
    partition_report = {
        partition: summarize(torch.tensor(indices, dtype=torch.long))
        for partition, indices in sorted(partition_rows.items())
    }
    macro = {
        name: sum(report[name] for report in partition_report.values()) / len(partition_report)
        for name in next(iter(partition_report.values()))
        if name != "pairs"
    }
    macro["pairs"] = count
    pooled = summarize(torch.arange(count, dtype=torch.long))
    return {"partitions": partition_report, "macro": macro, "pooled": pooled}


def evaluate_counterfactual_source_selection(
    clean_anchor_values: torch.Tensor,
    clean_distractor_values: torch.Tensor,
    mixed_anchor_condition_values: torch.Tensor,
    mixed_distractor_condition_values: torch.Tensor,
    partition_ids: list[str],
    *,
    distance_metric: str = "cosine",
) -> dict[str, object]:
    """Measure whether each precursor selects its own clean source representation.

    ``cosine`` L2-normalizes representation vectors before comparison.
    ``euclidean`` intentionally preserves raw feature norms. ``jensen_shannon``
    expects non-negative row distributions, such as softmaxed DINO prototype
    logits, and uses their symmetric Jensen-Shannon divergence.
    """
    tensors = (
        clean_anchor_values,
        clean_distractor_values,
        mixed_anchor_condition_values,
        mixed_distractor_condition_values,
    )
    count = clean_anchor_values.shape[0]
    if count < 1 or any(values.shape != clean_anchor_values.shape for values in tensors):
        raise ValueError("All counterfactual embedding tensors must have matching nonempty shapes.")
    if len(partition_ids) != count:
        raise ValueError("Partition IDs must align with counterfactual embeddings.")

    if distance_metric == "cosine":
        clean_a, clean_b, mixed_a, mixed_b = [
            F.normalize(values.float(), dim=1) for values in tensors
        ]

        def distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return 1.0 - (left * right).sum(dim=1)

    elif distance_metric == "euclidean":
        clean_a, clean_b, mixed_a, mixed_b = [values.float() for values in tensors]

        def distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return torch.linalg.vector_norm(left - right, dim=1)

    elif distance_metric == "jensen_shannon":
        if any(not torch.isfinite(values).all() for values in tensors):
            raise ValueError("Jensen-Shannon evaluation requires finite distributions.")
        if any((values < 0).any() for values in tensors):
            raise ValueError(
                "Jensen-Shannon evaluation requires non-negative row distributions; "
                "apply softmax to DINO logits before evaluation."
            )
        clean_a, clean_b, mixed_a, mixed_b = [
            values.float().clamp_min(torch.finfo(torch.float32).tiny)
            for values in tensors
        ]
        clean_a, clean_b, mixed_a, mixed_b = [
            values / values.sum(dim=1, keepdim=True)
            for values in (clean_a, clean_b, mixed_a, mixed_b)
        ]

        def distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            midpoint = 0.5 * (left + right)
            return 0.5 * (
                (left * (left.log() - midpoint.log())).sum(dim=1)
                + (right * (right.log() - midpoint.log())).sum(dim=1)
            )

    else:
        raise ValueError(f"Unsupported counterfactual distance metric {distance_metric!r}.")
    a_to_a = distance(mixed_a, clean_a)
    a_to_b = distance(mixed_a, clean_b)
    b_to_b = distance(mixed_b, clean_b)
    b_to_a = distance(mixed_b, clean_a)
    source_accuracy_a = a_to_a.lt(a_to_b).float()
    source_accuracy_b = b_to_b.lt(b_to_a).float()
    paired_source_accuracy = (source_accuracy_a.bool() & source_accuracy_b.bool()).float()
    source_margin_a = a_to_b - a_to_a
    source_margin_b = b_to_a - b_to_b
    condition_distance = distance(mixed_a, mixed_b)
    clean_source_distance = distance(clean_a, clean_b)

    values = {
        "source_accuracy_anchor_condition": source_accuracy_a,
        "source_accuracy_distractor_condition": source_accuracy_b,
        "paired_source_accuracy": paired_source_accuracy,
        "mean_source_accuracy": 0.5 * (source_accuracy_a + source_accuracy_b),
        "anchor_condition_margin": source_margin_a,
        "distractor_condition_margin": source_margin_b,
        "mean_source_margin": 0.5 * (source_margin_a + source_margin_b),
        f"mixed_condition_{distance_metric}_distance": condition_distance,
        f"clean_source_{distance_metric}_distance": clean_source_distance,
        "condition_to_clean_distance_ratio": condition_distance / clean_source_distance.clamp_min(1e-8),
    }

    def summarize(indices: torch.Tensor) -> dict[str, float]:
        summary = {name: float(value[indices].mean().item()) for name, value in values.items()}
        summary["pairs"] = int(indices.numel())
        return summary

    partition_rows: dict[str, list[int]] = defaultdict(list)
    for index, partition in enumerate(partition_ids):
        partition_rows[str(partition)].append(index)
    partition_report = {
        partition: summarize(torch.tensor(indices, dtype=torch.long))
        for partition, indices in sorted(partition_rows.items())
    }
    macro = {
        name: sum(report[name] for report in partition_report.values()) / len(partition_report)
        for name in next(iter(partition_report.values()))
        if name != "pairs"
    }
    macro["pairs"] = count
    pooled = summarize(torch.arange(count, dtype=torch.long))
    return {"partitions": partition_report, "macro": macro, "pooled": pooled}
