"""Deterministic labelled peptide-ion pair-discrimination metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F


def _prepare_vectors(values: torch.Tensor, metric: str) -> torch.Tensor:
    """Keep embeddings on CPU; pair chunks are copied to the metric device."""
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if metric == "cosine":
        return F.normalize(values, dim=1)
    if metric == "euclidean":
        return values
    raise ValueError(f"Unsupported pair distance metric: {metric!r}")


def _indexed_records(
    spectrum_index: Mapping[str, int], records: Iterable[Mapping[str, object]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve pair IDs once, storing compact CPU index and label tensors."""
    left_indices = []
    right_indices = []
    labels = []
    for record in records:
        left_id = str(record["left_spectrum_id"])
        right_id = str(record["right_spectrum_id"])
        if left_id == right_id:
            raise ValueError(f"Pair benchmark contains a self-pair: {left_id!r}")
        try:
            left_indices.append(spectrum_index[left_id])
            right_indices.append(spectrum_index[right_id])
        except KeyError as exc:
            raise ValueError(
                f"Pair benchmark references an embedding that was not extracted: {exc.args[0]!r}"
            ) from exc
        label = int(record["label"])
        if label not in {0, 1}:
            raise ValueError(f"Pair label must be 0 or 1, found {label!r}.")
        labels.append(label)
    if not labels:
        raise ValueError("Pair metric received no pair records.")
    return (
        torch.tensor(left_indices, dtype=torch.long),
        torch.tensor(right_indices, dtype=torch.long),
        torch.tensor(labels, dtype=torch.bool),
    )


def _pair_distances(
    values: torch.Tensor,
    spectrum_index: Mapping[str, int],
    records: Iterable[Mapping[str, object]],
    *,
    metric: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact distances in bounded memory, retaining only scalar results."""
    left_indices, right_indices, labels = _indexed_records(spectrum_index, records)
    distances = []
    for start in range(0, labels.numel(), batch_size):
        end = min(start + batch_size, labels.numel())
        left = values[left_indices[start:end]].to(device)
        right = values[right_indices[start:end]].to(device)
        if metric == "cosine":
            distance = (1.0 - (left * right).sum(dim=1)).clamp_min_(0.0)
        else:
            distance = torch.linalg.vector_norm(left - right, dim=1)
        distances.append(distance.cpu())
    return torch.cat(distances), labels


def _cross_view_pair_distances(
    query_values: torch.Tensor,
    gallery_values: torch.Tensor,
    spectrum_index: Mapping[str, int],
    records: Iterable[Mapping[str, object]],
    *,
    metric: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average both cross-view directions in bounded memory."""
    left_indices, right_indices, labels = _indexed_records(spectrum_index, records)
    distances = []
    for start in range(0, labels.numel(), batch_size):
        end = min(start + batch_size, labels.numel())
        left_index = left_indices[start:end]
        right_index = right_indices[start:end]
        query_left = query_values[left_index].to(device)
        query_right = query_values[right_index].to(device)
        gallery_left = gallery_values[left_index].to(device)
        gallery_right = gallery_values[right_index].to(device)
        if metric == "cosine":
            forward = (1.0 - (query_left * gallery_right).sum(dim=1)).clamp_min_(0.0)
            reverse = (1.0 - (query_right * gallery_left).sum(dim=1)).clamp_min_(0.0)
        else:
            forward = torch.linalg.vector_norm(query_left - gallery_right, dim=1)
            reverse = torch.linalg.vector_norm(query_right - gallery_left, dim=1)
        distances.append(((forward + reverse) / 2.0).cpu())
    return torch.cat(distances), labels


def _grouped_distance_counts(
    distances: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate true/false pair counts at each distinct accepted distance."""
    order = torch.argsort(distances)
    ordered_distances = distances[order]
    ordered_labels = labels[order]
    boundary = torch.ones_like(ordered_labels, dtype=torch.bool)
    boundary[1:] = ordered_distances[1:] != ordered_distances[:-1]
    group_id = boundary.cumsum(dim=0) - 1
    group_count = int(group_id[-1].item()) + 1
    true_counts = torch.zeros(group_count, dtype=torch.float64, device=distances.device)
    false_counts = torch.zeros_like(true_counts)
    true_counts.scatter_add_(0, group_id, ordered_labels.to(torch.float64))
    false_counts.scatter_add_(0, group_id, (~ordered_labels).to(torch.float64))
    thresholds = ordered_distances[boundary]
    return thresholds, true_counts.cumsum(0), false_counts.cumsum(0), true_counts


def _roc_auc(distances: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact ROC AUC for the score ``-distance``, with average tied ranks."""
    positive_count = int(labels.sum().item())
    negative_count = int((~labels).sum().item())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = torch.argsort(distances)
    sorted_distances = distances[order]
    sorted_labels = labels[order]
    rank_sum_positive = 0.0
    start = 0
    while start < sorted_distances.numel():
        end = start + 1
        while end < sorted_distances.numel() and sorted_distances[end] == sorted_distances[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum_positive += average_rank * float(sorted_labels[start:end].sum().item())
        start = end
    low_distance_u = rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    return float(1.0 - low_distance_u / (positive_count * negative_count))


def _average_precision(
    true_cumulative: torch.Tensor,
    false_cumulative: torch.Tensor,
    true_increment: torch.Tensor,
    positive_count: int,
) -> float:
    """Average precision at distance thresholds, invariant to within-tie ordering."""
    if positive_count == 0:
        return float("nan")
    accepted = true_cumulative + false_cumulative
    precision = true_cumulative / accepted
    recall_increment = true_increment / positive_count
    return float((precision * recall_increment).sum().item())


def _fdr_operating_point(
    thresholds: torch.Tensor,
    true_cumulative: torch.Tensor,
    false_cumulative: torch.Tensor,
    positive_count: int,
    fdr_level: float,
) -> tuple[float, float | None]:
    """Return FNR and largest distance threshold satisfying balanced-pair FDR."""
    if not 0.0 <= fdr_level < 1.0:
        raise ValueError("Each balanced-pair FDR level must be in [0, 1).")
    accepted = true_cumulative + false_cumulative
    fdr = false_cumulative / accepted
    valid = torch.nonzero(fdr <= fdr_level, as_tuple=False).flatten()
    if valid.numel() == 0:
        return 1.0, None
    index = int(valid[-1].item())
    fnr = 1.0 - float(true_cumulative[index].item()) / positive_count
    return fnr, float(thresholds[index].item())


def _purity_curve(
    thresholds: torch.Tensor,
    true_cumulative: torch.Tensor,
    false_cumulative: torch.Tensor,
    positive_count: int,
    coverage_levels: Iterable[float],
) -> list[dict[str, float]]:
    """Report pair purity at fixed positive-pair coverage targets."""
    accepted = true_cumulative + false_cumulative
    coverage = true_cumulative / positive_count
    curve = []
    for target in coverage_levels:
        target_float = float(target)
        if not 0.0 < target_float <= 1.0:
            raise ValueError("purity coverage levels must be in (0, 1].")
        index = int(torch.searchsorted(coverage, target_float, right=False).item())
        index = min(index, coverage.numel() - 1)
        curve.append(
            {
                "target_positive_coverage": target_float,
                "positive_coverage": float(coverage[index].item()),
                "pair_purity": float((true_cumulative[index] / accepted[index]).item()),
                "distance_threshold": float(thresholds[index].item()),
            }
        )
    return curve


def pair_discrimination_metrics(
    distances: torch.Tensor,
    labels: torch.Tensor,
    *,
    fdr_levels: Iterable[float],
    purity_coverage_levels: Iterable[float],
    require_balanced_pairs: bool = True,
) -> dict[str, object]:
    """Evaluate same-peptide separation on a fixed labelled pair sample.

    FDR values are *balanced-pair FDR*: accepted negative pairs divided by all
    accepted pairs under a deliberately fixed positive:negative sampling ratio.
    They are not database-search FDR estimates.
    """
    if distances.ndim != 1 or labels.ndim != 1 or distances.numel() != labels.numel():
        raise ValueError("distances and labels must be aligned one-dimensional tensors.")
    positive_count = int(labels.sum().item())
    negative_count = int((~labels).sum().item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Each pair metric requires both positive and negative pairs.")
    if require_balanced_pairs and positive_count != negative_count:
        raise ValueError(
            "Pair benchmark must use a 1:1 positive:negative ratio; "
            f"found {positive_count}:{negative_count}."
        )

    thresholds, true_cumulative, false_cumulative, true_increment = _grouped_distance_counts(
        distances, labels
    )
    result: dict[str, object] = {
        "roc_auc": _roc_auc(distances, labels),
        "average_precision": _average_precision(
            true_cumulative, false_cumulative, true_increment, positive_count
        ),
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "pair_count": positive_count + negative_count,
        "positive_to_negative_ratio": positive_count / negative_count,
    }
    for level in fdr_levels:
        level_float = float(level)
        suffix = f"{level_float:.2f}".replace(".", "p")
        fnr, threshold = _fdr_operating_point(
            thresholds,
            true_cumulative,
            false_cumulative,
            positive_count,
            level_float,
        )
        result[f"fnr_at_balanced_fdr_{suffix}"] = fnr
        result[f"threshold_at_balanced_fdr_{suffix}"] = threshold
    result["purity_at_positive_coverage"] = _purity_curve(
        thresholds,
        true_cumulative,
        false_cumulative,
        positive_count,
        purity_coverage_levels,
    )
    return result


def _availability_metrics(availability: Mapping[str, object] | None) -> dict[str, float]:
    if not availability:
        return {}
    result: dict[str, float] = {}
    for key in (
        "hard_negative_anchor_coverage",
        "hard_negative_pair_count",
        "hard_negative_anchor_count",
        "positive_anchor_count",
    ):
        value = availability.get(key)
        if value is not None:
            result[key] = float(value)
    return result


def _macro(results: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    metric_names = sorted(
        {
            metric_name
            for result in results.values()
            for metric_name, value in result.items()
            if metric_name in {"roc_auc", "average_precision"}
            or metric_name.startswith("fnr_at_balanced_fdr_")
            if isinstance(value, float) and math.isfinite(value)
        }
    )
    return {
        metric_name: sum(
            float(result[metric_name])
            for result in results.values()
            if isinstance(result.get(metric_name), float)
            and math.isfinite(float(result[metric_name]))
        )
        / sum(
            isinstance(result.get(metric_name), float)
            and math.isfinite(float(result[metric_name]))
            for result in results.values()
        )
        for metric_name in metric_names
    }


def evaluate_pair_discrimination(
    values: torch.Tensor,
    spectrum_ids: list[str],
    pair_records: Iterable[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    availability_by_set_species: Mapping[str, Mapping[str, Mapping[str, object]]] | None,
    device: torch.device,
    comparison_values: torch.Tensor | None = None,
) -> dict[str, object]:
    """Evaluate random and precursor-matched static pair protocols by species."""
    if values.ndim != 2:
        raise ValueError("Embedding values must have shape [spectra, dimension].")
    if len(spectrum_ids) != values.shape[0]:
        raise ValueError("spectrum_ids must align with embedding rows.")
    if len(set(spectrum_ids)) != len(spectrum_ids):
        raise ValueError("Pair benchmark spectrum IDs must be unique.")
    metric = str(config.get("metric", "cosine"))
    pair_distance_batch_size = int(config.get("pair_distance_batch_size", 65_536))
    if pair_distance_batch_size < 1:
        raise ValueError("pair_distance_batch_size must be positive.")
    prepared = _prepare_vectors(values, metric)
    prepared_comparison = (
        _prepare_vectors(comparison_values, metric)
        if comparison_values is not None
        else None
    )
    if prepared_comparison is not None and prepared_comparison.shape != prepared.shape:
        raise ValueError("comparison_values must match values shape for cross-view pair evaluation.")
    spectrum_index = {spectrum_id: index for index, spectrum_id in enumerate(spectrum_ids)}
    records_by_set_species: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in pair_records:
        pair_set = str(record["pair_set"])
        species = str(record["species"])
        records_by_set_species[pair_set][species].append(record)
    if not records_by_set_species:
        raise ValueError("Pair benchmark contains no protocols.")

    fdr_levels = [float(level) for level in config.get("balanced_pair_fdr_levels", [0.01, 0.05, 0.10])]
    purity_coverage_levels = [
        float(level)
        for level in config.get("purity_coverage_levels", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0])
    ]
    pair_set_results: dict[str, object] = {}
    for pair_set, by_species in sorted(records_by_set_species.items()):
        partitions: dict[str, dict[str, object]] = {}
        pooled_distances = []
        pooled_labels = []
        for species, records in sorted(by_species.items()):
            distances, labels = (
                _cross_view_pair_distances(
                    prepared,
                    prepared_comparison,
                    spectrum_index,
                    records,
                    metric=metric,
                    device=device,
                    batch_size=pair_distance_batch_size,
                )
                if prepared_comparison is not None
                else _pair_distances(
                    prepared,
                    spectrum_index,
                    records,
                    metric=metric,
                    device=device,
                    batch_size=pair_distance_batch_size,
                )
            )
            metrics = pair_discrimination_metrics(
                distances,
                labels,
                fdr_levels=fdr_levels,
                purity_coverage_levels=purity_coverage_levels,
            )
            availability = None
            if availability_by_set_species:
                availability = availability_by_set_species.get(pair_set, {}).get(species)
            metrics.update(_availability_metrics(availability))
            partitions[species] = metrics
            pooled_distances.append(distances)
            pooled_labels.append(labels)
        pooled = pair_discrimination_metrics(
            torch.cat(pooled_distances),
            torch.cat(pooled_labels),
            fdr_levels=fdr_levels,
            purity_coverage_levels=purity_coverage_levels,
        )
        if pair_set == "same_charge_10ppm":
            anchor_count = sum(
                float(result.get("hard_negative_anchor_count", 0.0))
                for result in partitions.values()
            )
            positive_anchor_count = sum(
                float(result.get("positive_anchor_count", 0.0))
                for result in partitions.values()
            )
            if positive_anchor_count:
                pooled["hard_negative_anchor_coverage"] = (
                    anchor_count / positive_anchor_count
                )
            pooled["hard_negative_pair_count"] = sum(
                float(result.get("hard_negative_pair_count", 0.0))
                for result in partitions.values()
            )
        pair_set_results[pair_set] = {
            "partitions": partitions,
            "macro": _macro(partitions),
            "pooled": pooled,
        }
    return {
        "pair_sets": pair_set_results,
        "distance_metric": metric,
        "cross_view": comparison_values is not None,
    }
