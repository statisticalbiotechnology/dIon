"""Peptide retrieval and compactness metrics for spectrum embeddings."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _stable_index(seed: int, size: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _group_indices(peptide_ids: Iterable[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, peptide_id in enumerate(peptide_ids):
        groups[str(peptide_id)].append(index)
    return {peptide_id: indices for peptide_id, indices in groups.items() if len(indices) >= 2}


def _prepare_vectors(values: torch.Tensor, metric: str) -> torch.Tensor:
    values = values.detach().to(dtype=torch.float32)
    if metric == "cosine":
        return F.normalize(values, dim=1)
    if metric == "euclidean":
        return values
    raise ValueError(f"Unsupported retrieval metric: {metric!r}")


def _distance_matrix(left: torch.Tensor, right: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "cosine":
        return (1.0 - left @ right.T).clamp_min_(0.0)
    return torch.cdist(left, right)


def _mean_pairwise_distance(values: torch.Tensor, metric: str) -> torch.Tensor:
    distances = _distance_matrix(values, values, metric)
    mask = ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
    return distances[mask].mean()


def _retrieval_metrics(
    values: torch.Tensor,
    peptide_ids: list[str],
    groups: dict[str, list[int]],
    *,
    metric: str,
    ks: list[int],
    query_seed: int,
    query_batch_size: int,
    show_progress: bool,
    progress_desc: str,
) -> dict[str, float]:
    if query_batch_size < 1:
        raise ValueError("retrieval.query_batch_size must be positive.")
    ordered_groups = sorted(groups)
    query_indices = torch.tensor(
        [
            groups[peptide_id][
                _stable_index(query_seed, len(groups[peptide_id]), peptide_id)
            ]
            for peptide_id in ordered_groups
        ],
        dtype=torch.long,
        device=values.device,
    )
    label_lookup = {peptide_id: label for label, peptide_id in enumerate(ordered_groups)}
    label_ids = torch.tensor(
        [label_lookup[peptide_id] for peptide_id in peptide_ids],
        dtype=torch.long,
        device=values.device,
    )
    group_sizes = torch.bincount(label_ids, minlength=len(ordered_groups))
    sums = {f"r_at_{k}": 0.0 for k in ks}
    sums.update({f"hit_at_{k}": 0.0 for k in ks})
    map_sum = 0.0
    mrr_sum = 0.0

    for query_chunk in tqdm(
        query_indices.split(query_batch_size),
        desc=progress_desc,
        unit="chunk",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        distances = _distance_matrix(values[query_chunk], values, metric)
        row_indices = torch.arange(query_chunk.numel(), device=values.device)
        distances[row_indices, query_chunk] = float("inf")
        ordered = distances.argsort(dim=1)
        relevant = label_ids[ordered].eq(label_ids[query_chunk].unsqueeze(1))
        # The query itself sorts last after its distance is set to infinity;
        # remove it explicitly so full-ranking AP uses only other spectra.
        relevant &= ordered.ne(query_chunk.unsqueeze(1))
        n_relevant = group_sizes[label_ids[query_chunk]] - 1
        ranks = torch.arange(1, relevant.shape[1] + 1, device=values.device)
        precision = relevant.cumsum(dim=1).to(values.dtype) / ranks
        map_sum += float(
            (precision * relevant).sum(dim=1).div(n_relevant).sum().item()
        )
        first_rank = relevant.float().argmax(dim=1).add(1)
        mrr_sum += float(first_rank.to(values.dtype).reciprocal().sum().item())
        for k in ks:
            top_relevant = relevant[:, : min(k, relevant.shape[1])].sum(dim=1)
            sums[f"r_at_{k}"] += float(
                top_relevant.to(values.dtype).div(n_relevant).sum().item()
            )
            sums[f"hit_at_{k}"] += float(top_relevant.gt(0).sum().item())

    query_count = int(query_indices.numel())
    metrics = {name: value / query_count for name, value in sums.items()}
    metrics["map"] = map_sum / query_count
    metrics["mrr"] = mrr_sum / query_count
    metrics["n_queries"] = float(query_count)
    return metrics


def _expected_random_hit_at_k(candidate_count: int, positive_count: int, k: int) -> float:
    """Probability that a random ranking contains a positive in its top-k."""
    k = min(k, candidate_count)
    if k < 1 or positive_count < 1:
        return 0.0
    if candidate_count - positive_count < k:
        return 1.0
    no_hit = 1.0
    for offset in range(k):
        no_hit *= (candidate_count - positive_count - offset) / (candidate_count - offset)
    return 1.0 - no_hit


def _same_charge_ppm_candidates(
    precursor_mz: torch.Tensor,
    precursor_charges: torch.Tensor,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Index local row ids by charge then precursor m/z for exact ppm ranges."""
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for charge in torch.unique(precursor_charges).tolist():
        indices = precursor_charges.eq(charge).nonzero(as_tuple=False).flatten()
        order = precursor_mz[indices].argsort()
        result[int(charge)] = (precursor_mz[indices][order], indices[order])
    return result


def _mass_controlled_retrieval_metrics(
    values: torch.Tensor,
    peptide_ids: list[str],
    groups: dict[str, list[int]],
    precursor_mz: torch.Tensor,
    precursor_charges: torch.Tensor,
    *,
    metric: str,
    ks: list[int],
    query_seed: int,
    ppm_tolerance: float,
    show_progress: bool = False,
    progress_desc: str = "Mass-controlled retrieval",
) -> dict[str, float]:
    """Exact same-charge/ppm retrieval using sorted precursor neighborhoods."""
    if ppm_tolerance <= 0:
        raise ValueError("mass_controlled_retrieval.ppm_tolerance must be positive.")
    ordered_groups = sorted(groups)
    query_indices = [
        groups[peptide_id][_stable_index(query_seed, len(groups[peptide_id]), peptide_id)]
        for peptide_id in ordered_groups
    ]
    label_lookup = {peptide_id: label for label, peptide_id in enumerate(ordered_groups)}
    label_ids = torch.tensor([label_lookup[peptide_id] for peptide_id in peptide_ids], dtype=torch.long, device=values.device)
    precursor_mz = precursor_mz.to(values.device, dtype=torch.float32)
    precursor_charges = precursor_charges.to(values.device)
    charge_index = _same_charge_ppm_candidates(precursor_mz, precursor_charges)
    tolerance = ppm_tolerance * 1e-6
    sums = {f"mass_controlled_r_at_{k}": 0.0 for k in ks}
    sums.update({f"mass_controlled_hit_at_{k}": 0.0 for k in ks})
    random_sums = {f"mass_controlled_random_r_at_{k}": 0.0 for k in ks}
    random_sums.update({f"mass_controlled_random_hit_at_{k}": 0.0 for k in ks})
    map_sum = mrr_sum = 0.0
    candidate_counts: list[int] = []
    eligible = 0
    for query_index in tqdm(query_indices, desc=progress_desc, unit="query", dynamic_ncols=True, disable=not show_progress):
        sorted_mz, sorted_indices = charge_index[int(precursor_charges[query_index].item())]
        query_mz = precursor_mz[query_index]
        lower = torch.searchsorted(sorted_mz, query_mz * (1.0 - tolerance), right=False)
        upper = torch.searchsorted(sorted_mz, query_mz * (1.0 + tolerance), right=True)
        candidates = sorted_indices[lower:upper]
        candidates = candidates[candidates.ne(query_index)]
        if candidates.numel() == 0:
            continue
        relevant = label_ids[candidates].eq(label_ids[query_index])
        positive_count = int(relevant.sum().item())
        negative_count = int((~relevant).sum().item())
        if positive_count < 1 or negative_count < 1:
            continue
        eligible += 1
        candidate_count = int(candidates.numel())
        candidate_counts.append(candidate_count)
        distances = _distance_matrix(values[query_index : query_index + 1], values[candidates], metric).squeeze(0)
        ranked_relevant = relevant[distances.argsort()]
        ranks = torch.arange(1, candidate_count + 1, device=values.device)
        precision = ranked_relevant.cumsum(dim=0).to(values.dtype) / ranks
        map_sum += float((precision * ranked_relevant).sum().div(positive_count).item())
        mrr_sum += float(ranked_relevant.float().argmax().add(1).reciprocal().item())
        for k in ks:
            top_relevant = ranked_relevant[: min(k, candidate_count)].sum().item()
            sums[f"mass_controlled_r_at_{k}"] += top_relevant / positive_count
            sums[f"mass_controlled_hit_at_{k}"] += float(top_relevant > 0)
            random_sums[f"mass_controlled_random_r_at_{k}"] += min(k, candidate_count) / candidate_count
            random_sums[f"mass_controlled_random_hit_at_{k}"] += _expected_random_hit_at_k(candidate_count, positive_count, k)
    query_count = len(query_indices)
    if eligible == 0:
        return {"mass_controlled_eligible_query_fraction": 0.0, "mass_controlled_n_eligible_queries": 0.0, "mass_controlled_mean_candidate_count": float("nan")}
    metrics = {name: value / eligible for name, value in sums.items()}
    metrics.update({name: value / eligible for name, value in random_sums.items()})
    metrics["mass_controlled_map"] = map_sum / eligible
    metrics["mass_controlled_mrr"] = mrr_sum / eligible
    metrics["mass_controlled_eligible_query_fraction"] = eligible / query_count
    metrics["mass_controlled_n_eligible_queries"] = float(eligible)
    metrics["mass_controlled_mean_candidate_count"] = sum(candidate_counts) / eligible
    return metrics


def _cross_charge_retrieval_metrics(
    values: torch.Tensor,
    peptide_ids: list[str],
    groups: dict[str, list[int]],
    precursor_mz: torch.Tensor,
    precursor_charges: torch.Tensor,
    *,
    metric: str,
    ks: list[int],
    query_seed: int,
    neutral_mass_ppm_tolerance: float | None,
    query_batch_size: int = 128,
    show_progress: bool = False,
    progress_desc: str = "Cross-charge retrieval",
) -> dict[str, float]:
    """Exact cross-charge retrieval, batched broadly and indexed when strict."""
    ordered_groups = sorted(groups)
    label_lookup = {peptide_id: label for label, peptide_id in enumerate(ordered_groups)}
    label_ids = torch.tensor([label_lookup[peptide_id] for peptide_id in peptide_ids], dtype=torch.long, device=values.device)
    charges = precursor_charges.to(values.device)
    neutral_masses = precursor_mz.to(values.device, dtype=torch.float32) * charges.to(values.device, dtype=torch.float32)
    query_indices = torch.tensor(
        [groups[peptide_id][_stable_index(query_seed, len(groups[peptide_id]), peptide_id)] for peptide_id in ordered_groups],
        dtype=torch.long, device=values.device,
    )
    sums = {f"cross_charge_r_at_{k}": 0.0 for k in ks}
    sums.update({f"cross_charge_hit_at_{k}": 0.0 for k in ks})
    random_sums = {f"cross_charge_random_r_at_{k}": 0.0 for k in ks}
    random_sums.update({f"cross_charge_random_hit_at_{k}": 0.0 for k in ks})
    map_sum = mrr_sum = 0.0
    candidate_counts: list[int] = []
    eligible = 0

    if neutral_mass_ppm_tolerance is None:
        # All queries of one charge share one gallery. This replaces thousands
        # of single-query full-partition scans with GPU matrix batches.
        for charge in torch.unique(charges).tolist():
            charge_queries = query_indices[charges[query_indices].eq(charge)]
            gallery = charges.ne(charge).nonzero(as_tuple=False).flatten()
            if gallery.numel() == 0:
                continue
            gallery_labels = label_ids[gallery]
            for chunk in tqdm(charge_queries.split(query_batch_size), desc=progress_desc, unit="query", dynamic_ncols=True, disable=not show_progress):
                distances = _distance_matrix(values[chunk], values[gallery], metric)
                ordered = distances.argsort(dim=1)
                relevant = gallery_labels[ordered].eq(label_ids[chunk].unsqueeze(1))
                positive_counts = relevant.sum(dim=1)
                valid = positive_counts.gt(0) & positive_counts.lt(gallery.numel())
                if not valid.any():
                    continue
                eligible_indices = valid.nonzero(as_tuple=False).flatten()
                relevant = relevant[eligible_indices]
                positive_counts = positive_counts[eligible_indices]
                eligible += int(eligible_indices.numel())
                candidate_count = int(gallery.numel())
                candidate_counts.extend([candidate_count] * int(eligible_indices.numel()))
                ranks = torch.arange(1, candidate_count + 1, device=values.device)
                precision = relevant.cumsum(dim=1).to(values.dtype) / ranks
                map_sum += float((precision * relevant).sum(dim=1).div(positive_counts).sum().item())
                mrr_sum += float(relevant.float().argmax(dim=1).add(1).to(values.dtype).reciprocal().sum().item())
                for k in ks:
                    top_relevant = relevant[:, :min(k, candidate_count)].sum(dim=1)
                    sums[f"cross_charge_r_at_{k}"] += float(top_relevant.to(values.dtype).div(positive_counts).sum().item())
                    sums[f"cross_charge_hit_at_{k}"] += float(top_relevant.gt(0).sum().item())
                for positive_count in positive_counts.tolist():
                    for k in ks:
                        random_sums[f"cross_charge_random_r_at_{k}"] += min(k, candidate_count) / candidate_count
                        random_sums[f"cross_charge_random_hit_at_{k}"] += _expected_random_hit_at_k(candidate_count, int(positive_count), k)
    else:
        if neutral_mass_ppm_tolerance <= 0:
            raise ValueError("cross_charge_retrieval.neutral_mass_ppm_tolerance must be positive.")
        order = neutral_masses.argsort()
        sorted_masses, sorted_indices = neutral_masses[order], order
        tolerance = neutral_mass_ppm_tolerance * 1e-6
        for query_index in tqdm(query_indices.tolist(), desc=progress_desc, unit="query", dynamic_ncols=True, disable=not show_progress):
            mass = neutral_masses[query_index]
            lower = torch.searchsorted(sorted_masses, mass * (1.0 - tolerance), right=False)
            upper = torch.searchsorted(sorted_masses, mass * (1.0 + tolerance), right=True)
            candidates = sorted_indices[lower:upper]
            candidates = candidates[charges[candidates].ne(charges[query_index])]
            if candidates.numel() == 0:
                continue
            relevant = label_ids[candidates].eq(label_ids[query_index])
            positive_count = int(relevant.sum().item())
            negative_count = int((~relevant).sum().item())
            if positive_count < 1 or negative_count < 1:
                continue
            eligible += 1
            candidate_count = int(candidates.numel())
            candidate_counts.append(candidate_count)
            distances = _distance_matrix(values[query_index : query_index + 1], values[candidates], metric).squeeze(0)
            ranked_relevant = relevant[distances.argsort()]
            ranks = torch.arange(1, candidate_count + 1, device=values.device)
            precision = ranked_relevant.cumsum(dim=0).to(values.dtype) / ranks
            map_sum += float((precision * ranked_relevant).sum().div(positive_count).item())
            mrr_sum += float(ranked_relevant.float().argmax().add(1).reciprocal().item())
            for k in ks:
                top_relevant = ranked_relevant[:min(k, candidate_count)].sum().item()
                sums[f"cross_charge_r_at_{k}"] += top_relevant / positive_count
                sums[f"cross_charge_hit_at_{k}"] += float(top_relevant > 0)
                random_sums[f"cross_charge_random_r_at_{k}"] += min(k, candidate_count) / candidate_count
                random_sums[f"cross_charge_random_hit_at_{k}"] += _expected_random_hit_at_k(candidate_count, positive_count, k)

    if eligible == 0:
        return {"cross_charge_eligible_query_fraction": 0.0, "cross_charge_n_eligible_queries": 0.0, "cross_charge_mean_candidate_count": float("nan")}
    metrics = {name: value / eligible for name, value in sums.items()}
    metrics.update({name: value / eligible for name, value in random_sums.items()})
    metrics["cross_charge_map"] = map_sum / eligible
    metrics["cross_charge_mrr"] = mrr_sum / eligible
    metrics["cross_charge_eligible_query_fraction"] = eligible / len(ordered_groups)
    metrics["cross_charge_n_eligible_queries"] = float(eligible)
    metrics["cross_charge_mean_candidate_count"] = sum(candidate_counts) / eligible
    if neutral_mass_ppm_tolerance is not None:
        metrics["cross_charge_neutral_mass_ppm_tolerance"] = float(neutral_mass_ppm_tolerance)
    return metrics


def _cross_view_retrieval_metrics(
    query_values: torch.Tensor,
    gallery_values: torch.Tensor,
    peptide_ids: list[str],
    groups: dict[str, list[int]],
    *,
    metric: str,
    ks: list[int],
    query_seed: int,
    candidate_mask_factory=None,
) -> dict[str, float]:
    """Retrieve clean-gallery peptide peers from a second embedding view.

    The matching clean selected row is excluded. This prevents the result from
    reducing to instance matching and asks whether a perturbed query still finds
    *other* spectra from the same peptide.
    """
    ordered_groups = sorted(groups)
    label_lookup = {peptide_id: label for label, peptide_id in enumerate(ordered_groups)}
    label_ids = torch.tensor(
        [label_lookup[peptide_id] for peptide_id in peptide_ids],
        dtype=torch.long,
        device=query_values.device,
    )
    sums = {f"cross_view_r_at_{k}": 0.0 for k in ks}
    sums.update({f"cross_view_hit_at_{k}": 0.0 for k in ks})
    random_sums = {f"cross_view_random_r_at_{k}": 0.0 for k in ks}
    random_sums.update({f"cross_view_random_hit_at_{k}": 0.0 for k in ks})
    map_sum = 0.0
    mrr_sum = 0.0
    candidate_counts = []
    eligible = 0
    all_indices = torch.arange(len(peptide_ids), device=query_values.device)

    for peptide_id in ordered_groups:
        query_index = groups[peptide_id][
            _stable_index(query_seed, len(groups[peptide_id]), peptide_id)
        ]
        candidate_mask = torch.ones(len(peptide_ids), dtype=torch.bool, device=query_values.device)
        candidate_mask[query_index] = False
        if candidate_mask_factory is not None:
            candidate_mask &= candidate_mask_factory(query_index)
        candidates = all_indices[candidate_mask]
        if candidates.numel() == 0:
            continue
        relevant = label_ids[candidates].eq(label_ids[query_index])
        positive_count = int(relevant.sum().item())
        negative_count = int((~relevant).sum().item())
        if positive_count < 1 or negative_count < 1:
            continue
        eligible += 1
        candidate_count = int(candidates.numel())
        candidate_counts.append(candidate_count)
        distances = _distance_matrix(
            query_values[query_index : query_index + 1], gallery_values[candidates], metric
        ).squeeze(0)
        ranked_relevant = relevant[distances.argsort()]
        ranks = torch.arange(1, candidate_count + 1, device=query_values.device)
        precision = ranked_relevant.cumsum(dim=0).to(query_values.dtype) / ranks
        map_sum += float((precision * ranked_relevant).sum().div(positive_count).item())
        mrr_sum += float(ranked_relevant.float().argmax().add(1).reciprocal().item())
        for k in ks:
            top_relevant = ranked_relevant[: min(k, candidate_count)].sum().item()
            sums[f"cross_view_r_at_{k}"] += top_relevant / positive_count
            sums[f"cross_view_hit_at_{k}"] += float(top_relevant > 0)
            random_sums[f"cross_view_random_r_at_{k}"] += min(k, candidate_count) / candidate_count
            random_sums[f"cross_view_random_hit_at_{k}"] += _expected_random_hit_at_k(
                candidate_count, positive_count, k
            )

    if eligible == 0:
        return {
            "cross_view_eligible_query_fraction": 0.0,
            "cross_view_n_eligible_queries": 0.0,
            "cross_view_mean_candidate_count": float("nan"),
        }
    metrics = {name: value / eligible for name, value in sums.items()}
    metrics.update({name: value / eligible for name, value in random_sums.items()})
    metrics["cross_view_map"] = map_sum / eligible
    metrics["cross_view_mrr"] = mrr_sum / eligible
    metrics["cross_view_eligible_query_fraction"] = eligible / len(ordered_groups)
    metrics["cross_view_n_eligible_queries"] = float(eligible)
    metrics["cross_view_mean_candidate_count"] = sum(candidate_counts) / eligible
    return metrics


def evaluate_clean_gallery_robustness(
    query_embeddings: torch.Tensor,
    clean_gallery_embeddings: torch.Tensor,
    peptide_ids: list[str],
    partition_ids: list[str],
    precursor_mz: torch.Tensor,
    precursor_charges: torch.Tensor,
    config: dict,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate perturbed queries against a clean gallery by species.

    Reports broad retrieval and a same-charge, 10-ppm precursor-controlled
    retrieval variant. The latter is the relevant robustness result when broad
    peptide identity can be inferred from precursor metadata.
    """
    if query_embeddings.shape != clean_gallery_embeddings.shape:
        raise ValueError("Query and clean-gallery embeddings must have matching shape.")
    metric = config.get("retrieval", {}).get("metric", "cosine")
    ks = sorted({int(k) for k in config.get("retrieval", {}).get("ks", [1, 5, 10])})
    query_values = _prepare_vectors(query_embeddings.to(device), metric)
    gallery_values = _prepare_vectors(clean_gallery_embeddings.to(device), metric)
    partitions: dict[str, list[int]] = defaultdict(list)
    for index, partition_id in enumerate(partition_ids):
        partitions[str(partition_id)].append(index)
    query_seed = int(config.get("retrieval", {}).get("query_seed", 0))
    ppm_tolerance = float(config.get("mass_controlled_retrieval", {}).get("ppm_tolerance", 10.0))
    partition_results = {}
    for partition_id, indices in sorted(partitions.items()):
        part_peptides = [peptide_ids[index] for index in indices]
        groups = _group_indices(part_peptides)
        if not groups:
            continue
        part_query = query_values[indices]
        part_gallery = gallery_values[indices]
        part_mz = precursor_mz[indices].to(device, dtype=torch.float32)
        part_charges = precursor_charges[indices].to(device)
        broad = _cross_view_retrieval_metrics(
            part_query, part_gallery, part_peptides, groups,
            metric=metric, ks=ks, query_seed=query_seed,
        )
        def strict_mask(query_index: int) -> torch.Tensor:
            ppm = (part_mz - part_mz[query_index]).abs().div(part_mz[query_index]).mul(1e6)
            return part_charges.eq(part_charges[query_index]) & ppm.le(ppm_tolerance)
        strict = _cross_view_retrieval_metrics(
            part_query, part_gallery, part_peptides, groups,
            metric=metric, ks=ks, query_seed=query_seed, candidate_mask_factory=strict_mask,
        )
        partition_results[partition_id] = {
            **{f"robust_{name}": value for name, value in broad.items()},
            **{f"robust_mass_controlled_{name.removeprefix('cross_view_')}": value for name, value in strict.items()},
        }
    if not partition_results:
        raise ValueError("No partition contains repeated peptide groups for robustness evaluation.")
    metric_names = sorted(
        {
            name
            for result in partition_results.values()
            for name, value in result.items()
            if isinstance(value, float) and math.isfinite(value)
        }
    )
    macro = {
        name: sum(result[name] for result in partition_results.values() if name in result)
        / sum(name in result for result in partition_results.values())
        for name in metric_names
    }
    return {"partitions": partition_results, "macro": macro}


def _compactness_metrics(
    values: torch.Tensor,
    groups: dict[str, list[int]],
    *,
    metric: str,
    seed: int,
    max_between_samples: int,
    show_progress: bool,
    progress_desc: str,
) -> dict[str, float]:
    if max_between_samples < 1:
        raise ValueError("compactness.max_between_samples must be positive.")
    all_indices = torch.arange(values.shape[0], device=values.device)
    radii = []
    pairwise_distances = []
    distance_ratios = []
    for peptide_id, indices in tqdm(
        groups.items(),
        total=len(groups),
        desc=progress_desc,
        unit="peptide",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        group_indices = torch.tensor(indices, dtype=torch.long, device=values.device)
        group_values = values[group_indices]
        radii.append(torch.linalg.vector_norm(group_values - group_values.mean(dim=0), dim=1).mean())
        within_distance = _mean_pairwise_distance(group_values, metric)
        pairwise_distances.append(within_distance)

        non_group_count = values.shape[0] - group_indices.numel()
        if non_group_count == 0:
            continue
        if non_group_count <= max_between_samples:
            non_group_indices = all_indices[~torch.isin(all_indices, group_indices)]
        else:
            # Draw directly from the complement's rank space. Constructing a
            # full non-group tensor and randperm for every peptide made this
            # diagnostic O(groups * spectra), despite using only 1,000 rows.
            generator = torch.Generator(device=values.device)
            generator.manual_seed(_stable_index(seed, 2**31 - 1, peptide_id))
            selected_ranks = torch.empty(0, dtype=torch.long, device=values.device)
            while selected_ranks.numel() < max_between_samples:
                draws = torch.randint(
                    non_group_count,
                    (max_between_samples * 2,),
                    generator=generator,
                    device=values.device,
                )
                draws = torch.cat((selected_ranks, draws))
                unique_draws, inverse = torch.unique(draws, return_inverse=True)
                positions = torch.arange(draws.numel(), device=values.device)
                first_positions = torch.full(
                    (unique_draws.numel(),),
                    draws.numel(),
                    dtype=torch.long,
                    device=values.device,
                )
                first_positions.scatter_reduce_(
                    0, inverse, positions, reduce="amin", include_self=True
                )
                # Keep first occurrence order: taking sorted unique values would
                # bias every sampled complement toward low spectrum indices.
                selected_ranks = unique_draws[first_positions.argsort()]
            selected_ranks = selected_ranks[:max_between_samples]
            sorted_group_indices = group_indices.sort().values
            offsets = torch.searchsorted(
                sorted_group_indices - torch.arange(
                    sorted_group_indices.numel(), device=values.device
                ),
                selected_ranks,
                right=True,
            )
            non_group_indices = selected_ranks + offsets
        between_distance = _distance_matrix(
            group_values, values[non_group_indices], metric
        ).mean()
        if between_distance > 0:
            distance_ratios.append(within_distance / between_distance)

    if not radii:
        raise ValueError("Compactness requires at least one repeated peptide group.")
    distance_ratio = (
        float(torch.stack(distance_ratios).mean().item())
        if distance_ratios
        else float("nan")
    )
    return {
        "centroid_radius": float(torch.stack(radii).mean().item()),
        "mean_pairwise_distance": float(torch.stack(pairwise_distances).mean().item()),
        "distance_ratio": distance_ratio,
        "n_groups": float(len(groups)),
    }


def _silhouette_score(
    values: torch.Tensor,
    peptide_ids: list[str],
    *,
    metric: str,
    max_samples: int,
    seed: int,
) -> float:
    """Optional, deterministic silhouette computation kept out of the hot path."""
    from sklearn.metrics import silhouette_score

    groups = _group_indices(peptide_ids)
    valid_indices = [index for indices in groups.values() for index in indices]
    if len(groups) < 2 or len(valid_indices) < 3:
        return float("nan")
    if len(valid_indices) > max_samples:
        ordered = sorted(
            valid_indices,
            key=lambda index: _stable_index(seed, 2**31 - 1, index),
        )[:max_samples]
    else:
        ordered = valid_indices
    labels = [peptide_ids[index] for index in ordered]
    return float(
        silhouette_score(values[ordered].cpu().numpy(), labels, metric=metric)
    )


def evaluate_peptide_embeddings(
    embeddings: torch.Tensor,
    peptide_ids: list[str],
    partition_ids: list[str],
    config: dict,
    precursor_mz: torch.Tensor | None = None,
    precursor_charges: torch.Tensor | None = None,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Compute retrieval and compactness metrics separately for each partition."""
    if embeddings.shape[0] != len(peptide_ids) or len(peptide_ids) != len(partition_ids):
        raise ValueError("Embeddings, peptide_ids, and partition_ids must have equal length.")
    retrieval_cfg = config.get("retrieval", {})
    compactness_cfg = config.get("compactness", {})
    metric = retrieval_cfg.get("metric", "cosine")
    ks = sorted({int(k) for k in retrieval_cfg.get("ks", [1, 5, 10])})
    if not ks or min(ks) < 1:
        raise ValueError("retrieval.ks must contain positive integers.")

    embeddings = _prepare_vectors(embeddings.to(device), metric)
    partitions: dict[str, list[int]] = defaultdict(list)
    for index, partition_id in enumerate(partition_ids):
        partitions[str(partition_id)].append(index)

    partition_results: dict[str, dict[str, float]] = {}
    for partition_id, indices in sorted(partitions.items()):
        part_values = embeddings[indices]
        part_peptides = [peptide_ids[index] for index in indices]
        groups = _group_indices(part_peptides)
        if not groups:
            continue
        result = {
            "n_spectra": float(len(indices)),
            "n_groups": float(len(groups)),
        }
        if retrieval_cfg.get("enabled", True):
            result.update(
                _retrieval_metrics(
                    part_values,
                    part_peptides,
                    groups,
                    metric=metric,
                    ks=ks,
                    query_seed=int(retrieval_cfg.get("query_seed", 0)),
                    query_batch_size=int(retrieval_cfg.get("query_batch_size", 128)),
                    show_progress=bool(retrieval_cfg.get("show_progress", False)),
                    progress_desc=f"Retrieval {partition_id}",
                )
            )
        mass_controlled_cfg = config.get("mass_controlled_retrieval", {})
        if mass_controlled_cfg.get("enabled", False):
            if precursor_mz is None or precursor_charges is None:
                raise ValueError("Mass-controlled retrieval requires precursor metadata.")
            result.update(
                _mass_controlled_retrieval_metrics(
                    part_values,
                    part_peptides,
                    groups,
                    precursor_mz[indices],
                    precursor_charges[indices],
                    metric=metric,
                    ks=ks,
                    query_seed=int(retrieval_cfg.get("query_seed", 0)),
                    ppm_tolerance=float(mass_controlled_cfg.get("ppm_tolerance", 10.0)),
                    show_progress=bool(retrieval_cfg.get("show_progress", False)),
                    progress_desc=f"Strict retrieval {partition_id}",
                )
            )
        cross_charge_cfg = config.get("cross_charge_retrieval", {})
        if cross_charge_cfg.get("enabled", False):
            if precursor_mz is None or precursor_charges is None:
                raise ValueError("Cross-charge retrieval requires precursor metadata.")
            cross_charge_common = {
                "metric": metric,
                "ks": ks,
                "query_seed": int(retrieval_cfg.get("query_seed", 0)),
            }
            cross_charge_inputs = (
                part_values,
                part_peptides,
                groups,
                precursor_mz[indices],
                precursor_charges[indices],
            )
            # Broad cross-charge retrieval measures whether a peptide remains
            # nearby across charge states. It can still be precursor-solvable,
            # so report the strict neutral-mass-matched counterpart separately.
            result.update(
                _cross_charge_retrieval_metrics(
                    *cross_charge_inputs,
                    **cross_charge_common,
                    neutral_mass_ppm_tolerance=None,
                    query_batch_size=int(retrieval_cfg.get("query_batch_size", 128)),
                    show_progress=bool(retrieval_cfg.get("show_progress", False)),
                    progress_desc=f"Cross-charge retrieval {partition_id}",
                )
            )
            strict_tolerance = cross_charge_cfg.get("neutral_mass_ppm_tolerance")
            if strict_tolerance is not None:
                strict = _cross_charge_retrieval_metrics(
                    *cross_charge_inputs,
                    **cross_charge_common,
                    neutral_mass_ppm_tolerance=float(strict_tolerance),
                    query_batch_size=int(retrieval_cfg.get("query_batch_size", 128)),
                    show_progress=bool(retrieval_cfg.get("show_progress", False)),
                    progress_desc=f"Strict cross-charge retrieval {partition_id}",
                )
                result.update(
                    {
                        f"cross_charge_mass_controlled_{name.removeprefix('cross_charge_')}": value
                        for name, value in strict.items()
                    }
                )
        if compactness_cfg.get("enabled", True):
            result.update(
                _compactness_metrics(
                    part_values,
                    groups,
                    metric=metric,
                    seed=int(compactness_cfg.get("seed", 0)),
                    max_between_samples=int(
                        compactness_cfg.get("max_between_samples", 1000)
                    ),
                    show_progress=bool(compactness_cfg.get("show_progress", False)),
                    progress_desc=f"Compactness {partition_id}",
                )
            )
            silhouette_cfg = compactness_cfg.get("silhouette", {})
            if silhouette_cfg.get("enabled", False):
                result["silhouette"] = _silhouette_score(
                    part_values,
                    part_peptides,
                    metric=metric,
                    max_samples=int(silhouette_cfg.get("max_samples", 3000)),
                    seed=int(silhouette_cfg.get("seed", 0)),
                )
        partition_results[partition_id] = result

    if not partition_results:
        raise ValueError("No evaluation partition contains a repeated peptide group.")
    metric_names = sorted(
        {
            metric_name
            for result in partition_results.values()
            for metric_name, value in result.items()
            if isinstance(value, float) and math.isfinite(value)
        }
    )
    macro = {
        metric_name: sum(result[metric_name] for result in partition_results.values() if metric_name in result)
        / sum(metric_name in result for result in partition_results.values())
        for metric_name in metric_names
    }
    return {"partitions": partition_results, "macro": macro}
