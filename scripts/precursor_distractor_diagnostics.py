#!/usr/bin/env python3
"""Measure precursor-condition collisions for in-batch distractor sampling."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

import torch
import yaml
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTON_MASS_DA = 1.007276466621
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.collate_functions import pad_peaks
from src.data.lance_data_module import LanceDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Master training YAML.")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--tolerances-th",
        type=float,
        nargs="+",
        default=[],
        help="Same-charge precursor m/z exclusion radii in Th to evaluate.",
    )
    parser.add_argument(
        "--tolerances-ppm",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0, 10.0],
        help="Same-charge precursor m/z exclusion radii in ppm of pair midpoint.",
    )
    parser.add_argument(
        "--neutral-mass-tolerance-ppm",
        type=float,
        default=10.0,
        help="Neutral-mass exclusion radius across all charge states.",
    )
    parser.add_argument(
        "--distractor-peak-fraction",
        type=float,
        default=1.0,
        help="Fraction of a same-scale distractor crop included before merge/capping.",
    )
    parser.add_argument(
        "--mix-apply-to",
        choices=["global_only", "all"],
        default="global_only",
        help="Student crop groups that receive distractor peaks.",
    )
    parser.add_argument(
        "--simulate-mixed-global-crops",
        action="store_true",
        help="Measure actual 5-ppm component/post-mix merge outcomes for global crops.",
    )
    parser.add_argument(
        "--distractor-condition-separation-ppm",
        type=float,
        default=10.0,
        help="Same-charge precursor m/z separation used when simulating B choices.",
    )
    return parser.parse_args()


def make_loader(config: dict, batch_size: int, workers: int):
    collate_fn = partial(
        pad_peaks,
        max_peaks=int(config["max_peaks"]),
        precursor_mz_name="precursor_mz",
        precursor_mass_name=False,
        filter_method=config.get("peak_filter_method", "default"),
        intensity_scaling=config.get("intensity_scaling", "minmax"),
        min_mz=float(config.get("min_mz", 50.0)),
        max_mz=float(config.get("max_mz", 2500.0)),
        min_intensity=float(config.get("min_intensity", 0.01)),
        remove_precursor_tol=float(config.get("remove_precursor_tol", 2.0)),
        min_peaks=int(config.get("min_peaks", 0)),
    )
    data_module = LanceDataModule(
        config["data_root_dir"],
        batch_size=batch_size,
        collate_fn=collate_fn,
        seed=int(config.get("seed", 0)),
        include_test=False,
        num_workers=workers,
        pin_memory=False,
        expected_world_size=1,
    )
    data_module.setup("fit")
    return data_module.train_dataloader()


def load_pretrain_task_config(master_config: dict) -> dict:
    pretrain_path = Path(master_config["pretrain_config"])
    if not pretrain_path.is_absolute():
        pretrain_path = REPOSITORY_ROOT / pretrain_path
    with open(pretrain_path, "r", encoding="utf-8") as handle:
        pretrain_config = yaml.safe_load(handle)
    return pretrain_config[master_config["pretraining_task"]]


def _mib(num_float_values: int) -> float:
    return num_float_values * torch.tensor([], dtype=torch.float32).element_size() / 2**20


def merge_peaks_ppm(peaks: torch.Tensor, origins: torch.Tensor, ppm: float):
    """Merge sorted-or-unsorted peaks and preserve pure/mixed provenance."""
    if peaks.shape[0] <= 1:
        return peaks, origins

    order = torch.argsort(peaks[:, 0], stable=True)
    peaks = peaks[order]
    origins = origins[order]
    merged_peaks = [peaks[0].clone()]
    merged_origins = [origins[0].clone()]
    for peak, origin in zip(peaks[1:], origins[1:]):
        current = merged_peaks[-1]
        midpoint = 0.5 * (peak[0] + current[0])
        if (peak[0] - current[0]).item() < ppm * 1e-6 * midpoint.item():
            total_intensity = current[1] + peak[1]
            if total_intensity.item() > 0:
                current[0] = (
                    current[0] * current[1] + peak[0] * peak[1]
                ) / total_intensity
            else:
                current[0] = 0.5 * (current[0] + peak[0])
            current[1] = total_intensity
            if merged_origins[-1].item() != origin.item():
                merged_origins[-1] = torch.tensor(2, dtype=origins.dtype)
        else:
            merged_peaks.append(peak.clone())
            merged_origins.append(origin.clone())
    return torch.stack(merged_peaks), torch.stack(merged_origins)


def random_subset(peaks: torch.Tensor, origins: torch.Tensor, fraction: float):
    n_select = max(1, int(peaks.shape[0] * fraction))
    if n_select >= peaks.shape[0]:
        return peaks, origins
    indices = torch.randperm(peaks.shape[0])[:n_select]
    return peaks[indices], origins[indices]


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    batch_size = args.batch_size or int(config["batch_size"])
    if batch_size < 2:
        raise ValueError("batch-size must be at least 2 for distractor sampling.")
    if args.batches < 1:
        raise ValueError("batches must be positive.")
    if not 0.0 <= args.distractor_peak_fraction <= 1.0:
        raise ValueError("distractor-peak-fraction must be in [0, 1].")

    pretrain_task = load_pretrain_task_config(config)
    loader = make_loader(config, batch_size, args.workers)
    thresholds = [(f"{tolerance:g} Th", lambda _, delta=tolerance: delta) for tolerance in args.tolerances_th]
    thresholds += [
        (
            f"{tolerance:g} ppm",
            lambda midpoint, ppm=tolerance: ppm * 1e-6 * midpoint,
        )
        for tolerance in args.tolerances_ppm
    ]
    totals = {
        label: {
            "threshold": threshold,
            "pairs": 0,
            "condition_rejected": 0,
            "neutral_rejected": 0,
            "rejected": 0,
            "anchors": 0,
            "no_candidate": 0,
        }
        for label, threshold in thresholds
    }
    observed_batches = 0
    memory_stats = []
    mix_simulation = {
        "anchor_selected": 0,
        "distractor_selected": 0,
        "post_merge": 0,
        "pure_anchor": 0,
        "pure_distractor": 0,
        "mixed": 0,
        "global_group_lengths": [],
    }

    progress = tqdm(
        loader,
        total=args.batches,
        desc="Scanning precursor pairs",
        unit="batch",
        dynamic_ncols=True,
    )
    for batch in progress:
        precursor_mz = batch.get("precursor_mz")
        charge = batch.get("precursor_charge")
        if precursor_mz is None or charge is None:
            raise KeyError("Batch must contain precursor_mz and precursor_charge.")

        precursor_mz = precursor_mz.reshape(-1).float()
        charge = charge.reshape(-1)
        valid = torch.isfinite(precursor_mz) & torch.isfinite(charge.float())
        if not bool(valid.all()):
            raise ValueError("Found missing or non-finite precursor m/z or charge.")

        same_charge = charge[:, None] == charge[None, :]
        mz_delta = (precursor_mz[:, None] - precursor_mz[None, :]).abs()
        mz_midpoint = 0.5 * (precursor_mz[:, None] + precursor_mz[None, :])
        neutral_mass = charge.float() * (precursor_mz - PROTON_MASS_DA)
        neutral_delta = (neutral_mass[:, None] - neutral_mass[None, :]).abs()
        neutral_midpoint = 0.5 * (neutral_mass[:, None] + neutral_mass[None, :])
        nonself = ~torch.eye(precursor_mz.numel(), dtype=torch.bool)
        neutral_rejected = nonself & (
            neutral_delta
            < args.neutral_mass_tolerance_ppm * 1e-6 * neutral_midpoint
        )
        simulated_condition_rejected = nonself & same_charge & (
            mz_delta
            < args.distractor_condition_separation_ppm * 1e-6 * mz_midpoint
        )
        eligible_for_simulation = nonself & ~(
            simulated_condition_rejected | neutral_rejected
        )

        lengths = batch["peak_lengths"].reshape(-1).long()
        global_scale = float(pretrain_task["global_crops_scale"][1])
        local_scale = float(pretrain_task["local_crops_scale"][1])
        num_global = int(pretrain_task["num_global_crops"])
        num_local = int(pretrain_task["num_local_crops"])
        distractor_lengths = lengths.roll(1)

        def padded_crop_length(scale: float, mixed: bool) -> int:
            anchor_lengths = (lengths.float() * scale).long().clamp(min=1)
            if not mixed:
                return int(anchor_lengths.max())
            distractor_crop_lengths = (
                anchor_lengths.float() * args.distractor_peak_fraction
            ).long()
            distractor_crop_lengths = torch.minimum(
                distractor_crop_lengths, distractor_lengths
            )
            return int((anchor_lengths + distractor_crop_lengths).max())

        global_length = padded_crop_length(global_scale, mixed=False)
        local_length = padded_crop_length(local_scale, mixed=False)
        mixed_global_length = padded_crop_length(global_scale, mixed=True)
        mixed_local_length = (
            padded_crop_length(local_scale, mixed=True)
            if args.mix_apply_to == "all"
            else local_length
        )
        baseline_student_tokens = batch_size * (
            num_global * global_length + num_local * local_length
        )
        mixed_student_tokens = batch_size * (
            num_global * mixed_global_length + num_local * mixed_local_length
        )
        teacher_tokens = batch_size * num_global * global_length
        baseline_attention_proxy = batch_size * (
            num_global * global_length**2 + num_local * local_length**2
        )
        mixed_attention_proxy = batch_size * (
            num_global * mixed_global_length**2 + num_local * mixed_local_length**2
        )
        teacher_attention_proxy = batch_size * num_global * global_length**2
        memory_stats.append(
            {
                "baseline_student_mib": _mib(baseline_student_tokens * 2),
                "mixed_student_mib": _mib(mixed_student_tokens * 2),
                "baseline_total_mib": _mib((baseline_student_tokens + teacher_tokens) * 2),
                "mixed_total_mib": _mib((mixed_student_tokens + teacher_tokens) * 2),
                "baseline_attention": baseline_attention_proxy + teacher_attention_proxy,
                "mixed_attention": mixed_attention_proxy + teacher_attention_proxy,
                "global_length": global_length,
                "mixed_global_length": mixed_global_length,
                "local_length": local_length,
                "mixed_local_length": mixed_local_length,
            }
        )

        if args.simulate_mixed_global_crops:
            premerged_components = []
            for sample_idx, sample_length in enumerate(lengths.tolist()):
                component = batch["mz_array"].new_zeros((sample_length, 2))
                component[:, 0] = batch["mz_array"][sample_idx, :sample_length]
                component[:, 1] = batch["intensity_array"][sample_idx, :sample_length]
                component, _ = merge_peaks_ppm(
                    component,
                    torch.zeros(sample_length, dtype=torch.int8),
                    ppm=5.0,
                )
                premerged_components.append(component)

            global_group_length = 0
            for _ in range(num_global):
                view_length = 0
                for anchor_idx, anchor_component in enumerate(premerged_components):
                    candidates = torch.nonzero(
                        eligible_for_simulation[anchor_idx], as_tuple=False
                    ).reshape(-1)
                    distractor_idx = candidates[torch.randint(candidates.numel(), (1,)).item()].item()
                    distractor_component = premerged_components[distractor_idx]
                    anchor_origins = torch.zeros(anchor_component.shape[0], dtype=torch.int8)
                    distractor_origins = torch.ones(distractor_component.shape[0], dtype=torch.int8)
                    anchor_peaks, anchor_origins = random_subset(
                        anchor_component, anchor_origins, global_scale
                    )
                    distractor_count = min(
                        int(anchor_peaks.shape[0] * args.distractor_peak_fraction),
                        distractor_component.shape[0],
                    )
                    distractor_peaks, distractor_origins = random_subset(
                        distractor_component,
                        distractor_origins,
                        distractor_count / max(distractor_component.shape[0], 1),
                    )
                    mixed_peaks, mixed_origins = merge_peaks_ppm(
                        torch.cat([anchor_peaks, distractor_peaks]),
                        torch.cat([anchor_origins, distractor_origins]),
                        ppm=5.0,
                    )
                    mix_simulation["anchor_selected"] += anchor_peaks.shape[0]
                    mix_simulation["distractor_selected"] += distractor_peaks.shape[0]
                    mix_simulation["post_merge"] += mixed_peaks.shape[0]
                    mix_simulation["pure_anchor"] += int((mixed_origins == 0).sum())
                    mix_simulation["pure_distractor"] += int((mixed_origins == 1).sum())
                    mix_simulation["mixed"] += int((mixed_origins == 2).sum())
                    view_length = max(view_length, mixed_peaks.shape[0])
                global_group_length = max(global_group_length, view_length)
            mix_simulation["global_group_lengths"].append(global_group_length)

        for total in totals.values():
            condition_rejected = nonself & same_charge & (
                mz_delta < total["threshold"](mz_midpoint)
            )
            rejected = condition_rejected | neutral_rejected
            eligible = nonself & ~rejected
            total["pairs"] += int(nonself.sum())
            total["condition_rejected"] += int(condition_rejected.sum())
            total["neutral_rejected"] += int(neutral_rejected.sum())
            total["rejected"] += int(rejected.sum())
            total["anchors"] += precursor_mz.numel()
            total["no_candidate"] += int((~eligible.any(dim=1)).sum())

        observed_batches += 1
        progress.set_postfix(anchors=f"{observed_batches * batch_size:,}")
        if observed_batches >= args.batches:
            break
    progress.close()

    print(
        f"Checked {observed_batches} train batches of {batch_size} spectra "
        f"({observed_batches * batch_size:,} anchors)."
    )
    print(
        "Candidate rejection: same-charge precursor m/z exclusion plus "
        f"neutral-mass exclusion at {args.neutral_mass_tolerance_ppm:g} ppm."
    )
    for label, total in totals.items():
        rejected_fraction = total["rejected"] / total["pairs"]
        eligible_fraction = 1.0 - rejected_fraction
        expected_draws = 1.0 / eligible_fraction if eligible_fraction else float("inf")
        print(
            f"  m/z < {label}: condition {total['condition_rejected'] / total['pairs']:7.4%}; "
            f"neutral {total['neutral_rejected'] / total['pairs']:7.4%}; "
            f"combined {rejected_fraction:7.4%}; "
            f"eligible {eligible_fraction:7.4%}; "
            f"expected draws {expected_draws:.4f}; "
            f"anchors with no eligible B {total['no_candidate']}/{total['anchors']}"
        )

    def mean_stat(key: str) -> float:
        return sum(stat[key] for stat in memory_stats) / len(memory_stats)

    print(f"\nPre-merge, uncapped {args.mix_apply_to} mixed-crop memory estimate:")
    print(
        f"  mean padded global length {mean_stat('global_length'):.1f} -> "
        f"{mean_stat('mixed_global_length'):.1f}; local "
        f"{mean_stat('local_length'):.1f} -> {mean_stat('mixed_local_length'):.1f}"
    )
    print(
        f"  student peak tensors {mean_stat('baseline_student_mib'):.2f} MiB -> "
        f"{mean_stat('mixed_student_mib'):.2f} MiB "
        f"({mean_stat('mixed_student_mib') / mean_stat('baseline_student_mib'):.2f}x)"
    )
    print(
        f"  student + teacher peak tensors {mean_stat('baseline_total_mib'):.2f} MiB -> "
        f"{mean_stat('mixed_total_mib'):.2f} MiB "
        f"({mean_stat('mixed_total_mib') / mean_stat('baseline_total_mib'):.2f}x)"
    )
    print(
        f"  attention-length-squared proxy, including clean teacher: "
        f"{mean_stat('mixed_attention') / mean_stat('baseline_attention'):.2f}x"
    )

    if args.simulate_mixed_global_crops:
        total_post_merge = mix_simulation["post_merge"]
        simulated_global_length = sum(mix_simulation["global_group_lengths"]) / len(
            mix_simulation["global_group_lengths"]
        )
        simulated_student_tokens = batch_size * (
            num_global * simulated_global_length + num_local * mean_stat("local_length")
        )
        simulated_total_tokens = simulated_student_tokens + batch_size * num_global * mean_stat(
            "global_length"
        )
        print("\nSimulated per-view global distractors with 5-ppm pre/post merging:")
        print(
            f"  mean shared global padded length after merge: {simulated_global_length:.1f}"
        )
        print(
            f"  selected peaks A={mix_simulation['anchor_selected'] / (observed_batches * batch_size * num_global):.1f}, "
            f"B={mix_simulation['distractor_selected'] / (observed_batches * batch_size * num_global):.1f}, "
            f"post-merge={total_post_merge / (observed_batches * batch_size * num_global):.1f}"
        )
        print(
            f"  provenance after merge: anchor={mix_simulation['pure_anchor'] / total_post_merge:.2%}, "
            f"distractor={mix_simulation['pure_distractor'] / total_post_merge:.2%}, "
            f"mixed={mix_simulation['mixed'] / total_post_merge:.2%}"
        )
        print(
            f"  student + teacher peak tensors using simulated global padding: "
            f"{_mib(simulated_total_tokens * 2):.2f} MiB"
        )


if __name__ == "__main__":
    main()
