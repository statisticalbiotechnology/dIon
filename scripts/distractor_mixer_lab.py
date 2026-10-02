#!/usr/bin/env python3
"""Correctness checks and microbenchmark for the batched distractor mixer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_augmentation import BatchedRandomSelectionAugmentation
from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_eligibility(device):
    mixer = BatchedStudentDistractorMixAugmentation()
    proton = mixer.PROTON_MASS_DA
    neutral_mass = 997.985
    precursor_mz = torch.tensor(
        [
            500.0,
            500.003,
            neutral_mass / 3 + proton,
            800.0,
        ],
        device=device,
    )
    charge = torch.tensor([2, 2, 3, 2], device=device)
    eligible = mixer.eligible_distractors(precursor_mz, charge)
    assert not eligible[0, 0]
    assert not eligible[0, 1], "same-charge precursor collision was not rejected"
    assert not eligible[0, 2], "cross-charge neutral-mass collision was not rejected"
    assert eligible[0, 3]


def check_merge(device):
    mixer = BatchedStudentDistractorMixAugmentation(merge_ppm=5.0)
    peaks = torch.tensor(
        [[[100.0, 1.0], [100.0004, 3.0], [200.0, 2.0]]],
        device=device,
    )
    padding = torch.zeros((1, 3), dtype=torch.bool, device=device)
    provenance = torch.tensor(
        [[mixer.ANCHOR, mixer.DISTRACTOR, mixer.ANCHOR]],
        dtype=torch.int8,
        device=device,
    )
    merged, merged_padding, merged_provenance = mixer.merge_batched(
        peaks, padding, provenance
    )
    assert int((~merged_padding).sum()) == 2
    torch.testing.assert_close(merged[0, 0, 1], torch.tensor(4.0, device=device))
    torch.testing.assert_close(
        merged[0, 0, 0], torch.tensor(100.0003, device=device), atol=1e-5, rtol=0
    )
    assert int(merged_provenance[0, 0]) == mixer.MIXED
    assert int(merged_provenance[0, 1]) == mixer.ANCHOR


def reference_merge(peaks, origins, ppm):
    order = torch.argsort(peaks[:, 0], stable=True)
    peaks = peaks[order]
    origins = origins[order]
    groups = [[peaks[0].clone(), {int(origins[0])}]]
    previous_mz = peaks[0, 0]
    for peak, origin in zip(peaks[1:], origins[1:]):
        threshold = ppm * 1e-6 * 0.5 * (peak[0] + previous_mz)
        if peak[0] - previous_mz < threshold:
            current, current_origins = groups[-1]
            total_intensity = current[1] + peak[1]
            if total_intensity > 0:
                current[0] = (
                    current[0] * current[1] + peak[0] * peak[1]
                ) / total_intensity
            else:
                current[0] = 0.5 * (current[0] + peak[0])
            current[1] = total_intensity
            current_origins.add(int(origin))
        else:
            groups.append([peak.clone(), {int(origin)}])
        previous_mz = peak[0]
    merged_peaks = torch.stack([group[0] for group in groups])
    merged_origins = torch.tensor(
        [2 if len(group[1]) > 1 else next(iter(group[1])) for group in groups],
        dtype=torch.int8,
        device=peaks.device,
    )
    return merged_peaks, merged_origins


def check_merge_against_reference(device):
    mixer = BatchedStudentDistractorMixAugmentation(merge_ppm=5.0)
    base = torch.linspace(100, 1000, 32, device=device)
    peaks = torch.stack(
        [
            torch.cat([base, base[:8] + base[:8] * 2e-6]),
            torch.rand(40, device=device),
        ],
        dim=-1,
    )
    origins = torch.cat(
        [
            torch.zeros(32, dtype=torch.int8, device=device),
            torch.ones(8, dtype=torch.int8, device=device),
        ]
    )
    reference_peaks, reference_origins = reference_merge(peaks, origins, 5.0)
    batched_peaks, batched_padding, batched_origins = mixer.merge_batched(
        peaks.unsqueeze(0),
        torch.zeros((1, peaks.shape[0]), dtype=torch.bool, device=device),
        origins.unsqueeze(0),
    )
    count = int((~batched_padding[0]).sum())
    torch.testing.assert_close(
        batched_peaks[0, :count], reference_peaks, atol=1e-5, rtol=1e-6
    )
    assert torch.equal(batched_origins[0, :count], reference_origins)


def synthetic_batch(batch_size, seq_len, device):
    mz = torch.sort(
        50 + 1950 * torch.rand((batch_size, seq_len), device=device), dim=1
    ).values
    intensity = torch.rand((batch_size, seq_len), device=device)
    spectra = torch.stack([mz, intensity], dim=-1)
    lengths = torch.full((batch_size, 1), seq_len, dtype=torch.long, device=device)
    precursor_mz = torch.linspace(350, 1600, batch_size, device=device)
    charge = 2 + torch.arange(batch_size, device=device) % 3
    return spectra, lengths, precursor_mz, charge


def make_crops(spectra, lengths):
    augmentation = BatchedRandomSelectionAugmentation(
        global_crops_scale=(0.95, 0.95),
        local_crops_scale=(0.2, 0.2),
        num_global_crops=2,
        num_local_crops=5,
    )
    return augmentation(spectra, lengths)


def run_mixer(mixer, crops, spectra, lengths, precursor_mz, charge):
    return mixer(
        crops,
        spectra=spectra,
        lengths=lengths,
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        strength=1.0,
        num_global_crops=2,
        return_provenance=True,
    )


def check_global_only(device):
    spectra, lengths, precursor_mz, charge = synthetic_batch(8, 40, device)
    crops = make_crops(spectra, lengths)
    clean_crops = [(peaks.clone(), mask.clone()) for peaks, mask in crops]
    mixer = BatchedStudentDistractorMixAugmentation(mix_apply_to="global_only")
    mixed_crops, provenance = run_mixer(
        mixer, crops, spectra, lengths, precursor_mz, charge
    )

    for crop_idx in range(2, len(crops)):
        torch.testing.assert_close(mixed_crops[crop_idx][0], clean_crops[crop_idx][0])
        assert torch.equal(mixed_crops[crop_idx][1], clean_crops[crop_idx][1])
        assert torch.all(provenance[crop_idx][~clean_crops[crop_idx][1]] == mixer.ANCHOR)
    for crop_idx in range(2):
        assert mixed_crops[crop_idx][0].shape[1] > clean_crops[crop_idx][0].shape[1]
        valid_provenance = provenance[crop_idx][~mixed_crops[crop_idx][1]]
        assert torch.any(valid_provenance == mixer.ANCHOR)
        assert torch.any(valid_provenance == mixer.DISTRACTOR)
    for original, clean in zip(crops, clean_crops):
        torch.testing.assert_close(original[0], clean[0])
        assert torch.equal(original[1], clean[1])


def check_strength_uses_anchor_crop_count(device):
    mixer = BatchedStudentDistractorMixAugmentation(mix_apply_to="all")
    spectra = torch.zeros((2, 100, 2), device=device)
    spectra[0, :40, 0] = 100 + torch.arange(40, device=device)
    spectra[1, :, 0] = 1000 + torch.arange(100, device=device)
    spectra[..., 1] = 1.0
    lengths = torch.tensor([40, 100], device=device)
    anchor_peaks = spectra[:, :40].clone()
    anchor_padding = torch.zeros((2, 40), dtype=torch.bool, device=device)

    mixed, padding, provenance = mixer._mix_view(
        (anchor_peaks, anchor_padding),
        spectra,
        lengths,
        distractor_indices=torch.tensor([1, 0], device=device),
        has_eligible=torch.ones(2, dtype=torch.bool, device=device),
        strength=0.5,
    )
    assert torch.equal((~padding).sum(dim=1), torch.tensor([60, 60], device=device))
    assert torch.equal(
        (provenance == mixer.DISTRACTOR).sum(dim=1),
        torch.tensor([20, 20], device=device),
    )


def benchmark(args, device):
    spectra, lengths, precursor_mz, charge = synthetic_batch(
        args.batch_size, args.seq_len, device
    )
    crops = make_crops(spectra, lengths)
    mixer = BatchedStudentDistractorMixAugmentation(mix_apply_to="global_only")

    for _ in range(args.warmup):
        run_mixer(mixer, crops, spectra, lengths, precursor_mz, charge)
    synchronize(device)

    timings = []
    for _ in range(args.iters):
        start = time.perf_counter()
        run_mixer(mixer, crops, spectra, lengths, precursor_mz, charge)
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    values = torch.tensor(timings)
    print(
        f"batched global-only mixer: {values.mean():.3f} ms +/- "
        f"{values.std(unbiased=False):.3f} (batch={args.batch_size}, seq={args.seq_len})"
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(0)
    check_eligibility(device)
    check_merge(device)
    check_merge_against_reference(device)
    check_global_only(device)
    check_strength_uses_anchor_crop_count(device)
    print("correctness checks: passed")
    benchmark(args, device)


if __name__ == "__main__":
    main()
