import torch

from src.embed_eval.dense_counterfactual import evaluate_dense_precursor_source_selection


def test_dense_metric_requires_source_aligned_query_change():
    report = evaluate_dense_precursor_source_selection(
        torch.tensor([0.4, 0.2]),
        torch.tensor([0.3, 0.1]),
        torch.tensor([0.5, 0.4]),
        torch.tensor([0.6, 0.3]),
        torch.tensor([3, 2]),
        torch.tensor([4, 1]),
        ["a", "b"],
    )
    assert report["pooled"]["paired_query_alignment"] == 1.0
    assert report["pooled"]["matched_anchor_peaks"] == 5
    assert report["pooled"]["matched_distractor_peaks"] == 5
    assert abs(report["macro"]["mean_query_advantage"] - 0.25) < 1e-6

def test_dense_metric_rejects_missing_source_peak_rows():
    try:
        evaluate_dense_precursor_source_selection(
            torch.tensor([0.1]), torch.tensor([0.1]),
            torch.tensor([0.1]), torch.tensor([0.1]),
            torch.tensor([0]), torch.tensor([1]), ["a"],
        )
    except ValueError as exc:
        assert "A-only and B-only" in str(exc)
    else:
        raise AssertionError("Expected missing source peaks to fail.")


def test_target_swapped_mixer_returns_eligible_partner_and_mixed_rows():
    from src.distractor_augmentation import BatchedStudentDistractorMixAugmentation

    torch.manual_seed(4)
    mixer = BatchedStudentDistractorMixAugmentation(
        condition_separation_ppm=10.0,
        neutral_mass_separation_ppm=10.0,
        merge_ppm=0.0,
    )
    spectra = torch.tensor(
        [
            [[100.0, 1.0], [200.0, 0.4], [0.0, 0.0]],
            [[300.0, 0.7], [400.0, 0.3], [0.0, 0.0]],
        ]
    )
    anchor_peaks = spectra.clone()
    anchor_padding = torch.tensor([[False, False, True], [False, False, True]])
    mixed_peaks, mixed_padding, provenance, partners, eligible = (
        mixer.mix_sampled_distractors(
            anchor_peaks,
            anchor_padding,
            spectra,
            torch.tensor([2, 2]),
            torch.tensor([500.0, 700.0]),
            torch.tensor([2, 2]),
            strength=1.0,
            return_provenance=True,
        )
    )
    assert eligible.tolist() == [True, True]
    assert partners.tolist() == [1, 0]
    assert mixed_peaks.shape[0] == mixed_padding.shape[0] == provenance.shape[0] == 2
    assert (~mixed_padding).sum(dim=1).tolist() == [4, 4]
