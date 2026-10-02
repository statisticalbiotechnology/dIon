import torch

from src.embed_eval.counterfactual import evaluate_null_local_source_consistency


def test_both_null_local_views_are_scored_against_the_same_clean_source():
    report = evaluate_null_local_source_consistency(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.9, 0.1]]),
        torch.tensor([[0.8, 0.2]]),
        ["test"],
    )["pooled"]

    assert report["local_view_1_source_accuracy"] == 1.0
    assert report["local_view_2_source_accuracy"] == 1.0
    assert report["paired_source_accuracy"] == 1.0
    assert report["local_view_1_margin"] > 0.0
    assert report["local_view_2_margin"] > 0.0
