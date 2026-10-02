import torch

from src.embed_eval.counterfactual import evaluate_counterfactual_source_selection


def _evaluate(values, metric):
    clean_a, clean_b, mixed_a, mixed_b = values
    return evaluate_counterfactual_source_selection(
        torch.tensor(clean_a),
        torch.tensor(clean_b),
        torch.tensor(mixed_a),
        torch.tensor(mixed_b),
        ["test"],
        distance_metric=metric,
    )["pooled"]


def test_euclidean_preserves_raw_norms_while_cosine_normalizes_them():
    values = (
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        [[2.0, 0.0]],
        [[0.0, 2.0]],
    )
    cosine = _evaluate(values, "cosine")
    euclidean = _evaluate(values, "euclidean")

    assert cosine["source_accuracy_anchor_condition"] == 1.0
    assert cosine["mixed_condition_cosine_distance"] == 1.0
    assert euclidean["source_accuracy_anchor_condition"] == 1.0
    assert euclidean["mixed_condition_euclidean_distance"] > 2.0


def test_jensen_shannon_uses_probability_distributions_and_selects_own_source():
    values = (
        [[0.99, 0.01]],
        [[0.01, 0.99]],
        [[0.90, 0.10]],
        [[0.10, 0.90]],
    )
    report = _evaluate(values, "jensen_shannon")

    assert report["paired_source_accuracy"] == 1.0
    assert report["mixed_condition_jensen_shannon_distance"] > 0.0
