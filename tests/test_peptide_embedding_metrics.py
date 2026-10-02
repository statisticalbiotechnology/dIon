import torch

from src.embed_eval.peptide_metrics import evaluate_peptide_embeddings


def _config():
    return {
        "retrieval": {
            "metric": "cosine",
            "ks": [1, 2],
            "query_seed": 0,
            "query_batch_size": 2,
        },
        "compactness": {"max_between_samples": 10},
    }


def test_retrieval_excludes_query_from_average_precision():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    )
    report = evaluate_peptide_embeddings(
        embeddings,
        ["a", "a", "b", "b"],
        ["species"] * 4,
        _config(),
        device=torch.device("cpu"),
    )
    metrics = report["partitions"]["species"]
    assert metrics["r_at_1"] == 1.0
    assert metrics["map"] == 1.0
    assert metrics["mrr"] == 1.0


def test_collapsed_embedding_does_not_crash_distance_ratio():
    embeddings = torch.zeros((4, 2))
    config = _config()
    config["retrieval"]["metric"] = "euclidean"
    report = evaluate_peptide_embeddings(
        embeddings,
        ["a", "a", "b", "b"],
        ["species"] * 4,
        config,
        device=torch.device("cpu"),
    )
    metrics = report["partitions"]["species"]
    assert torch.isnan(torch.tensor(metrics["distance_ratio"]))


def test_cross_view_retrieval_excludes_matching_clean_row():
    from src.embed_eval.peptide_metrics import evaluate_clean_gallery_robustness

    clean = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    perturbed = torch.tensor([[0.95, 0.05], [0.85, 0.15], [0.05, 0.95], [0.15, 0.85]])
    report = evaluate_clean_gallery_robustness(
        perturbed,
        clean,
        ["p1", "p1", "p2", "p2"],
        ["a"] * 4,
        torch.tensor([500.0, 500.0, 500.001, 500.001]),
        torch.tensor([2, 2, 2, 2]),
        {
            "retrieval": {"metric": "cosine", "ks": [1]},
            "mass_controlled_retrieval": {"ppm_tolerance": 10.0},
        },
        device=torch.device("cpu"),
    )
    assert report["partitions"]["a"]["robust_cross_view_hit_at_1"] == 1.0
