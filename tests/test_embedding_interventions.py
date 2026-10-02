"""Focused checks for test-time embedding input interventions."""

from types import SimpleNamespace

import pyarrow as pa
import torch

from src.embed_eval.interventions import InputIntervention
from src.embed_eval.peptide_metrics import evaluate_peptide_embeddings


def test_mass_permutation_stays_within_species_charge_strata():
    dataset = SimpleNamespace(
        table=pa.table(
            {
                "precursor_mz": [500.0, 600.0, 700.0, 800.0],
                "precursor_charge": [2, 2, 3, 3],
                "species": ["a", "a", "a", "a"],
            }
        )
    )
    intervention = InputIntervention(
        "permute_mass_within_species_charge", dataset, seed=7
    )
    spectra = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    padding_mask = torch.zeros((4, 3), dtype=torch.bool)
    mass = torch.tensor([1000.0, 1200.0, 2100.0, 2400.0])
    charge = torch.tensor([2, 2, 3, 3])

    _, _, changed_mass, unchanged_charge = intervention.apply(
        spectra, padding_mask, mass, charge, row_indices=torch.arange(4)
    )

    assert sorted(changed_mass[:2].tolist()) == [1000.0, 1200.0]
    assert sorted(changed_mass[2:].tolist()) == [2100.0, 2400.0]
    assert torch.equal(unchanged_charge, charge)


def test_mass_controlled_retrieval_requires_positive_and_negative_candidates():
    values = torch.tensor([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]])
    report = evaluate_peptide_embeddings(
        values,
        ["p1", "p1", "p2", "p2"],
        ["a"] * 4,
        {
            "retrieval": {"metric": "cosine", "ks": [1]},
            "compactness": {"enabled": False},
            "mass_controlled_retrieval": {"enabled": True, "ppm_tolerance": 10.0},
        },
        device=torch.device("cpu"),
        precursor_mz=torch.tensor([500.0, 500.0, 500.001, 500.001]),
        precursor_charges=torch.tensor([2, 2, 2, 2]),
    )

    metrics = report["partitions"]["a"]
    assert metrics["mass_controlled_n_eligible_queries"] == 2.0
    assert metrics["mass_controlled_hit_at_1"] == 1.0


def test_cross_charge_retrieval_requires_a_mass_matched_negative():
    values = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    report = evaluate_peptide_embeddings(
        values,
        ["p1", "p1", "p2", "p2"],
        ["a"] * 4,
        {
            "retrieval": {"metric": "cosine", "ks": [1]},
            "compactness": {"enabled": False},
            "cross_charge_retrieval": {
                "enabled": True,
                "neutral_mass_ppm_tolerance": 10.0,
            },
        },
        device=torch.device("cpu"),
        precursor_mz=torch.tensor([500.0, 250.0, 500.001, 250.0005]),
        precursor_charges=torch.tensor([2, 4, 2, 4]),
    )
    metrics = report["partitions"]["a"]
    assert metrics["cross_charge_n_eligible_queries"] == 2.0
    assert metrics["cross_charge_hit_at_1"] == 1.0
    assert metrics["cross_charge_mass_controlled_n_eligible_queries"] == 2.0
    assert metrics["cross_charge_mass_controlled_hit_at_1"] == 1.0


def test_cross_view_pair_metrics_are_bidirectional():
    from src.embed_eval.pair_metrics import evaluate_pair_discrimination

    report = evaluate_pair_discrimination(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        ["s0", "s1", "s2", "s3"],
        [
            {"pair_set": "test", "species": "a", "left_spectrum_id": "s0", "right_spectrum_id": "s1", "label": 1},
            {"pair_set": "test", "species": "a", "left_spectrum_id": "s0", "right_spectrum_id": "s2", "label": 0},
        ],
        {"metric": "cosine"},
        availability_by_set_species=None,
        device=torch.device("cpu"),
        comparison_values=torch.tensor([[0.95, 0.05], [0.85, 0.15], [0.05, 0.95], [0.15, 0.85]]),
    )
    assert report["cross_view"] is True
    assert report["pair_sets"]["test"]["macro"]["roc_auc"] == 1.0
