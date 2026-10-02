import torch

from src.precursor_conditioning import condition_precursor_inputs


def test_null_precursor_uses_zero_mass_and_charge():
    mass = torch.tensor([1000.0, 1500.0])
    charge = torch.tensor([2, 3])

    null_mass, null_charge = condition_precursor_inputs(mass, charge, "null")

    assert torch.equal(null_mass, torch.zeros_like(mass))
    assert torch.equal(null_charge, torch.zeros_like(charge))
    assert torch.equal(mass, torch.tensor([1000.0, 1500.0]))
    assert torch.equal(charge, torch.tensor([2, 3]))
