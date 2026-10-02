import torch
import torch.nn as nn


class DummyEmbedderWrapper(nn.Module):
    """Wraps binned encoder to match expected embedder interface."""

    def __init__(self, encoder, **kwargs):
        super().__init__()
        self.encoder = encoder

        # Expose expected attributes
        self.running_units = encoder.running_units
        self.use_mass = encoder.use_mass
        self.use_charge = encoder.use_charge
        self.use_energy = getattr(encoder, "use_energy", False)

    def get_embedder(self, trainable: bool = False, **kwargs):
        # The binned baseline has no trainable encoder parameters. Accept the
        # common downstream interface so it can be compared directly to DINO.
        return self

    def forward(
        self,
        spectra: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        mass: torch.Tensor = None,
        charge: torch.Tensor = None,
    ) -> torch.Tensor:
        out = self.encoder(
            spectra,
            mass=mass,
            charge=charge,
            key_padding_mask=key_padding_mask,
        )
        return out["emb"].squeeze(1)  # [B, D]
