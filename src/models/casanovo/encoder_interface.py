import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .transformers import SpectrumEncoder  # Importing from your local module


class CasanovoSpectrumTransformerEncoder(SpectrumEncoder):
    """A wrapper around Casanovo's SpectrumEncoder to match the expected interface."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        peak_encoder: bool = True,
        dim_intensity: Optional[int] = None,
    ):
        super().__init__(
            dim_model=d_model,
            n_head=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            peak_encoder=peak_encoder,
            dim_intensity=dim_intensity,
        )
        self.n_layers = n_layers
        self.running_units = d_model

        # No support for this
        self.use_mass = False
        self.use_charge = False
        self.use_energy = False

    def forward(
        self,
        mz_int: torch.Tensor | None = None,
        spectra: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        """The forward pass.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        emb : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        if spectra is None:
            spectra = mz_int
        if spectra is None:
            spectra = kwargs.get("spectra")
        if spectra is None:
            raise ValueError("Expected mz_int or spectra.")

        if key_padding_mask is None:
            zeros = ~spectra.sum(dim=2).bool()
            mask = [
                torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
        else:
            mask = torch.cat(
                [
                    torch.tensor([[False]] * spectra.shape[0]).type_as(
                        key_padding_mask
                    ),
                    key_padding_mask,
                ],
                dim=1,
            )

        peaks = self.peak_encoder(spectra)

        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        peaks = torch.cat([latent_spectra, peaks], dim=1)
        emb = self.transformer_encoder(peaks, src_key_padding_mask=mask)

        return {"emb": emb, "mask": mask, "num_cem_tokens": 0}

    def get_layer_id(self, param_name: str) -> int:
        """
        Assign a parameter with its layer id for layer-wise learning rate decay.
        """
        if param_name.startswith("peak_encoder") or param_name.startswith(
            "latent_spectrum"
        ):
            return 0
        elif param_name.startswith("transformer_encoder.layers."):
            return int(param_name.split(".")[2])
        else:
            return self.n_layers

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


def casanovo_encoder_base(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    dropout=0.0,
    cls_token=False,
    dim_intensity=None,
    **kwargs,
):
    if use_charge or use_energy or use_mass:
        raise ValueError(
            "Casanovo's SpectrumEncoder does not support use_charge, use_energy, or use_mass."
        )

    if static_peak_encoder:
        raise ValueError(
            "Casanovo's SpectrumEncoder does not support static_peak_encoder."
        )

    if cls_token:
        raise ValueError("Casanovo's SpectrumEncoder does not support cls_token.")


    # Set hyperparameters according to Casanovo's configuration
    d_model = 512
    nhead = 8
    dim_feedforward = 1024
    n_layers = 9
    dropout = dropout if dropout is not None else 0.0
    peak_encoder = True  # Casanovo uses PeakEncoder by default

    # Create an instance of the CasanovoSpectrumTransformerEncoder
    model = CasanovoSpectrumTransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        n_layers=n_layers,
        dropout=dropout,
        peak_encoder=peak_encoder,
        dim_intensity=dim_intensity,
    )
    return model
