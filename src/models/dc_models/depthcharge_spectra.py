"""Transformer models to handle mass spectra (ported from depthcharge)."""
from collections.abc import Callable

import torch

from src.models.sinusoidal import PeakEncoder


class SpectrumTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout

        if callable(peak_encoder):
            self.peak_encoder = peak_encoder
        elif peak_encoder:
            self.peak_encoder = PeakEncoder(d_model)
        else:
            self.peak_encoder = torch.nn.Linear(2, d_model)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer, num_layers=n_layers
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def nhead(self) -> int:
        return self._nhead

    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def dropout(self) -> float:
        return self._dropout

    def forward(
        self,
        mz_int: torch.Tensor | None = None,
        mz_array: torch.Tensor | None = None,
        intensity_array: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> dict:
        if mz_int is None:
            if mz_array is not None and intensity_array is not None:
                mz_int = torch.stack([mz_array, intensity_array], dim=2)
            else:
                mz_int = kwargs.pop("spectra", None)
        if mz_int is None:
            raise ValueError("Expected mz_int or mz_array+intensity_array or spectra.")

        n_batch = mz_int.shape[0]
        if key_padding_mask is None:
            key_padding_mask = ~mz_int.sum(dim=2).bool()

        mask = torch.cat(
            [
                torch.tensor([[False]] * n_batch).type_as(key_padding_mask),
                key_padding_mask,
            ],
            dim=1,
        )
        peaks = self.peak_encoder(mz_int)

        latent_spectra = self.precursor_hook(
            mz_array=mz_int[:, :, 0],
            intensity_array=mz_int[:, :, 1],
            charge=charge,
            mass=mass,
            **kwargs,
        )

        peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)
        emb = self.transformer_encoder(peaks, src_key_padding_mask=mask)
        return {
            "emb": emb,
            "mask": mask,
            "num_cem_tokens": 1,
        }

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        return torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
