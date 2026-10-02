from math import ceil
import torch
import torch.nn as nn
from src.models.sinusoidal import FloatEncoder as DcFloatEncoder
from src.models.sinusoidal import PositionalEncoder as DcPosEncoder
from src.models.casanovo.encoders import PeakEncoder


class StaticPeakEncoder(torch.nn.Module):
    """Encode mass spectrum.

    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_mz_wavelength : float, optional
        The minimum wavelength to use for m/z.
    max_mz_wavelength : float, optional
        The maximum wavelength to use for m/z.
    min_intensity_wavelength : float, optional
        The minimum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    max_intensity_wavelength : float, optional
        The maximum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    mz_to_int_ratio: int, optional
        1 means only consider mz, 0 only intensity
    """

    def __init__(
        self,
        d_model: int,
        min_mz_wavelength: float = 0.001,
        max_mz_wavelength: float = 10000,
        min_intensity_wavelength: float = 1e-6,
        max_intensity_wavelength: float = 1,
        learnable_wavelengths: bool = False,
        mz_to_int_ratio: float = 2 / 3,
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__()
        self.d_model = d_model
        self.learnable_wavelengths = learnable_wavelengths

        d_mz = ceil(mz_to_int_ratio * d_model)
        d_int = d_model - d_mz

        self.mz_encoder = DcFloatEncoder(
            d_model=d_mz,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
            learnable_wavelengths=False,
        )

        self.int_encoder = DcFloatEncoder(
            d_model=d_int,
            min_wavelength=min_intensity_wavelength,
            max_wavelength=max_intensity_wavelength,
            learnable_wavelengths=False,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, d_model)
            The encoded features for the mass spectra.
        """
        encoded = torch.cat(
            [
                self.mz_encoder(X[:, :, 0]),
                self.int_encoder(X[:, :, 1]),
            ],
            dim=2,
        )

        return encoded

