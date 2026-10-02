"""Released InstaNovo-FM v0.1.0 spectrum preprocessing.

The released FoundationModel expects m/z values scaled to ``[0, 1]`` and
sqrt/L2-normalized intensities.  It does not apply those transformations
inside the peak encoder.  This module mirrors the released PyTorch fallback in
``instanovo_fm.data.data.DataProcessor._process_spectrum`` without importing
the external package, so dIon's external-baseline runner can remain small
and environment-independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InstaNovoFMPreprocessing:
    """Values used by the released ``instanovo-fm-v0.1.0`` FoundationModel."""

    n_peaks: int = 200
    min_mz: float = 50.0
    max_mz: float = 2500.0
    min_intensity: float = 0.01
    remove_precursor_tolerance_da: float = 0.0
    normalize_mz: bool = True

    def manifest(self) -> dict[str, int | float | bool]:
        return asdict(self)


DEFAULT_PREPROCESSING = InstaNovoFMPreprocessing()


def process_spectrum(
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    precursor_mz: float | None,
    config: InstaNovoFMPreprocessing = DEFAULT_PREPROCESSING,
) -> np.ndarray:
    """Return one released-model-ready ``[m/z, intensity]`` spectrum.

    ``torch.topk`` in InstaNovo-FM's selected PyTorch path returns its top-N
    peaks in descending intensity order; preserve that order here. The released
    processor returns ``[[0, 1]]`` when filtering empties a spectrum; preserve
    that sentinel so exported embeddings remain row-aligned with the source
    benchmark rather than silently changing a downstream split.
    """
    mz = np.asarray(mz_values, dtype=np.float32)
    intensity = np.asarray(intensity_values, dtype=np.float32)
    if mz.ndim != 1 or intensity.ndim != 1 or len(mz) != len(intensity):
        raise ValueError("m/z and intensity values must be aligned 1D arrays.")

    keep = (mz >= config.min_mz) & (mz <= config.max_mz)
    mz, intensity = mz[keep], intensity[keep]
    if not len(mz):
        return np.asarray([[0.0, 1.0]], dtype=np.float32)

    if precursor_mz is not None:
        keep = np.abs(mz - float(precursor_mz)) > config.remove_precursor_tolerance_da
        mz, intensity = mz[keep], intensity[keep]
        if not len(mz):
            return np.asarray([[0.0, 1.0]], dtype=np.float32)

    keep = intensity >= config.min_intensity
    mz, intensity = mz[keep], intensity[keep]
    if not len(mz):
        return np.asarray([[0.0, 1.0]], dtype=np.float32)

    if len(mz) > config.n_peaks:
        indices = np.argsort(intensity, kind="stable")[::-1][: config.n_peaks]
        mz, intensity = mz[indices], intensity[indices]

    intensity = np.sqrt(intensity)
    norm = float(np.linalg.norm(intensity))
    if not np.isfinite(norm) or norm <= 0:
        return np.asarray([[0.0, 1.0]], dtype=np.float32)
    intensity = intensity / norm
    if config.normalize_mz:
        mz = mz / config.max_mz
    return np.column_stack((mz, intensity)).astype(np.float32, copy=False)
