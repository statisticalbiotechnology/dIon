"""Unit tests for the released InstaNovo-FM preprocessing contract."""

from __future__ import annotations

import numpy as np

from src.instanovo_fm_preprocessing import process_spectrum


def test_filters_and_normalizes_without_reordering_when_topk_is_not_needed():
    result = process_spectrum(
        np.array([40.0, 100.0, 200.0, 2600.0], dtype=np.float32),
        np.array([10.0, 4.0, 9.0, 100.0], dtype=np.float32),
        precursor_mz=500.0,
    )
    assert result is not None
    assert np.allclose(result[:, 0], [100.0 / 2500.0, 200.0 / 2500.0])
    assert np.isclose(np.linalg.norm(result[:, 1]), 1.0)
    assert np.allclose(result[:, 1], [2.0 / np.sqrt(13.0), 3.0 / np.sqrt(13.0)])


def test_returns_sentinel_when_no_peak_survives():
    result = process_spectrum(
        np.array([20.0, 2600.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        precursor_mz=None,
    )
    assert np.array_equal(result, np.array([[0.0, 1.0]], dtype=np.float32))
