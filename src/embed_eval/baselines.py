"""Fixed embedding-baseline construction and report provenance."""

from __future__ import annotations


def binned_spectrum_details(global_args, *, dimension: int = 1024) -> dict[str, object]:
    """Describe the fixed peak-only binned representation used for evaluation."""
    max_mz = float(global_args.max_mz)
    return {
        "representation": "peak_only_intensity_binned",
        "dimension": dimension,
        "metadata_features": [],
        "mz_range": [0.0, max_mz],
        "bin_count": dimension,
        "bin_width_da": max_mz / dimension,
        "bin_aggregation": "sum input intensities within each m/z bin",
        "input_intensity_scaling": global_args.intensity_scaling,
        "post_binning_normalization": "divide by each spectrum's maximum bin intensity",
        "similarity_normalization": "L2 normalization inside cosine evaluation",
    }


def similarity_metadata(metric: str) -> dict[str, object]:
    """Record cosine's equivalent normalized spectral-angle interpretation."""
    metadata: dict[str, object] = {"primary": metric}
    if metric == "cosine":
        metadata["equivalent_normalized_spectral_angle"] = {
            "formula": "1 - (2/pi) * arccos(clamp(cosine, -1, 1))",
            "ranking_metrics_identical_to_cosine": True,
            "separately_recomputed": False,
        }
    return metadata
