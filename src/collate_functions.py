from typing import Any, Dict, Iterable, List, Tuple, Union
import torch
from src.data.unified_tokenizer import PeptideTokenizer


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def subsample_max_peaks(mz_tensor, int_tensor, max_peaks=300):
    """Subsample mass spectra to retain the peaks with the highest intensities.

    Parameters
    ----------
    mz_tensor : torch.Tensor
        Tensor containing m/z values.
    int_tensor : torch.Tensor
        Tensor containing intensity values.
    max_peaks : int, optional
        Maximum number of peaks to retain, by default 300.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Subsampled m/z tensor and intensity tensor.
    """
    combined_tensor = torch.stack([mz_tensor, int_tensor], dim=-1)

    # Sort by intensity
    sorted_indices = combined_tensor[:, 1].argsort(descending=True)

    # Take the max_peaks highest pairs
    highest_indices = sorted_indices[:max_peaks]
    # Keep original order
    highest_indices_original_order, _ = highest_indices.sort()

    subsampled_tensor = combined_tensor[highest_indices_original_order]

    # Split back into m/z and intensity tensors
    subsampled_mz = subsampled_tensor[:, 0]
    subsampled_intensity = subsampled_tensor[:, 1]
    return subsampled_mz, subsampled_intensity


def default_filter_peaks(
    mz_tensor_b,
    intensity_tensor_b,
    max_peaks,
    intensity_scaling,
    min_mz,
    max_mz,
):
    """Default filtering: apply m/z window, subsample top peaks, and intensity scaling."""

    # Keep only peaks within the configured m/z range.
    mz_mask = (mz_tensor_b >= min_mz) & (mz_tensor_b <= max_mz)
    mz_tensor_b = mz_tensor_b[mz_mask]
    intensity_tensor_b = intensity_tensor_b[mz_mask]

    # Nothing left after filtering.
    if len(mz_tensor_b) == 0:
        return mz_tensor_b, intensity_tensor_b

    # Subsample if the number of peaks exceeds the maximum allowed
    if max_peaks > 0 and len(mz_tensor_b) > max_peaks:
        mz_tensor_b, intensity_tensor_b = subsample_max_peaks(
            mz_tensor_b, intensity_tensor_b, max_peaks
        )

    if intensity_scaling == "minmax":
        intensity_tensor_b = minmax_scale(intensity_tensor_b)
    elif intensity_scaling == "basepeak":
        intensity_tensor_b = basepeak_scale(intensity_tensor_b)
    elif intensity_scaling == "none":
        pass
    else:
        raise ValueError(f"Unknown intensity scaling method: '{intensity_scaling}'")

    return mz_tensor_b, intensity_tensor_b


def casanovo_filter_peaks(
    mz_tensor_b,
    intensity_tensor_b,
    precursor_mz,
    min_mz,
    max_mz,
    min_intensity,
    remove_precursor_tol,
    max_peaks,
    eps=1e-11,
):
    """Apply Casanovo-specific filtering and preprocessing to the peaks."""
    # Remove peaks outside the specified m/z range
    mz_mask = (mz_tensor_b >= min_mz) & (mz_tensor_b <= max_mz)
    mz_tensor_b = mz_tensor_b[mz_mask]
    intensity_tensor_b = intensity_tensor_b[mz_mask]

    # Remove peaks within 2 Da of the observed precursor m/z
    if remove_precursor_tol > 0:
        mz_mask = torch.abs(mz_tensor_b - precursor_mz) > remove_precursor_tol
        mz_tensor_b = mz_tensor_b[mz_mask]
        intensity_tensor_b = intensity_tensor_b[mz_mask]

    # Remove peaks with intensity lower than min_intensity fraction of the most intense peak
    max_intensity = intensity_tensor_b.max()
    intensity_threshold = (
        max_intensity * min_intensity
    )  # FIXME: very crude, noise level instead?
    intensity_mask = intensity_tensor_b >= intensity_threshold
    mz_tensor_b = mz_tensor_b[intensity_mask]
    intensity_tensor_b = intensity_tensor_b[intensity_mask]

    mz_tensor_b, intensity_tensor_b = (
        subsample_max_peaks(  # TODO: try subsampling top x peaks in y bands
            mz_tensor_b, intensity_tensor_b, max_peaks
        )
    )

    # Square-root transform the intensities
    intensity_tensor_b = torch.sqrt(
        intensity_tensor_b
    )  # Potentially move before intensity filtering step

    # Normalize intensities by dividing by the sum of the square-rooted intensities
    intensity_sum = intensity_tensor_b.sum()
    intensity_tensor_b = intensity_tensor_b / (intensity_sum + eps)

    return mz_tensor_b, intensity_tensor_b


def minmax_scale(
    tensor: torch.Tensor, scale_range: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    """Scale each sequence in the tensor to the specified range along the last dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be scaled.
    scale_range : Tuple[float, float], optional
        Range to scale the tensor to, by default (0, 1).

    Returns
    -------
    torch.Tensor
        Scaled tensor.
    """
    # Calculate min and max along the last dimension
    min_vals, _ = torch.min(tensor, dim=-1, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=-1, keepdim=True)

    # Scale each sequence to the specified range
    scaled_tensor = scale_range[0] + (tensor - min_vals) * (
        scale_range[1] - scale_range[0]
    ) / (max_vals - min_vals)

    return scaled_tensor


def basepeak_scale(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each sequence by its maximum (base peak) intensity."""
    max_vals, _ = torch.max(tensor, dim=-1, keepdim=True)
    return tensor / (max_vals + eps)


def process_precursor_info(
    b,
    precursor_mz_name: Union[str, bool] = "precursor_mz",
    precursor_mass_name: Union[str, bool] = "precursor_mass",
):
    """Process and standardize precursor m/z and mass information in the batch item."""
    # Handle the case where precursor_mz goes by another name in the source file
    if precursor_mz_name and precursor_mz_name != "precursor_mz":
        b["precursor_mz"] = b[precursor_mz_name]

    # Handle the case where precursor_mass goes by another name in the source file
    if precursor_mass_name and precursor_mass_name != "precursor_mass":
        b["precursor_mass"] = b[precursor_mass_name]

    # If precursor_mz doesn't exist in the source file, but precursor_mass does,
    # then calculate the mz from the mass and charge
    if precursor_mass_name and not precursor_mz_name:
        b["precursor_mz"] = b["precursor_mass"] / b["precursor_charge"]

    # If precursor_mass doesn't exist in the source file, but precursor_mz does,
    # then calculate the mass from the mz and charge
    if precursor_mz_name and not precursor_mass_name:
        b["precursor_mass"] = b["precursor_mz"] * b["precursor_charge"]


def top_intensity_minmax(
    mz: torch.Tensor, intensity: torch.Tensor, max_peaks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the most intense peaks, restore m/z order, then min-max scale."""
    if max_peaks > 0 and len(mz) > max_peaks:
        mz, intensity = subsample_max_peaks(mz, intensity, max_peaks)
    return mz, minmax_scale(intensity)


def pad_peaks(
    batch: Iterable[Dict[Any, Union[List, torch.Tensor]]],
    precision: torch.dtype = torch.float32,
    max_peaks: int = 300,
    # Dataset specific setting for grabbing the correct precursor mass/mz
    precursor_mz_name: Union[str, bool] = "precursor_mz",
    precursor_mass_name: Union[str, bool] = "precursor_mass",
    # Peak filter settings
    filter_method: str = "default",
    intensity_scaling: str = "minmax",
    min_mz: float = 50.0,
    max_mz: float = 2500.0,
    min_intensity: float = 0.01,
    remove_precursor_tol: float = 2,
    min_peaks: int = 0,
) -> Dict[str, Union[torch.Tensor, list[Any]]]:

    mz_tensors = []
    int_tensors = []
    kept_items = []
    lengths_list = []

    # Preprocess precursor m/z and mass for each item
    for b in batch:
        process_precursor_info(b, precursor_mz_name, precursor_mass_name)

    for i, b in enumerate(batch):
        mz_tensor_b = _as_tensor(b.pop("mz_array"), precision)
        intensity_tensor_b = _as_tensor(b.pop("intensity_array"), precision)

        if filter_method == "casanovo":
            precursor_mz = b.get("precursor_mz")
            if precursor_mz is None:
                raise ValueError("Precursor m/z not found in batch item.")
            mz_tensor_b, intensity_tensor_b = casanovo_filter_peaks(
                mz_tensor_b,
                intensity_tensor_b,
                precursor_mz,
                min_mz,
                max_mz,
                min_intensity,
                remove_precursor_tol,
                max_peaks,
            )
        elif filter_method == "default":
            mz_tensor_b, intensity_tensor_b = default_filter_peaks(
                mz_tensor_b,
                intensity_tensor_b,
                max_peaks,
                intensity_scaling,
                min_mz,
                max_mz,
            )
        elif filter_method == "top_intensity_minmax":
            mz_tensor_b, intensity_tensor_b = top_intensity_minmax(
                mz_tensor_b, intensity_tensor_b, max_peaks
            )
        else:
            raise ValueError(f"Unknown spectrum filtering method: '{filter_method}'")

        # ---------- DROP SMALL SPECTRA ----------
        if len(mz_tensor_b) < min_peaks:
            continue

        mz_tensors.append(mz_tensor_b)
        int_tensors.append(intensity_tensor_b)
        lengths_list.append(len(mz_tensor_b))
        kept_items.append(b)

    mz_array = torch.nn.utils.rnn.pad_sequence(mz_tensors, batch_first=True)
    intensity_array = torch.nn.utils.rnn.pad_sequence(int_tensors, batch_first=True)

    batch = torch.utils.data.default_collate(kept_items)
    batch["mz_array"] = mz_array
    batch["intensity_array"] = intensity_array
    batch["peak_lengths"] = torch.tensor(lengths_list, dtype=torch.int32).unsqueeze(1)

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
            batch[key] = val.type(precision)

    return batch


def pad_peptides(
    batch: Iterable[Dict[Any, Union[List, torch.Tensor]]],
    precision: torch.dtype = torch.float32,
    max_peaks: int = 300,
    # Peptide sequence settings
    max_length: int = 30,
    pad_token_id: int | None = None,
    tokenizer: PeptideTokenizer = None,
    label_name="sequence",
    # Dataset specific setting for grabbing the correct precursor mass/mz
    precursor_mz_name: Union[str, bool] = "precursor_mz",
    precursor_mass_name: Union[str, bool] = "precursor_mass",
    # Peak filter settings
    filter_method: str = "default",
    intensity_scaling: str = "minmax",
    min_mz: float = 50.0,
    max_mz: float = 2500.0,
    min_intensity: float = 0.01,
    remove_precursor_tol: float = 2,  # Remove peaks within 2 Da of precursor m/z
) -> Dict[str, Union[torch.Tensor, list[Any]]]:
    """
    Transform compatible data types into PyTorch tensors and
    pad the m/z and intensity arrays of each mass spectrum with
    zeros to be stacked into a tensor.
    """
    intseqs = []
    peptide_lengths = torch.zeros((len(batch), 1), dtype=torch.int32)

    for i, b in enumerate(batch):
        if tokenizer is not None:
            seq = b.pop(label_name)
            b["intseq"] = tokenizer.tokenize(seq)
        intseq_b = b.pop("intseq")
        intseq_b = intseq_b[:max_length]
        peptide_lengths[i] = len(intseq_b)
        intseqs.append(intseq_b)

    if pad_token_id is None:
        raise ValueError("pad_token_id must be provided for peptide padding.")
    intseq_array = torch.nn.utils.rnn.pad_sequence(
        intseqs, batch_first=True, padding_value=pad_token_id
    )

    batch = pad_peaks(
        batch,
        precision=precision,
        max_peaks=max_peaks,
        precursor_mz_name=precursor_mz_name,
        precursor_mass_name=precursor_mass_name,
        filter_method=filter_method,
        intensity_scaling=intensity_scaling,
        min_mz=min_mz,
        max_mz=max_mz,
        min_intensity=min_intensity,
        remove_precursor_tol=remove_precursor_tol,
    )

    batch["intseq"] = intseq_array
    batch["peptide_lengths"] = peptide_lengths

    return batch


def pad_metric_learning(
    batch: Iterable[Dict[Any, Union[List, torch.Tensor]]],
    *,
    label_name: str,
    **peak_kwargs,
) -> Dict[str, Union[torch.Tensor, list[Any]]]:
    """Pad spectra while preserving exact peptide identities for SupCon."""
    collated = pad_peaks(batch, **peak_kwargs)
    labels = collated.get(label_name)
    if labels is None:
        raise KeyError(f"Metric-learning batch is missing label column {label_name!r}.")
    collated["peptide_id"] = [str(label) for label in labels]
    return collated
