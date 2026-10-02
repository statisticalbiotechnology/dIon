""" From: https://github.com/Noble-Lab/casanovo/blob/c3d2bbac7cc2550c524e04accde4765cdf850bd4/casanovo/denovo/evaluate.py
    Methods to evaluate peptide-spectrum predictions. """

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
from spectrum_utils.utils import mass_diff


# AMINO ACID AND MODIFICATION VOCABULARY
# NOTE: str formatting modified by us
RESIDUES = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C_+57.02": 160.030649,  # 103.009185 + 57.021464
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    # Amino acid modifications.
    "M_+15.99": 147.035400,  # Met oxidation:   131.040485 + 15.994915
    "N_+.98": 115.026943,  # Asn deamidation: 114.042927 +  0.984016
    "Q_+.98": 129.042594,  # Gln deamidation: 128.058578 +  0.984016
    # N-terminal modifications.
    "_+42.011": 42.010565,  # Acetylation
    "_+43.006": 43.005814,  # Carbamylation
    "_-17.027": -17.026549,  # NH3 loss
    "_+43.006-17.027": 25.980265,  # Carbamylation and NH3 loss
}


def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float] = RESIDUES,
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    This is a similar evaluation criterion as used by DeepNovo.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match_prefix_suffix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float] = RESIDUES,
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix and suffix amino acids between two peptide
    sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    # Find longest mass-matching prefix.
    aa_matches, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match
    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float] = RESIDUES,
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    if mode == "best":
        return aa_match_prefix_suffix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "forward":
        return aa_match_prefix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "backward":
        aa_matches, pep_match = aa_match_prefix(
            list(reversed(peptide1)),
            list(reversed(peptide2)),
            aa_dict,
            cum_mass_threshold,
            ind_mass_threshold,
        )
        return aa_matches[::-1], pep_match
    else:
        raise ValueError("Unknown evaluation mode")


def aa_match_batch(
    peptides1: Iterable,
    peptides2: Iterable,
    aa_dict: Dict[str, float] = RESIDUES,
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : Iterable
        The first list of peptide sequences to be compared.
    peptides2 : Iterable
        The second list of peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    for peptide1, peptide2 in zip(peptides1, peptides2):
        # Split peptides into individual AAs if necessary.
        if isinstance(peptide1, str):
            peptide1 = re.split(r"(?<=.)(?=[A-Z])", peptide1)
        if isinstance(peptide2, str):
            peptide2 = re.split(r"(?<=.)(?=[A-Z])", peptide2)
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)
        aa_matches_batch.append(
            aa_match(
                peptide1,
                peptide2,
                aa_dict,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2


def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa_true: int
        Total number of amino acids in the true peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the predicted peptide sequences.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_precision: float
        The number of correct peptide predictions divided by the number of
        peptides.
    """
    n_aa_correct = sum([aa_matches[0].sum() for aa_matches in aa_matches_batch])
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_precision = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )
    return float(aa_precision), float(aa_recall), float(pep_precision)


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total
