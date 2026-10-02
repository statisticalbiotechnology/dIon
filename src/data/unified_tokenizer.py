import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from sortedcontainers import SortedDict, SortedSet


UNIFIED_RESIDUES: Dict[str, float] = {
    # Base amino acids
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C": 103.009185,
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
    "O": 237.147727,
    # Common modifications (canonical tokens)
    "C[+57.021]": 160.030649,  # 103.009185 + 57.021464
    "M[+15.995]": 147.035400,  # 131.040485 + 15.994915
    "N[+0.984]": 115.026943,  # 114.042927 + 0.984016
    "Q[+0.984]": 129.042594,  # 128.058578 + 0.984016
    "S[+79.97]": 166.998028,  # 87.032028 + 79.966
    "T[+79.97]": 181.013670,  # 101.047670 + 79.966
    "Y[+79.97]": 243.029329,  # 163.063329 + 79.966
    # N-terminal modifications
    "[+42.011]": 42.010565,
    "[+43.006]": 43.005814,
    "[-17.027]": -17.026549,
    "[+25.980]": 25.980265,
}

UNIFIED_N_TERMINAL: List[str] = [
    "[+42.011]",
    "[+43.006]",
    "[-17.027]",
    "[+25.980]",
]

UNIFIED_ALIAS: Dict[str, str] = {
    # Carbamidomethyl C
    "C+57.021": "C[+57.021]",
    "C[+57.02]": "C[+57.021]",
    "C[57.021]": "C[+57.021]",
    "C_+57.02": "C[+57.021]",
    "C(+57.02)": "C[+57.021]",
    # Oxidation / deamidation
    "M+15.995": "M[+15.995]",
    "M[+15.9949]": "M[+15.995]",
    "M[15.9949]": "M[+15.995]",
    "M_+15.99": "M[+15.995]",
    "N+0.984": "N[+0.984]",
    "N[+0.9840]": "N[+0.984]",
    "N[0.9840]": "N[+0.984]",
    "N_+.98": "N[+0.984]",
    "Q+0.984": "Q[+0.984]",
    "Q[+0.9840]": "Q[+0.984]",
    "Q[0.9840]": "Q[+0.984]",
    "Q_+.98": "Q[+0.984]",
    # Phospho
    "S[+79.966]": "S[+79.97]",
    "T[+79.966]": "T[+79.97]",
    "Y[+79.966]": "Y[+79.97]",
    # N-terminal mods
    "+42.011": "[+42.011]",
    "_+42.011": "[+42.011]",
    "[42.0106]": "[+42.011]",
    "+43.006": "[+43.006]",
    "_+43.006": "[+43.006]",
    "[43.0058]": "[+43.006]",
    "-17.027": "[-17.027]",
    "_-17.027": "[-17.027]",
    "[-17.0265]": "[-17.027]",
    "+43.006-17.027": "[+25.980]",
    "_+43.006-17.027": "[+25.980]",
    "[25.9803]": "[+25.980]",
}


_NUMERIC_MASS_DELTA_BASE_MASSES = {
    "G": 57.021463735, "A": 71.037113805, "S": 87.032028435,
    "P": 97.052763875, "V": 99.068413945, "T": 101.047678505,
    "C": 103.009184505, "L": 113.084064015, "I": 113.084064015,
    "N": 114.042927470, "D": 115.026943065, "Q": 128.058577540,
    "K": 128.094963050, "E": 129.042593135, "M": 131.040484645,
    "H": 137.058911875, "F": 147.068413945, "R": 156.101111050,
    "Y": 163.063328575, "W": 186.079312980, "O": 237.147726925,
}
_NUMERIC_MASS_DELTA_RESIDUE_RE = re.compile(
    r"^(?P<aa>[ACDEFGHIKLMNPQRSTVWYO])(?P<deltas>(?:[+-]\d+(?:\.\d+)?)*)$"
)
_NUMERIC_MASS_DELTA_NTERM_RE = re.compile(r"^(?:[+-]\d+(?:\.\d+)?)+$")
_NUMERIC_MASS_DELTA_KNOWN_DELTAS = {
    57.021: 57.021464,
    15.995: 15.994915,
    0.984: 0.984016,
    42.011: 42.010565,
    43.006: 43.005814,
    -17.027: -17.026549,
}


def _numeric_mass_delta_value(value: str) -> float:
    approximate = float(value)
    for rounded, exact in _NUMERIC_MASS_DELTA_KNOWN_DELTAS.items():
        if abs(approximate - rounded) <= 0.00051:
            return exact
    return approximate


def _numeric_mass_delta_token_mass(token: str) -> float:
    residue_match = _NUMERIC_MASS_DELTA_RESIDUE_RE.fullmatch(token)
    if residue_match is not None:
        return _NUMERIC_MASS_DELTA_BASE_MASSES[residue_match.group("aa")] + sum(
            _numeric_mass_delta_value(delta)
            for delta in re.findall(r"[+-]\d+(?:\.\d+)?", residue_match.group("deltas"))
        )
    if _NUMERIC_MASS_DELTA_NTERM_RE.fullmatch(token) is not None:
        return sum(
            _numeric_mass_delta_value(delta)
            for delta in re.findall(r"[+-]\d+(?:\.\d+)?", token)
        )
    raise ValueError(f"Unsupported numeric mass-delta token in manifest: {token!r}")


class PeptideTokenizer:
    """Tokenizer for peptide sequences with unified modification handling.

    Special tokens follow HuggingFace-style naming and attributes:
    - pad_token / pad_token_id
    - bos_token / bos_token_id
    - eos_token / eos_token_id
    - unk_token / unk_token_id
    """

    def __init__(
        self,
        residues: Dict[str, float],
        n_terminal: Optional[List[str]] = None,
        alias: Optional[Dict[str, str]] = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        pad_token: str = "<PAD>",
        bos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>",
        add_unk_token: bool = True,
        uppercase_input: bool = False,
        duplicate_nterm_token: bool = True,
    ) -> None:
        self.residues = residues
        self.n_terminal_mods = n_terminal or []
        self._raw_to_canonical = alias or {}
        self._canonical_to_raw: Dict[str, set[str]] = {}
        for raw, canonical in self._raw_to_canonical.items():
            self._canonical_to_raw.setdefault(canonical, set()).add(raw)
        self._auto_alias = self._build_auto_alias()

        self.replace_isoleucine_with_leucine = replace_isoleucine_with_leucine
        self.reverse = reverse

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token if add_unk_token else None
        self.uppercase_input = bool(uppercase_input)
        self.duplicate_nterm_token = bool(duplicate_nterm_token)

        tokens = SortedSet(residues.keys())
        self.index = SortedDict({k: i for i, k in enumerate(tokens)})
        self.reverse_index = list(tokens)

        self._special_tokens = {self.pad_token, self.bos_token, self.eos_token}
        if self.unk_token is not None:
            self._special_tokens.add(self.unk_token)
        for tok in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
            if tok is None:
                continue
            if tok not in self.index:
                self.index[tok] = len(self.index)
                self.reverse_index.append(tok)

        self.pad_token_id = self.index[self.pad_token]
        self.bos_token_id = self.index[self.bos_token]
        self.eos_token_id = self.index[self.eos_token]
        self.unk_token_id = self.index[self.unk_token] if self.unk_token else None

        raw_variants = set()
        for canonical in residues.keys():
            raw_variants.add(canonical)
            raw_variants.update(self._canonical_to_raw.get(canonical, set()))
            raw_variants.update(self._auto_alias.get(canonical, set()))

        self._special_residue_tokens = sorted(raw_variants, key=len, reverse=True)
        self._nterm_raw_variants = self._build_nterm_raw_variants()

    @classmethod
    def unified(
        cls,
        reverse: bool = False,
        replace_isoleucine_with_leucine: bool = False,
    ) -> "PeptideTokenizer":
        return cls(
            residues=UNIFIED_RESIDUES,
            n_terminal=UNIFIED_N_TERMINAL,
            alias=UNIFIED_ALIAS,
            replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
            reverse=reverse,
        )

    @classmethod
    def from_numeric_mass_delta_manifest(
        cls,
        manifest_path: str | Path,
        *,
        reverse: bool = False,
        replace_isoleucine_with_leucine: bool = False,
        pad_token: str = "X",
        bos_token: str = "<EOS>",
        eos_token: str = "<EOS>",
        add_unk_token: bool = False,
    ) -> "PeptideTokenizer":
        """Build the PA1.1-compatible tokenizer from a portable V3 manifest."""
        path = Path(manifest_path).expanduser()
        payload = json.loads(path.read_text())
        if payload.get("format") != "numeric_mass_delta_v1":
            raise ValueError(f"Unsupported numeric mass-delta manifest: {path}")
        residue_tokens = payload.get("residue_tokens")
        nterm_tokens = payload.get("nterm_prefix_tokens", payload.get("nterm_delta_tokens"))
        if not isinstance(residue_tokens, dict) or not isinstance(nterm_tokens, dict):
            raise ValueError(f"Malformed numeric mass-delta manifest: {path}")
        residues = {
            token: _numeric_mass_delta_token_mass(token)
            for token, count in residue_tokens.items()
            if int(count) > 0
        }
        residues.update({
            token: _numeric_mass_delta_token_mass(token)
            for token, count in nterm_tokens.items()
            if int(count) > 0
        })
        if not residues:
            raise ValueError(f"Numeric mass-delta manifest has no active tokens: {path}")
        aliases = {
            "C[CARBAMIDOMETHYL]": "C+57.021",
            "C[Carbamidomethyl]": "C+57.021",
            "M[OXIDATION]": "M+15.995",
            "M[Oxidation]": "M+15.995",
            "N[DEAMIDATED]": "N+0.984",
            "N[Deamidated]": "N+0.984",
            "Q[DEAMIDATED]": "Q+0.984",
            "Q[Deamidated]": "Q+0.984",
            "S[PHOSPHO]": "S+79.966",
            "S[Phospho]": "S+79.966",
            "T[PHOSPHO]": "T+79.966",
            "T[Phospho]": "T+79.966",
            "Y[PHOSPHO]": "Y+79.966",
            "Y[Phospho]": "Y+79.966",
            "[ACETYL]-": "+42.011",
            "[Acetyl]-": "+42.011",
            "[CARBAMYL]-": "+43.006",
            "[Carbamyl]-": "+43.006",
            "[AMMONIA-LOSS]-": "-17.027",
            "[Ammonia-loss]-": "-17.027",
            "[+25.980265]-": "+43.006-17.027",
        }
        return cls(
            residues=residues,
            n_terminal=sorted(token for token, count in nterm_tokens.items() if int(count) > 0),
            alias=aliases,
            reverse=reverse,
            replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_unk_token=add_unk_token,
            uppercase_input=True,
            duplicate_nterm_token=False,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.index)

    @property
    def stop_token_id(self) -> int:
        return self.eos_token_id

    def nterm_token_ids(self) -> List[int]:
        return [self.index[tok] for tok in self.n_terminal_mods if tok in self.index]

    def is_nterm_token_id(self, token_id: int) -> bool:
        return token_id in set(self.nterm_token_ids())

    def _build_nterm_raw_variants(self) -> List[str]:
        if not self.n_terminal_mods:
            return []
        raw_variants = set()
        for canonical in self.n_terminal_mods:
            raw_variants.add(canonical)
            raw_variants.update(self._canonical_to_raw.get(canonical, set()))
            raw_variants.update(self._auto_alias.get(canonical, set()))
        return sorted(raw_variants, key=len, reverse=True)

    def _canonicalize(self, token: str) -> str:
        if token in self._raw_to_canonical:
            return self._raw_to_canonical[token]
        for canonical, variants in self._auto_alias.items():
            if token in variants:
                return canonical
        return token

    def _build_auto_alias(self) -> Dict[str, set[str]]:
        auto_alias: Dict[str, set[str]] = {}

        def _round_variant(mod_str: str, ndigits: int) -> str | None:
            try:
                val = float(mod_str)
            except ValueError:
                return None
            return f"{val:+.{ndigits}f}"

        for canonical in self.residues.keys():
            variants = set()
            # AA[+57.021] -> AA+57.021, AA(+57.021), AA_+57.021, AA+57.02, AA_+57.02
            m = re.match(r"^([A-Z])\[(\+?-?[0-9.]+)\]$", canonical)
            if m:
                aa, mod = m.groups()
                variants.add(f"{aa}{mod}")
                variants.add(f"{aa}({mod})")
                variants.add(f"{aa}_{mod}")
                rounded = _round_variant(mod, 2)
                if rounded is not None:
                    variants.add(f"{aa}{rounded}")
                    variants.add(f"{aa}({rounded})")
                    variants.add(f"{aa}_{rounded}")

            # N-term [+42.011] -> +42.011, _+42.011, +42.01, _+42.01
            m = re.match(r"^\[(\+?-?[0-9.]+)\]$", canonical)
            if m:
                mod = m.groups()[0]
                variants.add(mod)
                variants.add(f"_{mod}")
                rounded = _round_variant(mod, 2)
                if rounded is not None:
                    variants.add(rounded)
                    variants.add(f"_{rounded}")

            if variants:
                auto_alias[canonical] = variants

        return auto_alias

    def _extract_nterm(self, sequence: str) -> tuple[str | None, str]:
        for raw in self._nterm_raw_variants:
            if sequence.startswith(raw):
                return self._canonicalize(raw), sequence[len(raw) :]
        return None, sequence

    def _parse_peptide(self, sequence: str) -> List[str]:
        sequence = str(sequence).strip()
        if self.uppercase_input:
            sequence = sequence.upper()
        sequence = sequence[1:] if sequence.startswith(".") else sequence
        sequence = sequence[:-1] if sequence.endswith(".") else sequence

        tokens: List[str] = []
        nterm, sequence = self._extract_nterm(sequence)
        if nterm:
            tokens.append(nterm)

        pattern = (
            "(" + "|".join(map(re.escape, self._special_residue_tokens)) + "|[A-Z])"
        )
        found = re.findall(pattern, sequence)
        for token in found:
            if not token:
                continue
            tokens.append(self._canonicalize(token))
        return tokens

    def preprocess_sequence(self, sequence: str) -> List[str]:
        if self.replace_isoleucine_with_leucine:
            sequence = sequence.replace("I", "L")
        seq_list = self._parse_peptide(sequence)
        if self.n_terminal_mods:
            nterm_set = set(self.n_terminal_mods)
            mid_nterm = [tok for tok in seq_list if tok in nterm_set]
            if mid_nterm:
                seq_list = [tok for tok in seq_list if tok not in nterm_set]
                seq_list = mid_nterm + seq_list
        if self.reverse:
            seq_list.reverse()
        return seq_list

    def tokenize(self, sequence: str) -> torch.Tensor:
        tokens = self.preprocess_sequence(sequence)
        intseq = []
        for tok in tokens:
            if tok not in self.index:
                if self.unk_token_id is None:
                    raise ValueError(f"Unknown peptide token {tok!r} in {sequence!r}")
                intseq.append(self.unk_token_id)
            else:
                intseq.append(self.index[tok])
        return torch.tensor(intseq, dtype=torch.long)

    def detokenize(
        self,
        tokens: torch.Tensor,
        join: bool = False,
        pad_token_idx: Optional[int] = None,
        EOS_token_idx: Optional[int] = None,
        exclude_stop: bool = False,
    ) -> Union[str, List[str]]:
        single_sequence = tokens.dim() == 1
        if single_sequence:
            tokens = tokens.unsqueeze(0)

        decoded = []
        for row in tokens:
            if pad_token_idx is not None:
                row = row[row != pad_token_idx]

            if EOS_token_idx is not None:
                eos_pos = (row == EOS_token_idx).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    row = row[: eos_pos[0]]
                    append_stop = True
                else:
                    append_stop = False
            else:
                append_stop = False

            seq = [self.reverse_index[i.item()] for i in row]

            if append_stop and not exclude_stop:
                seq.append("$")

            if self.reverse:
                seq.reverse()

            if join:
                seq = "".join(seq)

            decoded.append(seq)

        return decoded[0] if single_sequence else decoded
