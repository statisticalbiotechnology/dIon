"""Transformer encoder for peak sets with optional pairwise attention bias."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

import src.models.custom.model_parts as mp


def init_encoder_weights(module: nn.Module) -> None:
    if hasattr(module, "first"):
        module.first.weight = nn.init.xavier_uniform_(module.first.weight)
        if module.first.bias is not None:
            module.first.bias = nn.init.zeros_(module.first.bias)
    if isinstance(module, mp.SelfAttention):
        module.qkv.weight = nn.init.normal_(
            module.qkv.weight, 0.0, (1.0 / 3.0) * module.indim**-0.5
        )
        module.Wo.weight = nn.init.normal_(
            module.Wo.weight, 0.0, (1.0 / 3.0) * (module.h * module.d) ** -0.5
        )
        if hasattr(module, "Wb"):
            module.Wb.weight = nn.init.zeros_(module.Wb.weight)
            module.Wb.bias = nn.init.zeros_(module.Wb.bias)
        elif hasattr(module, "Wpw"):
            module.Wpw.weight = nn.init.zeros_(module.Wpw.weight)
            module.Wpw.bias = nn.init.zeros_(module.Wpw.bias)
        if hasattr(module, "Wg"):
            module.Wg.weight = nn.init.zeros_(module.Wg.weight)
            module.Wg.bias = nn.init.constant_(module.Wg.bias, 1.0)
    elif isinstance(module, mp.FFN):
        module.W1.weight = nn.init.normal_(
            module.W1.weight, 0.0, (1.0 / 3.0) * (module.indim) ** -0.5
        )
        module.W1.bias = nn.init.zeros_(module.W1.bias)
        module.W2.weight = nn.init.normal_(
            module.W2.weight, 0.0, (1.0 / 3.0) * (module.indim * module.mult) ** -0.5
        )
    elif isinstance(module, nn.Linear):
        module.weight = nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias = nn.init.zeros_(module.bias)


class Encoder(nn.Module):
    def __init__(
        self,
        # 1D options
        in_units: int = 2,
        running_units: int = 512,
        sequence_length: int = 100,
        mz_units: int = 512,
        ab_units: int = 256,
        subdivide: bool = False,
        use_charge: bool = False,
        use_energy: bool = False,
        use_mass: bool = False,
        ce_units: int = 256,
        att_d: int = 64,
        att_h: int = 4,
        gate: bool = False,
        alphabet: bool = False,
        ffn_multiplier: int = 4,
        prenorm: bool = True,
        norm_type: str = "layer",
        preembed: bool = True,
        depth: int = 9,
        dropout: float = 0,
        bias: str | bool | None = False,
        # Pairwise options
        pw_mz_units: int | None = None,
        pw_run_units: int | None = None,
        pw_attention_ch: int = 32,
        pw_attention_h: int = 4,
        pw_blocks: int = 2,
        # Miscellaneous
        recycling_its: int = 1,
        cls_token: bool = False,
        max_charge: int = 10,
    ) -> None:
        super().__init__()
        self.running_units = running_units
        self.sl = sequence_length
        self.mz_units = mz_units
        self.ab_units = ab_units
        self.subdivide = subdivide
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass
        self.ce_units = ce_units
        self.d = att_d
        self.h = self.nhead = att_h
        self.bias = bias
        self.dropout = dropout
        self.pw_mzunits = mz_units if pw_mz_units is None else pw_mz_units
        self.pw_runits = running_units if pw_run_units is None else pw_run_units
        self.depth = depth
        self.supports_embedding_token_mask = True
        self.prenorm = prenorm
        self.norm_type = norm_type
        self.preembed = preembed
        self.its = recycling_its
        # compat
        self.dim_feedforward = ffn_multiplier * running_units
        self.encode_peaks = self.encode_mz_ab
        self.n_layers = depth

        mdim = mz_units // 4 if subdivide else mz_units
        self.mdim = mdim
        self.mz_seq = nn.Identity()

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, running_units))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pw_tokenizer = None
        pw_perceiver_units = 0

        if bias == "pairwise":
            mdimpw = self.pw_mzunits // 4 if subdivide else self.pw_mzunits
            self.mdimpw = mdimpw
            self.mzpw_seq = nn.Identity()
            self.pw_first = nn.Identity()
            self.pw_seq = nn.Sequential(
                nn.Linear(self.pw_mzunits, ffn_multiplier * self.pw_runits),
                nn.SiLU(),
                nn.Linear(ffn_multiplier * self.pw_runits, self.pw_runits),
            )

        self.atleast1 = use_charge or use_energy or use_mass
        if self.use_charge:
            self.charge_emb = torch.nn.Embedding(max_charge + 1, self.running_units)

        self.first = nn.Linear(
            mz_units + ab_units + pw_perceiver_units, running_units, bias=False
        )

        if bias is None:
            bias = False
        assert bias in ["pairwise", "regular", False, None]
        attention_dict = {
            "indim": running_units,
            "d": att_d,
            "h": att_h,
            "bias": bias,
            "bias_in_units": self.pw_runits,
            "modulator": False,
            "gate": gate,
            "dropout": dropout,
            "alphabet": alphabet,
        }
        ffn_dict = {
            "indim": running_units,
            "unit_multiplier": ffn_multiplier,
            "dropout": dropout,
            "alphabet": alphabet,
        }
        self.main = nn.ModuleList(
            [
                mp.TransBlock(
                    attention_dict,
                    ffn_dict,
                    norm_type,
                    prenorm,
                    False,
                    ce_units,
                    preembed,
                )
                for _ in range(depth)
            ]
        )
        self.main_proj = nn.Identity()

        self.norm = mp.get_norm_type(norm_type)

        self.recyc = (
            nn.Sequential(
                self.norm(running_units) if prenorm else nn.Identity(),
                nn.Identity(),
                nn.Identity() if prenorm else self.norm(running_units),
            )
            if self.its > 1
            else nn.Identity()
        )

        self.apply(init_encoder_weights)

    def total_params(self) -> int:
        return sum([m.numel() for m in self.parameters()])

    def encode_mz_ab(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        mz, intensity = torch.split(x, 1, -1)
        mz = mz.squeeze(-1)

        if self.subdivide:
            mz_sub = mp.subdivide_float(mz)
            mz_emb = mp.fourier_features(mz_sub, 1, 500, self.mdim)
        else:
            mz_emb = mp.fourier_features(mz, 0.001, 10000, self.mz_units)
        mz_emb = self.mz_seq(mz_emb)
        mz_emb = mz_emb.reshape(x.shape[0], x.shape[1], -1)

        ab_emb = mp.fourier_features(intensity[..., 0], 0.000001, 1, self.ab_units)
        out = torch.cat([mz_emb, ab_emb], dim=-1)

        if self.bias == "pairwise":
            if self.cls_token is not None:
                mz = torch.cat([torch.zeros((mz.shape[0], 1), device=mz.device), mz], dim=1)
            dtsr = mp.delta_tensor(mz, 0.0)
            if self.subdivide:
                mzpw = mp.subdivide_float(dtsr)
                mzpw_emb = mp.fourier_features(mzpw, 1, 500, self.mdimpw)
            else:
                mzpw_emb = mp.fourier_features(dtsr, 0.001, 10000, self.pw_mzunits)
            mzpw_emb = self.mzpw_seq(mzpw_emb)
            mzpw_emb = mzpw_emb.reshape(x.shape[0], mz.shape[1], mz.shape[1], -1)
        else:
            mzpw_emb = None

        return {"1d": out, "2d": mzpw_emb}

    def _pad_pw_matrix(
        self, pwemb: torch.Tensor | None, num_prefix_tokens: int
    ) -> torch.Tensor | None:
        if pwemb is None or num_prefix_tokens == 0:
            return pwemb
        bsz, seq_len, _, dim = pwemb.shape
        padded = pwemb.new_zeros(
            bsz, seq_len + num_prefix_tokens, seq_len + num_prefix_tokens, dim
        )
        padded[:, num_prefix_tokens:, num_prefix_tokens:, :] = pwemb
        return padded

    def run_blocks(
        self,
        inp: torch.Tensor,
        embed: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pwtsr: torch.Tensor | None = None,
        return_full: bool = False,
    ) -> dict[str, Any]:
        out = inp
        other = []
        for layer in self.main:
            out = layer(
                out,
                embed_feats=embed,
                spec_mask=mask,
                biastsr=pwtsr,
                return_full=return_full,
            )
            other.append(out["other"])
            out = out["out"]
        return {"out": self.main_proj(out), "other": other}

    def update_embed(
        self,
        x: torch.Tensor,
        charge: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        emb: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
        mask_rank_embeddings: torch.Tensor | None = None,
        return_mask: bool = False,
        return_full: bool = False,
    ) -> dict[str, Any]:
        mzab_dic = self.encode_mz_ab(x)
        mabemb = mzab_dic["1d"]
        pwemb = mzab_dic["2d"]

        if self.pw_tokenizer is not None:
            pw_tokens = self.pw_tokenizer(torch.split(x, 1, -1)[0])
            mabemb = torch.cat([mabemb, pw_tokens], dim=-1)

        out = self.first(mabemb)
        if emb is None:
            emb = torch.zeros_like(out)
        out = out + self.recyc(emb)

        if token_mask is not None:
            if mask_token is None:
                raise ValueError("mask_token must be provided when token_mask is set.")
            if mask_rank_embeddings is None:
                rank_tokens = torch.zeros_like(out[:1])
            else:
                if mask_rank_embeddings.ndim != 2:
                    raise ValueError(
                        "mask_rank_embeddings must have shape [max_peaks, dim]."
                    )
                if mask_rank_embeddings.shape[1] != out.shape[-1]:
                    raise ValueError(
                        "mask_rank_embeddings width must match embedded peak width."
                    )
                if out.shape[1] > mask_rank_embeddings.shape[0]:
                    raise ValueError(
                        "Peak crop length exceeds the configured iBOT rank-embedding "
                        f"capacity ({out.shape[1]} > {mask_rank_embeddings.shape[0]})."
                    )
                rank_tokens = mask_rank_embeddings[: out.shape[1]].to(
                    device=out.device, dtype=out.dtype
                ).unsqueeze(0)
            # Matches official DINOv2 ViT ``prepare_tokens_with_masks``: replace
            # embedded patches/peaks, not raw input coordinates. The rank term
            # preserves the physical peak slot after m/z and intensity are hidden.
            replacement = mask_token.to(device=out.device, dtype=out.dtype).view(
                1, 1, -1
            ) + rank_tokens
            out = torch.where(token_mask.unsqueeze(-1), replacement, out)

        num_cem_tokens = 0

        if self.atleast1:
            ce_emb = None
            if self.use_charge:
                charge = charge.int()
                ce_emb = self.charge_emb(charge) if ce_emb is None else ce_emb + self.charge_emb(charge)
            if self.use_energy:
                energy_emb = mp.fourier_features(energy, self.running_units, 150.0)
                ce_emb = energy_emb if ce_emb is None else ce_emb + energy_emb
            if self.use_mass:
                mass_emb = mp.fourier_features(mass, 0.001, 10000, self.running_units)
                ce_emb = mass_emb if ce_emb is None else ce_emb + mass_emb

            out = torch.cat([ce_emb.unsqueeze(1), out], dim=1)
            num_cem_tokens += 1

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
            out = torch.cat([cls_tokens, out], dim=1)
            num_cem_tokens += 1

        if self.bias == "pairwise":
            pwemb = self._pad_pw_matrix(pwemb, num_cem_tokens)
            pwemb = self.pw_first(pwemb)
            pwemb = self.pw_seq(pwemb)

        if key_padding_mask is not None:
            cem_mask_pos = torch.tensor([[False] * num_cem_tokens] * out.shape[0]).type_as(
                key_padding_mask
            )
            key_padding_mask = torch.cat([cem_mask_pos, key_padding_mask], dim=1)
            mask = 1e7 * key_padding_mask.type(torch.float32)
        else:
            mask = None

        out = self.run_blocks(out, embed=None, mask=mask, pwtsr=pwemb, return_full=return_full)
        emb = out["out"]

        return {
            "emb": emb,
            "mask": key_padding_mask,
            "num_cem_tokens": num_cem_tokens,
        }

    def forward(
        self,
        mz_int: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        emb: torch.Tensor | None = None,
        its: int | None = None,
        token_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
        mask_rank_embeddings: torch.Tensor | None = None,
        return_mask: bool = False,
        return_full: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if mz_int is None:
            mz_int = x
        if mz_int is None:
            mz_int = kwargs.get("spectra")
        if mz_int is None:
            raise ValueError("Expected mz_int or spectra.")
        its = self.its if its is None else its

        emb = (
            emb
            if emb is not None
            else torch.zeros(mz_int.shape[0], mz_int.shape[1], self.running_units, device=mz_int.device, dtype=mz_int.dtype)
        )

        for _ in range(its):
            output = self.update_embed(
                mz_int,
                charge=charge,
                energy=energy,
                mass=mass,
                key_padding_mask=key_padding_mask,
                emb=emb,
                token_mask=token_mask,
                mask_token=mask_token,
                mask_rank_embeddings=mask_rank_embeddings,
                return_mask=return_mask,
                return_full=return_full,
            )

        return output

    def get_layer_id(self, param_name: str) -> int:
        if (
            param_name.startswith("mz_seq")
            or param_name.startswith("first")
            or param_name.startswith("cls_token")
            or param_name.startswith("mzpw_seq")
            or param_name.startswith("pw_first")
            or param_name.startswith("pw_seq")
            or param_name.startswith("pw_tokenizer")
            or param_name.startswith("charge_emb")
        ):
            return 0
        if param_name.startswith("main."):
            return int(param_name.split(".")[1])
        return self.n_layers


def encoder_tiny(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = False,
    dropout: float = 0.25,
    cls_token: bool = False,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=64,
        ab_units=32,
        subdivide=True,
        running_units=64,
        att_d=64,
        att_h=1,
        depth=9,
        ffn_multiplier=4,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=64,
        pw_run_units=64,
        pw_attention_ch=32,
        pw_attention_h=1,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_base_arch(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = False,
    dropout: float = 0.25,
    cls_token: bool = False,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=1024,
        ab_units=256,
        subdivide=False,
        running_units=512,
        att_d=64,
        att_h=8,
        depth=9,
        ffn_multiplier=2,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=512,
        pw_run_units=64,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_larger(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = None,
    dropout: float = 0.25,
    cls_token: bool = False,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=1024,
        ab_units=256,
        subdivide=True,
        running_units=1024,
        att_d=128,
        att_h=8,
        depth=9,
        ffn_multiplier=2,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=1024,
        pw_run_units=128,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_larger_deeper(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = None,
    dropout: float = 0.25,
    cls_token: bool = False,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=1024,
        ab_units=256,
        subdivide=True,
        running_units=1024,
        att_d=128,
        att_h=8,
        depth=15,
        ffn_multiplier=2,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=1024,
        pw_run_units=128,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_pairwise(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = "pairwise",
    dropout: float = 0.25,
    cls_token: bool = False,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=1024,
        ab_units=256,
        subdivide=False,
        running_units=512,
        att_d=64,
        att_h=8,
        depth=9,
        ffn_multiplier=2,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=128,
        pw_run_units=64,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_pairwise_smaller(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = "pairwise",
    cls_token: bool = False,
    dropout: float = 0.25,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=256,
        ab_units=256,
        subdivide=True,
        running_units=256,
        att_d=32,
        att_h=8,
        depth=9,
        ffn_multiplier=4,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=512,
        pw_run_units=256,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_pairwise_larger(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = "pairwise",
    cls_token: bool = False,
    dropout: float = 0.25,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=1024,
        ab_units=256,
        subdivide=True,
        running_units=1024,
        att_d=128,
        att_h=8,
        depth=9,
        ffn_multiplier=2,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=1024,
        pw_run_units=128,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_pairwise_larger_deeper(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = "pairwise",
    cls_token: bool = False,
    dropout: float = 0.25,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=1024,
        ab_units=256,
        subdivide=True,
        running_units=1024,
        att_d=128,
        att_h=8,
        depth=15,
        ffn_multiplier=2,
        prenorm=False,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=1024,
        pw_run_units=128,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )


def encoder_deeper(
    use_charge: bool = False,
    use_mass: bool = False,
    use_energy: bool = False,
    bias: str | bool | None = False,
    dropout: float = 0.25,
    cls_token: bool = False,
    max_charge: int = 10,
):
    return Encoder(
        norm_type="layer",
        mz_units=1024,
        ab_units=256,
        subdivide=True,
        running_units=512,
        att_d=64,
        att_h=8,
        depth=18,
        ffn_multiplier=4,
        prenorm=True,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=dropout,
        bias=bias,
        gate=False,
        alphabet=False,
        pw_mz_units=512,
        pw_run_units=64,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1,
        cls_token=cls_token,
        max_charge=max_charge,
    )
