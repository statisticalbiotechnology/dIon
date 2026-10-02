import math
from typing import Iterable

import torch
from torch import nn

TWOPI = 2.0 * math.pi


def fourier_features(t: torch.Tensor, min_lam: float, max_lam: float, embedsz: int) -> torch.Tensor:
    x = torch.arange(embedsz // 2, device=t.device, dtype=torch.float32)
    x = x / float(embedsz // 2 - 1)
    denom = (min_lam / TWOPI) * (max_lam / min_lam) ** x
    embed = t[..., None] / denom[None]
    return torch.cat([embed.sin(), embed.cos()], dim=-1)


def subdivide_float(x: torch.Tensor) -> torch.Tensor:
    mul = -1 * (x < 0).type(torch.float32) + (x >= 0).type(torch.float32)
    x = x.abs()
    a = x.floor_divide(100)
    b = (x - a * 100).floor_divide(1)
    x_frac = ((x - x.floor_divide(1)) * 10000).round()
    c = x_frac.floor_divide(100)
    d = (x_frac - c * 100).floor_divide(1)
    return mul[..., None] * torch.cat(
        [a[..., None], b[..., None], c[..., None], d[..., None]], -1
    )


def delta_tensor(mz: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
    return mz[..., None] - mz[:, None] + shift


class BatchTorch1d(nn.Module):
    def __init__(self, units: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(-1, -2)).transpose(-1, -2)


def get_norm_type(string: str):
    if string.lower() == "layer":
        return nn.LayerNorm
    if string.lower() == "batch":
        return BatchTorch1d
    raise ValueError(f"Unknown norm type: {string}")


class QKVAttention(nn.Module):
    def __init__(self, heads: int, dim: int, sl: int | None = None, is_relpos: bool = False, max_rel_dist: int | None = None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.sl = sl
        self.is_relpos = is_relpos
        self.maxd = max_rel_dist

        self.scale = dim**-0.5
        if is_relpos:
            assert sl is not None
            self.maxd = sl if max_rel_dist is None else (max_rel_dist - 1 if max_rel_dist == sl else max_rel_dist)
            self.ak = self.build_relpos_tensor(sl, self.maxd)
            self.av = self.build_relpos_tensor(sl, self.maxd)

    def build_relpos_tensor(self, seq_len: int, maxd: int | None = None) -> torch.Tensor:
        maxd = seq_len - 1 if maxd is None else (maxd - 1 if maxd == seq_len else maxd)
        a = torch.arange(seq_len, dtype=torch.int32)
        b = torch.arange(seq_len, dtype=torch.int32)
        relpos = a[:, None] - b[None]
        tsr = (
            torch.zeros(2 * seq_len - 1, self.dim)
            .normal_(0, seq_len**-0.5)
            .type(torch.float32)
        )
        relpos = relpos.clamp(-maxd, maxd)
        relpos += maxd
        relpos_tsr = tsr[relpos]
        relpos_tsr.requires_grad = True
        return relpos_tsr

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        return_full: bool = False,
    ):
        _, sl, _ = Q.shape
        qk = self.scale * torch.einsum("abc,adc->abd", Q, K)
        if self.is_relpos:
            qk += torch.einsum("abc,bec->abe", Q, self.ak)
        qk = qk.reshape(-1, self.heads, sl, K.shape[1])
        if bias is not None:
            qk += bias

        if mask is None:
            mask = torch.zeros_like(qk)
        elif len(mask.shape) == 2:
            mask = mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = mask[:, None]

        weights = torch.softmax(qk - mask, dim=-1)
        weights = weights.reshape(-1, sl, V.shape[1])

        att = torch.einsum("abc,acd->abd", weights, V)
        if self.is_relpos:
            att += torch.einsum("abc,bcd->abd", weights, self.av)

        if gate is not None:
            att = att * gate

        other = [qk, weights, att] if return_full else None
        return att, other


class BaseAttentionLayer(nn.Module):
    def __init__(self, indim, d, h, out_units=None, gate=False, dropout=0, alphabet=False):
        super().__init__()
        self.indim = indim
        self.d = d
        self.h = h
        self.out_units = indim if out_units is None else out_units
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.alphabet = alphabet

        self.attention_layer = QKVAttention(h, d)

        self.Wo = nn.Linear(d * h, self.out_units, bias=True)
        self.Wo.weight = nn.Parameter(
            nn.init.normal_(torch.empty(self.Wo.weight.shape), 0.0, 0.3 * (h * d) ** -0.5)
        )

        self.shortcut = nn.Identity() if self.out_units == indim else nn.Linear(indim, out_units)

        self.gate = gate
        if gate:
            self.Wg = nn.Linear(indim, d * h)

        if alphabet:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)


class SelfAttention(BaseAttentionLayer):
    def __init__(
        self,
        indim,
        d,
        h,
        out_units=None,
        gate=False,
        bias=False,
        bias_in_units=None,
        modulator=False,
        dropout=0,
        alphabet=False,
    ):
        super().__init__(
            indim=indim,
            d=d,
            h=h,
            out_units=out_units,
            gate=gate,
            dropout=dropout,
            alphabet=alphabet,
        )

        self.qkv = nn.Linear(indim, 3 * d * h, bias=True)

        self.bias = bias
        if bias == "pairwise":
            self.Wpw = nn.Linear(bias_in_units, h)
        elif bias == "regular":
            self.Wb = nn.Linear(indim, h)

        self.modulator = modulator
        if modulator:
            self.alphaq = nn.Parameter(torch.tensor(0.0))
            self.alphak = nn.Parameter(torch.tensor(0.0))
            self.alphav = nn.Parameter(torch.tensor(0.0))

    def get_qkv(self, qkv: torch.Tensor):
        bs, sl, units = qkv.shape
        q, k, v = qkv.split(units // 3, -1)
        q = q.reshape(-1, sl, self.d, self.h).permute([0, 3, 1, 2]).reshape(-1, sl, self.d)
        k = k.reshape(-1, sl, self.d, self.h).permute([0, 3, 1, 2]).reshape(-1, sl, self.d)
        v = v.reshape(-1, sl, self.d, self.h).permute([0, 3, 1, 2]).reshape(-1, sl, self.d)
        if self.modulator:
            q = q * torch.sigmoid(self.alphaq)
            k = k * torch.sigmoid(self.alphak)
            v = v * torch.sigmoid(self.alphav)
        return q, k, v

    def forward(self, x: torch.Tensor, mask=None, biastsr=None, return_full=False):
        bs, sl, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = self.get_qkv(qkv)

        if self.bias == "regular":
            b = self.Wb(x)[:, None]
            b = b.permute([0, 3, 1, 2])
        elif self.bias == "pairwise":
            b = self.Wpw(biastsr)
            b = b.permute([0, 3, 1, 2])
        else:
            b = None

        if self.gate:
            g = torch.sigmoid(self.Wg(x))
            g = g.reshape(bs, sl, self.d, self.h).permute([0, 3, 1, 2]).reshape(bs * self.h, sl, self.d)
        else:
            g = None

        att, other = self.attention_layer(q, k, v, mask, bias=b, gate=g, return_full=return_full)
        att = att.reshape(-1, self.h, sl, self.d)
        att = att.permute([0, 2, 3, 1]).reshape(-1, sl, self.d * self.h)
        resid = self.Wo(att)

        if self.alphabet:
            output = self.alpha * self.shortcut(x) + self.beta * self.drop(resid)
        else:
            output = self.shortcut(x) + self.drop(resid)

        extra = [q, k, v] + other + [resid] if return_full else None
        return {"out": output, "other": extra}


class CrossAttention(BaseAttentionLayer):
    def __init__(self, indim, kvindim, d, h, out_units=None, dropout=0, alphabet=False):
        super().__init__(
            indim=indim,
            d=d,
            h=h,
            out_units=out_units,
            dropout=dropout,
            alphabet=alphabet,
        )
        self.Wq = nn.Linear(indim, d * h, bias=False)
        self.Wkv = nn.Linear(kvindim, 2 * d * h, bias=False)

    def get_qkv(self, q: torch.Tensor, kv: torch.Tensor):
        bs, sl, _ = q.shape
        _, sl2, kvunits = kv.shape
        q = q.reshape(bs, sl, self.d, self.h).permute([0, 3, 1, 2]).reshape(-1, sl, self.d)
        k, v = kv.split(kvunits // 2, -1)
        k = k.reshape(bs, sl2, self.d, self.h).permute([0, 3, 1, 2]).reshape(-1, sl2, self.d)
        v = v.reshape(bs, sl2, self.d, self.h).permute([0, 3, 1, 2]).reshape(-1, sl2, self.d)
        return q, k, v

    def forward(self, q_feats: torch.Tensor, kv_feats: torch.Tensor, mask=None):
        _, slq, _ = q_feats.shape
        q = self.Wq(q_feats)
        kv = self.Wkv(kv_feats)
        q, k, v = self.get_qkv(q, kv)
        att, _ = self.attention_layer(q, k, v, mask)
        att = att.reshape(-1, self.h, slq, self.d).permute([0, 2, 1, 3]).reshape(-1, slq, self.h * self.d)
        resid = self.Wo(att)

        if self.alphabet:
            out = self.alpha * self.shortcut(q_feats) + self.beta * self.drop(resid)
        else:
            out = self.shortcut(q_feats) + self.drop(resid)
        return out


class FFN(nn.Module):
    def __init__(self, indim, unit_multiplier=1, out_units=None, dropout=0, alphabet=False):
        super().__init__()
        self.indim = indim
        self.mult = unit_multiplier
        self.out_units = indim if out_units is None else out_units
        self.alphabet = alphabet

        self.W1 = nn.Linear(indim, indim * self.mult)
        self.W2 = nn.Linear(indim * self.mult, self.out_units, bias=False)

        shape = self.W2.weight.shape
        self.W2.weight = nn.Parameter(
            nn.init.normal_(torch.empty(shape), 0.0, 0.3 * (indim * self.mult) ** -0.5)
        )

        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)

        if alphabet:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x: torch.Tensor, embed=None, return_full=False):
        out1 = self.W1(x)
        out2 = torch.relu(out1 + (0 if embed is None else embed))
        out3 = self.W2(out2)

        if self.alphabet:
            out = self.alpha * x + self.beta * self.drop(out3)
        else:
            out = x + self.drop(out3)

        other = [out1, out3] if return_full else None
        return {"out": out, "other": other}


class TransBlock(nn.Module):
    def __init__(
        self,
        attention_dict,
        ffn_dict,
        norm_type="layer",
        prenorm=True,
        is_embed=False,
        embed_indim=256,
        preembed=True,
        is_cross=False,
        kvindim=256,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.mult = ffn_dict["unit_multiplier"]
        self.prenorm = prenorm
        self.is_embed = is_embed
        self.preembed = preembed
        self.is_cross = is_cross

        if preembed:
            self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        norm = get_norm_type(norm_type)

        indim = attention_dict["indim"]
        self.norm1 = norm(indim)
        self.norm2 = norm(ffn_dict["indim"])
        self.selfattention = SelfAttention(**attention_dict)
        if is_cross:
            cross_dict = attention_dict.copy()
            if "pairwise_bias" in cross_dict:
                cross_dict.pop("pairwise_bias")
            if "bias_in_units" in cross_dict:
                cross_dict.pop("bias_in_units")
            cross_dict["kvindim"] = kvindim
            self.crossnorm = norm(indim)
            self.crossattention = CrossAttention(**cross_dict)
        self.ffn = FFN(**ffn_dict)

        if self.is_embed:
            assert isinstance(embed_indim, int)
            units = indim if self.preembed else indim * self.mult
            self.embed = nn.Linear(embed_indim, units)

    def forward(
        self,
        x,
        kv_feats=None,
        embed_feats=None,
        spec_mask=None,
        seq_mask=None,
        biastsr=None,
        return_full=False,
    ):
        selfmask = seq_mask if self.is_cross else spec_mask
        emb = self.embed(embed_feats)[:, None, :] if self.is_embed else 0

        out = x + self.alpha * emb if self.preembed else x
        out = self.norm1(out) if self.prenorm else out
        outsa = self.selfattention(out, selfmask, biastsr, return_full=return_full)
        out = outsa["out"]
        if self.is_cross:
            out = self.crossnorm(out) if self.prenorm else out
            out = self.crossattention(out, kv_feats, spec_mask)
            out = out if self.prenorm else self.crossnorm(out)
        out = self.norm2(out) if self.prenorm else self.norm1(out)
        outffn = self.ffn(out, None, return_full=return_full) if self.preembed else self.ffn(out, emb, return_full=return_full)
        out = outffn["out"]
        out = out if self.prenorm else self.norm2(out)

        other = outsa["other"] + outffn["other"] + [out] if return_full else None
        return {"out": out, "other": other}


class ActModule(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.act = activation

    def forward(self, x):
        return self.act(x)


class AbsPosFF(nn.Module):
    def __init__(self, units, max_seq_len=120):
        half_dim = units // 2
        a = torch.arange(max_seq_len)
        emb = fourier_features(a, 1, 1000, half_dim)
        emb_a = emb[:, None].tile(1, max_seq_len, 1)
        emb_b = emb[None].tile(max_seq_len, 1, 1)
        self.pos = torch.cat([emb_a, emb_b], -1)
