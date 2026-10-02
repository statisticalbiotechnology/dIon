import torch
from torch import nn
import torch.nn.functional as F

from src.models.custom.model_parts import fourier_features


def weight_init(module):
    if isinstance(module, TriangleAttention):
        dim = module.c * module.h
        module.Wo.weight = nn.init.normal_(module.Wo.weight, 0.0, 0.3 * dim**-0.5)
        module.Wo.bias = nn.init.zeros_(module.Wo.bias)
    if isinstance(module, TriangleMultiplicative):
        module.Wgg.bias = nn.init.constant_(module.Wgg.bias, -4)
    if isinstance(module, PairTransition):
        dim = module.in_dim * module.n
        module.Wo.weight = nn.init.normal_(module.Wo.weight, 0.0, 0.05 * dim**-0.5)
        module.Wo.bias = nn.init.zeros_(module.Wo.bias)


class QKVAttention(nn.Module):
    def __init__(self, nheads, typ="start"):
        super().__init__()
        self.h = nheads
        self.typ = typ

        if typ == "start":
            op1 = "abcd,abed->abce"
            op2 = "abcd,abde->abce"
            self._bias_adapter = lambda bias: bias[:, None].permute([0, 4, 1, 2, 3])
        elif typ == "end":
            op1 = "abcd,aecd->abce"
            op2 = "abcd,adce->abce"
            self._bias_adapter = lambda bias: bias[..., None, :].permute([0, 4, 3, 2, 1])
        else:
            raise ValueError(f"Unknown attention type: {typ}")

        self._einsum_qk = lambda q, k: torch.einsum(op1, q, k)
        self._einsum_av = lambda a, v: torch.einsum(op2, a, v)

    def forward(self, q, k, v, bias=None, gate=None):
        _, slq1, slq2, _ = q.shape
        _, _, slk2, _ = k.shape
        qk = self._einsum_qk(q, k)
        qk = qk.reshape(-1, self.h, slq1, slq2, slk2)
        if bias is not None:
            qk += self._bias_adapter(bias)

        a = torch.softmax(qk, -1)
        a = a.reshape(-1, slq1, slq2, slk2)
        o = self._einsum_av(a, v)
        if gate is not None:
            gate = (
                gate.reshape(-1, slq1, slq2, -1, self.h)
                .permute([0, 4, 1, 2, 3])
                .reshape(-1, slq1, slq2, gate.shape[-2])
            )
            o *= gate
        return o


class TriangleAttention(nn.Module):
    def __init__(self, in_dim, c=32, h=4, typ="end"):
        super().__init__()
        self.c = c
        self.h = h

        self.attention = QKVAttention(h, typ=typ)
        self.norm = nn.LayerNorm(in_dim)
        self.Wqkv = nn.Linear(in_dim, 3 * c * h, bias=False)
        self.Wbias = nn.Linear(in_dim, h, bias=False)
        self.Wg = nn.Linear(in_dim, c * h)
        self.Wo = nn.Linear(c * h, in_dim)
        self.scale = c**-0.25

    def get_qkv(self, qkv):
        bs, sl1, sl2, ch = qkv.shape
        q, k, v = qkv.split(ch // 3, -1)
        q = q.reshape(bs, sl1, sl2, self.h, self.c).permute([0, 3, 1, 2, 4]).reshape(-1, sl1, sl2, self.c)
        k = k.reshape(bs, sl1, sl2, self.h, self.c).permute([0, 3, 1, 2, 4]).reshape(-1, sl1, sl2, self.c)
        v = v.reshape(bs, sl1, sl2, self.h, self.c).permute([0, 3, 1, 2, 4]).reshape(-1, sl1, sl2, self.c)
        return q, k, v

    def forward(self, zij):
        bs, sl1, sl2, _ = zij.shape
        zij_norm = self.norm(zij)
        qkv = self.Wqkv(zij_norm)
        q, k, v = self.get_qkv(qkv)
        b = self.Wbias(zij_norm)
        g = torch.sigmoid(self.Wg(zij_norm))
        q *= self.scale
        k *= self.scale
        o = self.attention(q, k, v, b, g)
        o = (
            o.reshape(-1, self.h, sl1, sl2, self.c)
            .permute([0, 2, 3, 4, 1])
            .reshape(bs, sl1, sl2, -1)
        )
        zij_out = self.Wo(o)
        return zij_out


class TriangleMultiplicative(nn.Module):
    def __init__(self, in_dim, c=128, typ="out"):
        super().__init__()
        self.c = c
        op = "abcd,aecd->abed" if typ == "out" else "abcd,abed->aced"
        self._einsum_ab = lambda a, b: torch.einsum(op, a, b)

        self.norm1 = nn.LayerNorm(in_dim)
        self.Wag = nn.Linear(in_dim, c)
        self.Wa = nn.Linear(in_dim, c)
        self.Wbg = nn.Linear(in_dim, c)
        self.Wb = nn.Linear(in_dim, c)
        self.norm2 = nn.LayerNorm(c)
        self.Wgg = nn.Linear(in_dim, in_dim)
        self.Wg = nn.Linear(c, in_dim)

    def forward(self, zij):
        zij_norm = self.norm1(zij)
        a_gate = torch.sigmoid(self.Wag(zij_norm))
        a = self.Wa(zij_norm)
        b_gate = torch.sigmoid(self.Wbg(zij_norm))
        b = self.Wb(zij_norm)
        aij = a * a_gate
        bij = b * b_gate
        gij = torch.sigmoid(self.Wgg(zij_norm))
        ab = self._einsum_ab(aij, bij)
        zij_out = gij * self.Wg(self.norm2(ab))
        return zij_out


class PairTransition(nn.Module):
    def __init__(self, in_dim, n=4):
        super().__init__()
        self.in_dim = in_dim
        self.n = n

        self.norm = nn.LayerNorm(in_dim)
        self.Wa = nn.Linear(in_dim, n * in_dim)
        self.Wo = nn.Linear(n * in_dim, in_dim)

    def forward(self, zij):
        zij_norm = self.norm(zij)
        aij = self.Wa(zij_norm)
        zij_out = self.Wo(torch.relu(aij))
        return zij_out


class PairStack(nn.Module):
    def __init__(self, multdict, attdict, ptdict, drop_rate=0.25):
        super().__init__()
        self.drop_rate = drop_rate

        self.TAstart = TriangleAttention(**attdict, typ="start")
        self.TAend = TriangleAttention(**attdict, typ="end")
        self.PT = PairTransition(**ptdict)

        self.dropTAstart = nn.Identity() if drop_rate == 0 else nn.Dropout(drop_rate)
        self.dropTAend = nn.Identity() if drop_rate == 0 else nn.Dropout(drop_rate)

        self.apply(weight_init)

    def forward(self, zij):
        zij = zij + self.dropTAstart(self.TAstart(zij))
        zij = zij + self.dropTAend(self.TAend(zij))
        zij = zij + self.PT(zij)
        return zij


class RelPos(nn.Module):
    def __init__(self, seq_len, cz):
        super().__init__()
        self.cz = cz

        a = torch.arange(seq_len)
        A = a[None] - a[:, None]
        A += seq_len - 1
        self.ResInd = nn.Parameter(A, requires_grad=False)
        self.vbins = 2 * (seq_len - 1) + 1
        self.Wp = nn.Linear(self.vbins, cz)

    def forward(self, seq_len):
        ri = self.ResInd[:seq_len, :seq_len]
        pij = self.Wp(F.one_hot(ri, self.vbins).type(torch.float32))
        return pij


class AbsPosFF(nn.Module):
    def __init__(self, units, max_seq_len=120):
        half_dim = units // 2
        a = torch.arange(max_seq_len)
        emb = fourier_features(a, 1, 1000, half_dim)
        emb_a = emb[:, None].tile(1, max_seq_len, 1)
        emb_b = emb[None].tile(max_seq_len, 1, 1)
        self.pos = torch.cat([emb_a, emb_b], -1)
