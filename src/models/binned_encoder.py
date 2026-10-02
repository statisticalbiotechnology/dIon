import torch
import torch.nn as nn
import numpy as np


class BinnedEncoder(nn.Module):
    def __init__(
        self,
        out_dim=512,
        max_mz=2500.0,
        max_charge=6,
        use_mass=True,
        use_charge=True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.max_mz = max_mz
        self.max_charge = max_charge
        self.use_mass = use_mass
        self.use_charge = use_charge

        self.running_units = out_dim

        n_extra = int(use_mass) + int(use_charge)
        self.n_bins = out_dim - n_extra
        self.register_buffer(
            "bin_edges",
            torch.linspace(0, max_mz, self.n_bins + 1, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        mz_int: torch.Tensor | None = None,
        spectra: torch.Tensor | None = None,
        charge=None,
        energy=None,
        mass=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if mz_int is None:
            mz_int = spectra
        if mz_int is None:
            raise ValueError("Expected mz_int or spectra.")
        B, L, _ = mz_int.shape
        mz = mz_int[:, :, 0].contiguous()
        intens = mz_int[:, :, 1].contiguous()

        # digitize m/z into bin indices
        bin_idx = torch.bucketize(mz, self.bin_edges) - 1
        bin_idx = bin_idx.clamp(min=0, max=self.n_bins - 1)

        # scatter_add to get per-batch histograms
        flat_bins = bin_idx.view(-1)
        flat_batch = torch.arange(B, device=mz_int.device).repeat_interleave(L)
        flat_intens = intens.view(-1)
        binned = torch.zeros(B, self.n_bins, device=mz_int.device)
        binned.index_put_((flat_batch, flat_bins), flat_intens, accumulate=True)
        binned = binned / (binned.max(dim=1, keepdim=True).values + 1e-12)

        feats = [binned]
        if self.use_mass and mass is not None:
            mass_feat = (mass / self.max_mz).unsqueeze(1)
            feats.append(mass_feat)
        if self.use_charge and charge is not None:
            charge_feat = (charge / self.max_charge).unsqueeze(1)
            feats.append(charge_feat)

        out = torch.cat(feats, dim=1)
        return {
            "emb": out.unsqueeze(1),  # [B, 1, D]
            "mask": None,
            "num_cem_tokens": 0,
        }


def encoder_binned_baseline(
    out_dim=512, max_mz=2500.0, max_charge=6, use_mass=True, use_charge=True, **kwargs
):
    return BinnedEncoder(
        out_dim=out_dim,
        max_mz=max_mz,
        max_charge=max_charge,
        use_mass=use_mass,
        use_charge=use_charge,
    )


def encoder_binned_baseline_1024d(
    out_dim=1024, max_mz=2500.0, max_charge=6, use_mass=True, use_charge=True, **kwargs
):
    return BinnedEncoder(
        out_dim=out_dim,
        max_mz=max_mz,
        max_charge=max_charge,
        use_mass=use_mass,
        use_charge=use_charge,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 2, 4
    x = torch.zeros(B, L, 2)
    x[:, :, 0] = torch.rand(B, L) * 100  # m/z
    x[:, :, 1] = torch.rand(B, L)  # intensities
    mass = torch.rand(B) * 200
    charge = torch.randint(1, 4, (B,)).float()

    print("Input m/z:\n", x[:, :, 0])
    print("Input intensities:\n", x[:, :, 1])
    print("Mass:\n", mass)
    print("Charge:\n", charge)

    enc = encoder_binned_baseline(out_dim=8, max_mz=100.0, max_charge=6)
    out = enc(x, mass=mass, charge=charge)
    print("Output shape:", out["emb"].shape)
    print("Output embeddings:\n", out["emb"].squeeze(1))
