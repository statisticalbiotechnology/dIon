"""DRAFT CODE, NOT TESTED YET."""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.models.sinusoidal import PositionalEncoder, FloatEncoder


class PeptideEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.0,
        positional_encoder: bool = True,
        max_charge: int = 5,
        use_mass: bool = True,
        use_charge: bool = True,
    ):
        """
        Args:
            n_tokens: Number of tokens in the amino acid vocabulary.
            d_model: Embedding (and model) dimension.
            nhead: Number of attention heads.
            dim_feedforward: Hidden dimension of the feedforward layers.
            n_layers: Number of transformer encoder layers.
            dropout: Dropout probability.
            positional_encoder: If True, use a positional encoder.
            max_charge: Maximum charge state.
            use_mass: If True, use a mass encoder.
            use_charge: If True, use a charge encoder.
        """
        super().__init__()
        self.use_mass = use_mass
        self.use_charge = use_charge

        # Amino acid embedding.
        self.aa_encoder = nn.Embedding(n_tokens + 1, d_model, padding_idx=0)

        # Optional mass/charge encoders.
        if self.use_mass:
            self.mass_encoder = FloatEncoder(d_model)
        if self.use_charge:
            self.charge_encoder = nn.Embedding(max_charge + 1, d_model)

        self.positional_encoder = (
            PositionalEncoder(d_model) if positional_encoder else nn.Identity()
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mass: torch.Tensor = None,
        charge: torch.Tensor = None,
    ):
        """
        Args:
            tokens: LongTensor of shape (B, L) representing token indices.
            mass: FloatTensor of shape (B,) representing precursor masses.
            charge: Tensor of shape (B,) representing precursor charges.

        Returns:
            dict with:
              "emb": Tensor of shape (B, L_total, d_model), where L_total = L + num_extras.
              "mask": BoolTensor of shape (B, L_total), with True indicating padded positions.
        """
        token_emb = self.aa_encoder(tokens)  # (B, L, d_model)
        extras = []
        if self.use_mass and mass is not None:
            extras.append(self.mass_encoder(mass[:, None]))  # (B, 1, d_model)
        if self.use_charge and charge is not None:
            # Adjust charge indices: assume charge in {1,...,max_charge}
            extras.append(
                self.charge_encoder(charge.int() - 1)[:, None, :]
            )  # (B, 1, d_model)
        if extras:
            prefix = torch.cat(extras, dim=1)  # (B, num_extras, d_model)
            x = torch.cat([prefix, token_emb], dim=1)  # (B, num_extras+L, d_model)
        else:
            x = token_emb

        # Create a key padding mask: we assume padded tokens sum to zero.
        mask = ~x.sum(dim=2).bool()  # (B, num_extras+L)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return {"emb": x, "mask": mask}


# --------------------- Dummy Test Code ---------------------
if __name__ == "__main__":
    # Dummy parameters.
    batch_size = 4
    seq_length = 10
    n_tokens = 25  # vocabulary size (e.g. 25 amino acids)
    d_model = 128
    max_charge = 5

    # Create dummy inputs.
    tokens = torch.randint(
        1, n_tokens + 1, (batch_size, seq_length)
    )  # token indices in [1, n_tokens]
    mass = torch.rand(batch_size) * 1000.0  # dummy masses
    charge = torch.randint(
        1, max_charge + 1, (batch_size,)
    )  # dummy charge states between 1 and max_charge

    # Print inputs (full and shapes).
    print("Input tokens:")
    print(tokens)
    print("Tokens shape:", tokens.shape)
    print("Input mass:")
    print(mass)
    print("Mass shape:", mass.shape)
    print("Input charge:")
    print(charge)
    print("Charge shape:", charge.shape)

    # Instantiate the model.
    model = PeptideEncoder(
        n_tokens=n_tokens,
        d_model=d_model,
        max_charge=max_charge,
        use_mass=True,
        use_charge=True,
        positional_encoder=True,
        n_layers=2,  # for testing, use 2 layers
    )

    # Run forward pass.
    out = model(tokens, mass, charge)
    print("\nOutput embedding shape:", out["emb"].shape)
    print("Output embedding:")
    print(out["emb"])
    print("\nOutput mask shape:", out["mask"].shape)
    print("Output mask:")
    print(out["mask"])
