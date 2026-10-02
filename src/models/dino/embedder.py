from typing import Literal, Optional
import torch


class DINOEmbedder(torch.nn.Module):
    """Create a DINO embedding readout from a DINO training wrapper."""

    def __init__(
        self,
        pl_encoder,
        trainable: bool = True,
        embedding_readout: Literal["backbone", "dino_bottleneck", "dino_logits"] = "backbone",
    ) -> None:
        super(DINOEmbedder, self).__init__()
        if embedding_readout not in {"backbone", "dino_bottleneck", "dino_logits"}:
            raise ValueError(
                "embedding_readout must be 'backbone', 'dino_bottleneck', or "
                "'dino_logits', "
                f"got {embedding_readout!r}."
            )
        self.encoder = pl_encoder.get_encoder(trainable=trainable)
        self.pooler = pl_encoder.teacher.pooler
        self.embedding_readout = embedding_readout
        if embedding_readout in {"dino_bottleneck", "dino_logits"}:
            self.dino_head = getattr(
                pl_encoder.teacher,
                "dino_head",
                getattr(pl_encoder.teacher, "head", None),
            )
            if self.dino_head is None:
                raise TypeError(
                    "The DINO teacher does not expose the head required for "
                    "the dino_bottleneck readout."
                )
        if trainable:
            for parameter in self.pooler.parameters():
                parameter.requires_grad = True

        # For compatibility
        self.running_units = (
            self.encoder.running_units
            if embedding_readout == "backbone"
            else (
                self.dino_head.last_layer.in_features
                if embedding_readout == "dino_bottleneck"
                else self.dino_head.last_layer.out_features
            )
        )
        self.use_mass = self.encoder.use_mass
        self.use_charge = self.encoder.use_charge
        self.use_energy = self.encoder.use_energy

    def forward(
        self,
        spectra: torch.Tensor,
        key_padding_mask: torch.Tensor,
        mass: Optional[torch.Tensor] = None,
        charge: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            spectra (torch.Tensor): input tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask (torch.Tensor): mask tensor of shape (batch_size, seq_len)
            mass (torch.Tensor, optional): precursor masses of shape (batch_size, 1)
            charge (torch.Tensor, optional): precursor charges of shape (batch_size, 1)
        Returns:
            torch.Tensor: pooled embedding of shape (batch_size, embed_dim)
        """
        _out = self.encoder(
            spectra,
            key_padding_mask=key_padding_mask,
            mass=mass,
            charge=charge,
        )
        embeds = _out["emb"]
        # These masks accomadate the new shape of embeds due to additional c/e/m or cls tokens
        out_masks = _out["mask"]
        pooled = self.pooler(embeds, out_masks)
        if self.embedding_readout == "backbone":
            return pooled
        if self.embedding_readout == "dino_bottleneck":
            return self.dino_head.forward_bottleneck(pooled)
        return self.dino_head(pooled)
