from __future__ import annotations

import torch
import torch.nn as nn


class ClsTokenPooler(nn.Module):
    """Pool by selecting the first token."""

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        return embeddings[:, 0, :]


class ProjectedAveragePooler(nn.Module):
    """Pool by averaging projected, non-padding tokens."""

    def __init__(self, embedding_dim: int, freeze_projection: bool = False) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        if freeze_projection:
            for parameter in self.projection.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projection(embeddings)
        non_padding = ~padding_mask
        projected = non_padding.unsqueeze(-1) * projected
        return projected.sum(dim=1) / non_padding.sum(dim=-1, keepdim=True)


class CrossAttentionPooler(nn.Module):
    """Pool with a learned query token and multi-head attention."""

    def __init__(self, embedding_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.cross_attend_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        query = self.cross_attend_token.expand(batch_size, -1, -1)
        attn_output, _ = self.multihead_attn(
            query=query,
            key=embeddings,
            value=embeddings,
            key_padding_mask=padding_mask,
        )
        return attn_output.squeeze(1)


def build_pooler(
    pooling: str,
    embedding_dim: int,
    num_heads: int = 8,
) -> nn.Module:
    if pooling == "cls":
        return ClsTokenPooler()
    if pooling == "average":
        return ProjectedAveragePooler(embedding_dim, freeze_projection=False)
    if pooling == "avg_frozen":
        return ProjectedAveragePooler(embedding_dim, freeze_projection=True)
    if pooling in {"crossattend", "crossattend_cls"}:
        return CrossAttentionPooler(embedding_dim=embedding_dim, num_heads=num_heads)

    raise ValueError(f"Unsupported pooling strategy: {pooling}")
