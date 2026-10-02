"""Metric-learning components shared by the downstream Lightning task and probes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class MetricProjectionHead(nn.Module):
    """Fresh MLP used for supervised peptide metric learning.

    This deliberately does not reuse a DINO projection/prototype head.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None, output_dim: int = 128) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("Metric projection dimensions must be positive.")
        hidden_dim = input_dim if hidden_dim is None else int(hidden_dim)
        if hidden_dim < 1:
            raise ValueError("Metric projection hidden_dim must be positive.")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.input_dim = int(input_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(values), dim=-1, eps=1e-12)


def labels_to_tensor(labels: torch.Tensor | Sequence[str], device: torch.device) -> torch.Tensor:
    """Map arbitrary batch-local labels to deterministic integer IDs."""
    if isinstance(labels, torch.Tensor):
        return labels.reshape(-1).to(device=device)
    mapping: dict[str, int] = {}
    indices = []
    for label in labels:
        key = str(label)
        if key not in mapping:
            mapping[key] = len(mapping)
        indices.append(mapping[key])
    return torch.tensor(indices, dtype=torch.long, device=device)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor | Sequence[str],
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard SupCon loss over anchors that have at least one positive.

    For each valid anchor ``i``, the denominator is every non-self batch item
    and the numerator averages log-probabilities over all same-label positives.
    """
    if embeddings.ndim != 2:
        raise ValueError("SupCon embeddings must have shape [batch, dimension].")
    if embeddings.shape[0] < 2:
        raise ValueError("SupCon requires at least two batch examples.")
    if temperature <= 0:
        raise ValueError("SupCon temperature must be positive.")

    z = F.normalize(embeddings, dim=-1, eps=1e-12)
    label_ids = labels_to_tensor(labels, z.device)
    if label_ids.numel() != z.shape[0]:
        raise ValueError("SupCon labels must have one entry per embedding.")

    logits = z @ z.T / float(temperature)
    diagonal = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diagonal, float("-inf"))
    positive_mask = label_ids[:, None].eq(label_ids[None, :]) & ~diagonal
    positive_counts = positive_mask.sum(dim=1)
    valid_anchors = positive_counts > 0
    if not torch.any(valid_anchors):
        raise ValueError(
            "SupCon batch contains no valid anchors. Use a positive-aware batch sampler "
            "with at least two spectra per peptide identity."
        )

    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probabilities = log_probabilities.masked_fill(~positive_mask, 0.0)
    per_anchor = -positive_log_probabilities.sum(dim=1) / positive_counts.clamp_min(1)
    return per_anchor[valid_anchors].mean(), valid_anchors.float().mean()


class MetricLearningEmbedder(nn.Module):
    """Encoder global pooling plus the fresh metric projection head."""

    def __init__(self, encoder: nn.Module, pooler: nn.Module, metric_head: MetricProjectionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.metric_head = metric_head
        self.running_units = metric_head.output_dim
        self.use_mass = encoder.use_mass
        self.use_charge = encoder.use_charge
        self.use_energy = encoder.use_energy

    def forward(
        self,
        spectra: torch.Tensor,
        key_padding_mask: torch.Tensor,
        mass: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            spectra,
            key_padding_mask=key_padding_mask,
            mass=mass,
            charge=charge,
        )
        global_representation = self.pooler(encoded["emb"], encoded["mask"])
        return self.metric_head(global_representation)
