"""Dense-token Gram anchoring for dIon refinement.

Adapted from the Gram-anchoring refinement introduced in DINOv3:
https://github.com/facebookresearch/dinov3
"""

import torch
import torch.nn.functional as F


def per_spectrum_gram_mse(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-spectrum MSE between FP32 cosine-similarity Gram matrices.

    Each spectrum contributes one equally weighted loss, regardless of its peak
    count. ``padding_mask`` refers only to physical peak positions; precursor
    and CLS/CEM tokens must already have been removed by the caller.
    """
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            "Student and Gram-teacher token tensors must have identical shapes; "
            f"got {tuple(student_tokens.shape)} and {tuple(teacher_tokens.shape)}."
        )
    if student_tokens.ndim != 3:
        raise ValueError("Gram anchoring expects [batch, peaks, channels] tokens.")
    if padding_mask.shape != student_tokens.shape[:2]:
        raise ValueError(
            "padding_mask must have shape [batch, peaks] matching token tensors."
        )

    losses = []
    for student, teacher, padded in zip(student_tokens, teacher_tokens, padding_mask):
        valid = ~padded.bool()
        if not valid.any():
            continue
        student_normalized = F.normalize(student[valid].float(), dim=-1)
        teacher_normalized = F.normalize(teacher[valid].float(), dim=-1)
        losses.append(
            F.mse_loss(
                student_normalized @ student_normalized.transpose(0, 1),
                teacher_normalized @ teacher_normalized.transpose(0, 1),
            )
        )

    if not losses:
        # Keep a differentiable zero for defensive handling of malformed batches.
        return student_tokens.sum() * 0.0
    return torch.stack(losses).mean()
