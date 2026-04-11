"""Loss functions for binary 3D tumor segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3, 4)
    intersection = torch.sum(probabilities * targets, dim=dims)
    denominator = torch.sum(probabilities, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return (bce_weight * bce) + ((1.0 - bce_weight) * dice)
