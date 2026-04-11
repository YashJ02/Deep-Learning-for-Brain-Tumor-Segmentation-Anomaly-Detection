"""Loss functions for binary and multiclass 3D segmentation."""

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


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
    include_background: bool = False,
) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("multiclass_dice_loss expects logits with shape [B, C, D, H, W]")
    if targets.ndim != 4:
        raise ValueError("multiclass_dice_loss expects targets with shape [B, D, H, W]")

    num_classes = logits.shape[1]
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long().clamp(min=0, max=num_classes - 1), num_classes=num_classes)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()

    if not include_background and num_classes > 1:
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]

    dims = (0, 2, 3, 4)
    intersection = torch.sum(probabilities * one_hot, dim=dims)
    denominator = torch.sum(probabilities, dim=dims) + torch.sum(one_hot, dim=dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def multiclass_ce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float = 0.5,
    include_background: bool = False,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets.long())
    dice = multiclass_dice_loss(
        logits,
        targets,
        include_background=include_background,
    )
    return (ce_weight * ce) + ((1.0 - ce_weight) * dice)
