"""Metrics for binary segmentation model quality."""

from __future__ import annotations

import torch


def _safe_division(numerator: torch.Tensor, denominator: torch.Tensor, epsilon: float) -> torch.Tensor:
    return (numerator + epsilon) / (denominator + epsilon)


def binary_dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, epsilon: float = 1e-6) -> float:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    dims = (1, 2, 3, 4)
    intersection = torch.sum(predictions * targets, dim=dims)
    total = torch.sum(predictions, dim=dims) + torch.sum(targets, dim=dims)
    dice = _safe_division(2.0 * intersection, total, epsilon)
    return float(dice.mean().item())


def binary_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, epsilon: float = 1e-6) -> float:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    dims = (1, 2, 3, 4)
    intersection = torch.sum(predictions * targets, dim=dims)
    union = torch.sum(predictions, dim=dims) + torch.sum(targets, dim=dims) - intersection
    iou = _safe_division(intersection, union, epsilon)
    return float(iou.mean().item())
