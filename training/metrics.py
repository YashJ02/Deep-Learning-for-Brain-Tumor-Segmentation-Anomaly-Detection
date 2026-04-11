"""Metrics for binary and multiclass segmentation model quality."""

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


def multiclass_dice_iou_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
    include_background: bool = False,
) -> tuple[float, float, dict[int, dict[str, float]]]:
    if logits.ndim != 5:
        raise ValueError("multiclass_dice_iou_from_logits expects logits with shape [B, C, D, H, W]")
    if targets.ndim != 4:
        raise ValueError("multiclass_dice_iou_from_logits expects targets with shape [B, D, H, W]")

    predictions = torch.argmax(logits, dim=1)
    num_classes = int(logits.shape[1])

    class_indices = list(range(num_classes))
    if not include_background and num_classes > 1:
        class_indices = class_indices[1:]

    if not class_indices:
        return 0.0, 0.0, {}

    dice_values: list[float] = []
    iou_values: list[float] = []
    per_class: dict[int, dict[str, float]] = {}

    for class_index in class_indices:
        pred_mask = (predictions == class_index).float()
        target_mask = (targets == class_index).float()

        dims = (1, 2, 3)
        intersection = torch.sum(pred_mask * target_mask, dim=dims)
        pred_sum = torch.sum(pred_mask, dim=dims)
        target_sum = torch.sum(target_mask, dim=dims)
        union = pred_sum + target_sum - intersection

        dice_tensor = _safe_division(2.0 * intersection, pred_sum + target_sum, epsilon)
        iou_tensor = _safe_division(intersection, union, epsilon)

        dice = float(dice_tensor.mean().item())
        iou = float(iou_tensor.mean().item())

        per_class[int(class_index)] = {"dice": dice, "iou": iou}
        dice_values.append(dice)
        iou_values.append(iou)

    dice_macro = float(sum(dice_values) / len(dice_values))
    iou_macro = float(sum(iou_values) / len(iou_values))
    return dice_macro, iou_macro, per_class
