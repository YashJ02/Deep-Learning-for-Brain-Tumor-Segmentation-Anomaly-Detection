"""Checkpoint loading and volume inference for trained 3D U-Net models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import UNet3D
from .utils import resolve_device


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, Dict[str, object]]] = {}

BRATS_CLASS_LABELS = (1, 2, 4)
_INDEX_TO_BRATS_4CLASS = np.array([0, 1, 2, 4], dtype=np.uint8)
_INDEX_TO_BRATS_3CLASS = np.array([1, 2, 4], dtype=np.uint8)


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32)

    values = volume[mask]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        std = 1.0

    normalized = (volume - mean) / std
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def _extract_target_shape(config: Dict[str, object]) -> Tuple[int, int, int] | None:
    shape = config.get("target_shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 3:
        return tuple(int(value) for value in shape)
    return None


def _infer_task_from_config(config: Dict[str, object]) -> str:
    task = str(config.get("task", "")).strip().lower()
    if task in {"binary", "multiclass"}:
        return task

    out_channels = int(config.get("out_channels", 1))
    return "multiclass" if out_channels > 1 else "binary"


def _multiclass_channel_labels(num_classes: int) -> list[int]:
    if num_classes == 4:
        return [0, 1, 2, 4]
    if num_classes == 3:
        return [1, 2, 4]
    raise ValueError(
        f"Unsupported multiclass output channels: {num_classes}. Expected 3 (labels 1/2/4) or 4 (labels 0/1/2/4)."
    )


def _prepare_input_tensor(
    volume: np.ndarray,
    device: torch.device,
    target_shape: Tuple[int, int, int] | None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    volume = normalize_nonzero(volume)
    original_shape = tuple(int(v) for v in volume.shape)

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)
    if target_shape is not None and tuple(tensor.shape[2:]) != tuple(target_shape):
        tensor = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)

    return tensor, original_shape


def _restore_spatial_shape(tensor: torch.Tensor, original_shape: tuple[int, int, int]) -> torch.Tensor:
    if tuple(tensor.shape[2:]) != original_shape:
        tensor = F.interpolate(tensor, size=original_shape, mode="trilinear", align_corners=False)
    return tensor


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "auto",
    use_cache: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    checkpoint_path = Path(checkpoint_path).resolve()
    cache_key = (str(checkpoint_path), str(device))

    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
        config = dict(checkpoint.get("config", {}))
    elif isinstance(checkpoint, dict):
        model_state = checkpoint
        config = {}
    else:
        raise RuntimeError("Checkpoint format is not supported.")

    model = UNet3D(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 16)),
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    result = (model, config)
    if use_cache:
        _MODEL_CACHE[cache_key] = result
    return result


def predict_binary_mask_from_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device | str = "auto",
    target_shape: Tuple[int, int, int] | None = None,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device

    tensor, original_shape = _prepare_input_tensor(volume, device=device, target_shape=target_shape)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(tensor)
            probabilities = torch.sigmoid(logits)

    probabilities = _restore_spatial_shape(probabilities, original_shape)

    prob_np = probabilities.squeeze().detach().cpu().numpy().astype(np.float32)
    mask_np = (prob_np >= float(threshold)).astype(bool)
    return mask_np, prob_np


def predict_multiclass_from_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device | str = "auto",
    target_shape: Tuple[int, int, int] | None = None,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device

    tensor, original_shape = _prepare_input_tensor(volume, device=device, target_shape=target_shape)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)

    probabilities = _restore_spatial_shape(probabilities, original_shape)

    prob_np = probabilities.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if prob_np.ndim != 4:
        raise RuntimeError(f"Expected multiclass probability shape [C, D, H, W], got {prob_np.shape}")

    channel_labels = _multiclass_channel_labels(prob_np.shape[0])
    class_indices = np.argmax(prob_np, axis=0).astype(np.uint8)
    max_probability = np.max(prob_np, axis=0).astype(np.float32)

    if prob_np.shape[0] == 4:
        class_labels = _INDEX_TO_BRATS_4CLASS[class_indices]
    else:
        class_labels = _INDEX_TO_BRATS_3CLASS[class_indices]
        class_labels[max_probability < float(threshold)] = 0

    # For models with explicit background channel, low-confidence voxels are still coerced to background.
    class_labels[max_probability < float(threshold)] = 0

    return class_labels.astype(np.uint8), prob_np, max_probability, channel_labels


def predict_mask_from_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device | str = "auto",
    target_shape: Tuple[int, int, int] | None = None,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # Backward-compatible binary prediction helper.
    return predict_binary_mask_from_volume(
        model=model,
        volume=volume,
        device=device,
        target_shape=target_shape,
        threshold=threshold,
        use_amp=use_amp,
    )


def _multiclass_summary(
    class_label_map: np.ndarray,
    probabilities: np.ndarray,
    channel_labels: Sequence[int],
) -> dict[str, object]:
    class_voxels: dict[str, int] = {
        str(label): int(np.count_nonzero(class_label_map == label))
        for label in BRATS_CLASS_LABELS
    }

    class_probability_mean: dict[str, float] = {}
    class_probability_max: dict[str, float] = {}

    for channel_index, label in enumerate(channel_labels):
        if label not in BRATS_CLASS_LABELS:
            continue
        channel_prob = probabilities[channel_index]
        class_probability_mean[str(label)] = float(np.mean(channel_prob))
        class_probability_max[str(label)] = float(np.max(channel_prob))

    tumor_probability = np.zeros_like(class_label_map, dtype=np.float32)
    for channel_index, label in enumerate(channel_labels):
        if label in BRATS_CLASS_LABELS:
            tumor_probability = np.maximum(tumor_probability, probabilities[channel_index])

    return {
        "class_labels": list(BRATS_CLASS_LABELS),
        "class_voxels": class_voxels,
        "class_probability_mean": class_probability_mean,
        "class_probability_max": class_probability_max,
        "probability_mean": float(np.mean(tumor_probability)),
        "probability_max": float(np.max(tumor_probability)),
    }


def segment_with_checkpoint(
    volume: np.ndarray,
    checkpoint_path: str | Path,
    device: torch.device | str = "auto",
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    model, config = load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
    target_shape = _extract_target_shape(config)
    task = _infer_task_from_config(config)

    if task == "multiclass":
        class_label_map, class_probabilities, _, channel_labels = predict_multiclass_from_volume(
            model=model,
            volume=volume,
            device=device,
            target_shape=target_shape,
            threshold=threshold,
            use_amp=use_amp,
        )
        tumor_mask = class_label_map > 0
        details: Dict[str, object] = {
            "task": "multiclass",
            "target_shape": list(target_shape) if target_shape is not None else None,
            **_multiclass_summary(class_label_map, class_probabilities, channel_labels),
            "_class_label_map": class_label_map,
        }
        return tumor_mask.astype(bool), details

    mask, probability = predict_binary_mask_from_volume(
        model=model,
        volume=volume,
        device=device,
        target_shape=target_shape,
        threshold=threshold,
        use_amp=use_amp,
    )

    details = {
        "task": "binary",
        "target_shape": list(target_shape) if target_shape is not None else None,
        "probability_mean": float(np.mean(probability)),
        "probability_max": float(np.max(probability)),
    }
    return mask, details


def segment_with_checkpoint_ensemble(
    volume: np.ndarray,
    checkpoint_paths: Sequence[str | Path],
    device: torch.device | str = "auto",
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths cannot be empty")

    checkpoints = [str(Path(path).resolve()) for path in checkpoint_paths]
    member_stats = []

    tasks: list[str] = []
    models: list[tuple[torch.nn.Module, Dict[str, object], str]] = []
    for checkpoint_path in checkpoints:
        model, config = load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
        task = _infer_task_from_config(config)
        tasks.append(task)
        models.append((model, config, checkpoint_path))

    if len(set(tasks)) != 1:
        raise RuntimeError(f"Ensemble checkpoints have mixed tasks: {tasks}")

    task = tasks[0]

    if task == "multiclass":
        canonical_probs = []

        for model, config, checkpoint_path in models:
            target_shape = _extract_target_shape(config)
            class_label_map, class_probabilities, max_probability, channel_labels = predict_multiclass_from_volume(
                model=model,
                volume=volume,
                device=device,
                target_shape=target_shape,
                threshold=threshold,
                use_amp=use_amp,
            )

            converted = np.zeros((4, *class_label_map.shape), dtype=np.float32)
            label_to_index = {0: 0, 1: 1, 2: 2, 4: 3}
            for channel_index, label in enumerate(channel_labels):
                if label in label_to_index:
                    converted[label_to_index[label]] = class_probabilities[channel_index]

            canonical_probs.append(converted)
            member_stats.append(
                {
                    "checkpoint": checkpoint_path,
                    "target_shape": list(target_shape) if target_shape is not None else None,
                    "probability_mean": float(np.mean(max_probability)),
                    "probability_max": float(np.max(max_probability)),
                }
            )

        probability_stack = np.stack(canonical_probs, axis=0)
        probability_mean = np.mean(probability_stack, axis=0).astype(np.float32)

        class_indices = np.argmax(probability_mean, axis=0).astype(np.uint8)
        max_probability = np.max(probability_mean, axis=0).astype(np.float32)
        class_indices[max_probability < float(threshold)] = 0
        class_label_map = _INDEX_TO_BRATS_4CLASS[class_indices]
        tumor_mask = class_label_map > 0

        details = {
            "task": "multiclass",
            "ensemble_size": int(len(checkpoints)),
            "checkpoints": checkpoints,
            "members": member_stats,
            **_multiclass_summary(class_label_map, probability_mean, [0, 1, 2, 4]),
            "_class_label_map": class_label_map,
        }
        return tumor_mask.astype(bool), details

    probabilities = []
    for model, config, checkpoint_path in models:
        target_shape = _extract_target_shape(config)
        _, probability = predict_binary_mask_from_volume(
            model=model,
            volume=volume,
            device=device,
            target_shape=target_shape,
            threshold=threshold,
            use_amp=use_amp,
        )
        probabilities.append(probability)
        member_stats.append(
            {
                "checkpoint": checkpoint_path,
                "target_shape": list(target_shape) if target_shape is not None else None,
                "probability_mean": float(np.mean(probability)),
                "probability_max": float(np.max(probability)),
            }
        )

    probability_stack = np.stack(probabilities, axis=0)
    probability_mean = np.mean(probability_stack, axis=0).astype(np.float32)
    mask = probability_mean >= float(threshold)

    details: Dict[str, object] = {
        "task": "binary",
        "ensemble_size": int(len(checkpoints)),
        "checkpoints": checkpoints,
        "members": member_stats,
        "probability_mean": float(np.mean(probability_mean)),
        "probability_max": float(np.max(probability_mean)),
    }

    return mask.astype(bool), details
