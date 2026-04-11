"""Checkpoint loading and multimodal multiclass inference for trained 3D U-Net models."""

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


def normalize_multimodal_nonzero(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 4:
        raise ValueError(f"Expected multimodal volume shape [C, D, H, W], got {volume.shape}")

    normalized_channels: list[np.ndarray] = []
    for channel in volume:
        mask = channel != 0
        if not np.any(mask):
            normalized_channels.append(np.zeros_like(channel, dtype=np.float32))
            continue

        values = channel[mask]
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:
            std = 1.0

        normalized = (channel - mean) / std
        normalized[~mask] = 0.0
        normalized_channels.append(normalized.astype(np.float32))

    return np.stack(normalized_channels, axis=0).astype(np.float32)


def _extract_target_shape(config: Dict[str, object]) -> Tuple[int, int, int] | None:
    shape = config.get("target_shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 3:
        return tuple(int(value) for value in shape)
    return None


def _multiclass_channel_labels(num_classes: int) -> list[int]:
    if num_classes == 4:
        return [0, 1, 2, 4]
    if num_classes == 3:
        return [1, 2, 4]
    raise ValueError(
        f"Unsupported multiclass output channels: {num_classes}. Expected 3 (labels 1/2/4) or 4 (labels 0/1/2/4)."
    )


def _validate_multimodal_multiclass_config(config: Dict[str, object]) -> None:
    in_channels = int(config.get("in_channels", 4))
    out_channels = int(config.get("out_channels", 4))
    task = str(config.get("task", "multiclass")).strip().lower()

    if in_channels != 4:
        raise RuntimeError(
            "This repository now enforces multimodal checkpoints with in_channels=4. "
            f"Received in_channels={in_channels}."
        )

    if task not in {"", "multiclass"}:
        raise RuntimeError(
            "This repository now enforces multiclass checkpoints only. "
            f"Received task={task!r}."
        )

    if out_channels not in {3, 4}:
        raise RuntimeError(
            "Unsupported multiclass checkpoint out_channels. "
            f"Expected 3 or 4, received out_channels={out_channels}."
        )


def _prepare_input_tensor(
    volume: np.ndarray,
    device: torch.device,
    target_shape: Tuple[int, int, int] | None,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    volume = normalize_multimodal_nonzero(volume)
    original_shape = tuple(int(v) for v in volume.shape[1:])

    tensor = torch.from_numpy(volume).unsqueeze(0).float().to(device)
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
        in_channels=int(config.get("in_channels", 4)),
        out_channels=int(config.get("out_channels", 4)),
        base_channels=int(config.get("base_channels", 16)),
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    result = (model, config)
    if use_cache:
        _MODEL_CACHE[cache_key] = result
    return result


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

    class_labels[max_probability < float(threshold)] = 0

    return class_labels.astype(np.uint8), prob_np, max_probability, channel_labels


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
    _validate_multimodal_multiclass_config(config)

    target_shape = _extract_target_shape(config)
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
        "in_channels": 4,
        "target_shape": list(target_shape) if target_shape is not None else None,
        **_multiclass_summary(class_label_map, class_probabilities, channel_labels),
        "_class_label_map": class_label_map,
    }
    return tumor_mask.astype(bool), details


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
    models: list[tuple[torch.nn.Module, Dict[str, object], str]] = []

    for checkpoint_path in checkpoints:
        model, config = load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
        _validate_multimodal_multiclass_config(config)
        models.append((model, config, checkpoint_path))

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
        "in_channels": 4,
        "ensemble_size": int(len(checkpoints)),
        "checkpoints": checkpoints,
        "members": member_stats,
        **_multiclass_summary(class_label_map, probability_mean, [0, 1, 2, 4]),
        "_class_label_map": class_label_map,
    }
    return tumor_mask.astype(bool), details
