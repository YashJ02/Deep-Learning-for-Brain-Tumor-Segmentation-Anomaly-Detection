"""Checkpoint loading and volume inference for trained 3D U-Net models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import UNet3D
from .utils import resolve_device


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, Dict[str, object]]] = {}


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


def predict_mask_from_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device | str = "auto",
    target_shape: Tuple[int, int, int] | None = None,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device

    volume = normalize_nonzero(volume)
    original_shape = tuple(int(v) for v in volume.shape)

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(device)
    if target_shape is not None and tuple(tensor.shape[2:]) != tuple(target_shape):
        tensor = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(tensor)
            probs = torch.sigmoid(logits)

    if tuple(probs.shape[2:]) != original_shape:
        probs = F.interpolate(probs, size=original_shape, mode="trilinear", align_corners=False)

    prob_np = probs.squeeze().detach().cpu().numpy().astype(np.float32)
    mask_np = (prob_np >= float(threshold)).astype(bool)

    return mask_np, prob_np


def segment_with_checkpoint(
    volume: np.ndarray,
    checkpoint_path: str | Path,
    device: torch.device | str = "auto",
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    model, config = load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
    target_shape = _extract_target_shape(config)
    mask, probability = predict_mask_from_volume(
        model=model,
        volume=volume,
        device=device,
        target_shape=target_shape,
        threshold=threshold,
        use_amp=use_amp,
    )

    details: Dict[str, object] = {
        "target_shape": list(target_shape) if target_shape is not None else None,
        "probability_mean": float(np.mean(probability)),
        "probability_max": float(np.max(probability)),
    }
    return mask, details
