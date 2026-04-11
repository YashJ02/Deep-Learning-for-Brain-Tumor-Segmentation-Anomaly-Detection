from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import ball


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels].astype(bool)


def load_nifti_volume(file_path: str, modality_index: Optional[int] = None) -> Tuple[np.ndarray, Tuple[float, float, float], int]:
    image = nib.load(file_path)
    data = image.get_fdata(dtype=np.float32)
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

    if data.ndim == 4:
        channel_count = data.shape[3]
        if modality_index is None:
            modality_index = min(3, channel_count - 1)
        modality_index = int(np.clip(modality_index, 0, channel_count - 1))
        volume = data[..., modality_index]
    elif data.ndim == 3:
        modality_index = 0
        volume = data
    else:
        raise ValueError(f"Unsupported NIfTI shape: {data.shape}")

    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    p1, p99 = np.percentile(volume, [1, 99])
    if p99 <= p1:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized = np.clip((volume - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)

    return normalized, spacing, modality_index


def segment_tumor_baseline(volume: np.ndarray) -> np.ndarray:
    smooth = ndi.gaussian_filter(volume, sigma=1.0)
    non_zero = smooth[smooth > 0]

    if non_zero.size == 0:
        return np.zeros_like(smooth, dtype=bool)

    otsu = float(threshold_otsu(non_zero))
    high = float(np.percentile(non_zero, 85))
    threshold = max(otsu, high * 0.70)

    mask = smooth > threshold
    mask = ndi.binary_opening(mask, structure=ball(1))
    mask = ndi.binary_closing(mask, structure=ball(2))
    mask = _remove_small_components(mask, min_size=300)
    mask = ndi.binary_fill_holes(mask)

    if not np.any(mask):
        return mask.astype(bool)

    labels, count = ndi.label(mask)
    if count <= 1:
        return mask.astype(bool)

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_component = int(component_sizes.argmax())

    return (labels == largest_component).astype(bool)


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "best.pt"


def segment_tumor(
    volume: np.ndarray,
    engine: str = "auto",
    checkpoint_path: Optional[str] = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    engine = (engine or "auto").strip().lower()
    if engine not in {"auto", "deep", "baseline"}:
        raise ValueError("engine must be one of: auto, deep, baseline")

    checkpoint = Path(checkpoint_path) if checkpoint_path else default_checkpoint_path()

    if engine in {"auto", "deep"}:
        if checkpoint.exists():
            try:
                from training.inference import segment_with_checkpoint

                deep_mask, details = segment_with_checkpoint(
                    volume=volume,
                    checkpoint_path=checkpoint,
                    threshold=threshold,
                )
                return deep_mask.astype(bool), {
                    "engine": "deep",
                    "checkpoint": str(checkpoint),
                    **details,
                }
            except Exception as exc:
                if engine == "deep":
                    raise RuntimeError(f"Deep model inference failed: {exc}") from exc
        elif engine == "deep":
            raise FileNotFoundError(f"Deep-model checkpoint was not found: {checkpoint}")

    baseline_mask = segment_tumor_baseline(volume)
    return baseline_mask.astype(bool), {
        "engine": "baseline",
        "checkpoint": None,
        "probability_mean": None,
        "probability_max": None,
    }
