from __future__ import annotations

from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_closing, binary_opening, remove_small_holes, remove_small_objects


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
    mask = binary_opening(mask, footprint=ball(1))
    mask = binary_closing(mask, footprint=ball(2))
    mask = remove_small_objects(mask, min_size=300)
    mask = remove_small_holes(mask, area_threshold=400)

    if not np.any(mask):
        return mask.astype(bool)

    labels, count = ndi.label(mask)
    if count <= 1:
        return mask.astype(bool)

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_component = int(component_sizes.argmax())

    return (labels == largest_component).astype(bool)
