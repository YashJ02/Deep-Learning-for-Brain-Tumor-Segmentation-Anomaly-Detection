"""PyTorch dataset for multimodal BraTS multiclass segmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


BRATS_MODALITIES = ("flair", "t1", "t1ce", "t2")
BRATS_CLASS_LABELS = (0, 1, 2, 4)
MULTICLASS_INDEX_TO_BRATS_LABEL = {0: 0, 1: 1, 2: 2, 3: 4}


def _load_nifti(path: Path | str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return array, spacing


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


def seg_to_multiclass_indices(segmentation: np.ndarray) -> np.ndarray:
    indices = np.zeros_like(segmentation, dtype=np.int64)
    indices[segmentation == 1] = 1
    indices[segmentation == 2] = 2
    # Some datasets may contain label 3 as a tumor class; map it with label 4 channel.
    indices[np.logical_or(segmentation == 4, segmentation == 3)] = 3
    return indices


def multiclass_indices_to_brats_labels(indices: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(indices, dtype=np.uint8)
    labels[indices == 1] = 1
    labels[indices == 2] = 2
    labels[indices == 3] = 4
    return labels


class BraTSTorchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.target_shape = tuple(int(v) for v in target_shape)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _apply_resize(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = tuple(self.target_shape)
        if image.shape[2:] == expected:
            return image, mask

        image = F.interpolate(image, size=expected, mode="trilinear", align_corners=False)
        mask = F.interpolate(mask, size=expected, mode="nearest")
        return image, mask

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for axis in (2, 3, 4):
            if random.random() < 0.5:
                image = torch.flip(image, dims=[axis])
                mask = torch.flip(mask, dims=[axis])
        return image, mask

    def _load_multimodal_case(self, row: Dict[str, str]) -> tuple[np.ndarray, Tuple[float, float, float]]:
        channels: list[np.ndarray] = []
        first_shape: Tuple[int, ...] | None = None
        first_spacing: Tuple[float, float, float] | None = None

        for modality in BRATS_MODALITIES:
            image_np, spacing = _load_nifti(Path(row[modality]))
            if image_np.ndim != 3:
                raise ValueError(f"Expected 3D volume for {modality}, got shape {image_np.shape}")

            shape = tuple(int(v) for v in image_np.shape)
            if first_shape is None:
                first_shape = shape
                first_spacing = spacing
            else:
                if shape != first_shape:
                    raise ValueError(
                        f"All modalities must share identical shape. {modality}={shape}, expected={first_shape}"
                    )
                if first_spacing is not None and not np.allclose(np.array(spacing), np.array(first_spacing), atol=1e-4):
                    raise ValueError(
                        f"All modalities must share identical spacing. {modality}={spacing}, expected={first_spacing}"
                    )

            channels.append(normalize_nonzero(image_np))

        stacked = np.stack(channels, axis=0).astype(np.float32)
        spacing = first_spacing if first_spacing is not None else (1.0, 1.0, 1.0)
        return stacked, spacing

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.records[index]

        image_np, spacing = self._load_multimodal_case(row)
        seg_np, _ = _load_nifti(Path(row["seg"]))

        mask_np = seg_to_multiclass_indices(seg_np).astype(np.float32)

        image = torch.from_numpy(image_np).unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        image, mask = self._apply_resize(image, mask)
        if self.augment:
            image, mask = self._augment(image, mask)

        final_mask = mask.squeeze(0).squeeze(0).round().clamp(min=0, max=3).long()

        return {
            "image": image.squeeze(0).float(),
            "mask": final_mask,
            "case_id": row["case_id"],
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }
