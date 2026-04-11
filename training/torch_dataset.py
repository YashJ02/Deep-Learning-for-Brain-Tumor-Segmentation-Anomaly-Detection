"""PyTorch dataset for BraTS binary tumor segmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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


class BraTSBinaryTorchDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, str]],
        modality: str = "t1ce",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
    ) -> None:
        if modality not in {"flair", "t1", "t1ce", "t2"}:
            raise ValueError(f"Unsupported modality: {modality}")

        self.records = list(records)
        self.modality = modality
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.records[index]

        image_np, spacing = _load_nifti(Path(row[self.modality]))
        seg_np, _ = _load_nifti(Path(row["seg"]))

        image_np = normalize_nonzero(image_np)
        mask_np = (seg_np > 0).astype(np.float32)

        image = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        image, mask = self._apply_resize(image, mask)
        if self.augment:
            image, mask = self._augment(image, mask)

        return {
            "image": image.squeeze(0).float(),
            "mask": mask.squeeze(0).float(),
            "case_id": row["case_id"],
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }
