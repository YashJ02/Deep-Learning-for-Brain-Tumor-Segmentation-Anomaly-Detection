# -----yash jain------
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


BRATS_CLASS_DEFINITIONS = {
    1: {
        "key": "necrotic_non_enhancing_core",
        "name": "Necrotic/Non-Enhancing Core",
    },
    2: {
        "key": "peritumoral_edema",
        "name": "Peritumoral Edema",
    },
    4: {
        "key": "enhancing_tumor",
        "name": "Enhancing Tumor",
    },
}


def compute_tumor_metrics(mask: np.ndarray, spacing_mm: Tuple[float, float, float]) -> Dict[str, float]:
    voxel_count = int(np.count_nonzero(mask))
    total_voxels = int(mask.size)

    if voxel_count == 0:
        return {
            "detected": False,
            "voxel_count": 0,
            "occupancy_percent": 0.0,
            "volume_mm3": 0.0,
            "volume_ml": 0.0,
            "equivalent_diameter_mm": 0.0,
            "bbox_min": [0, 0, 0],
            "bbox_max": [0, 0, 0],
            "extent_mm": [0.0, 0.0, 0.0],
            "centroid_voxel": [0.0, 0.0, 0.0],
            "centroid_mm": [0.0, 0.0, 0.0],
        }

    coords = np.argwhere(mask)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)

    extent_vox = bbox_max - bbox_min + 1
    extent_mm = extent_vox * np.array(spacing_mm)

    centroid_voxel = coords.mean(axis=0)
    centroid_mm = centroid_voxel * np.array(spacing_mm)

    voxel_volume_mm3 = float(np.prod(spacing_mm))
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0
    occupancy_percent = (voxel_count / total_voxels) * 100.0
    equivalent_diameter_mm = (6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)

    return {
        "detected": True,
        "voxel_count": voxel_count,
        "occupancy_percent": float(occupancy_percent),
        "volume_mm3": float(volume_mm3),
        "volume_ml": float(volume_ml),
        "equivalent_diameter_mm": float(equivalent_diameter_mm),
        "bbox_min": [int(x) for x in bbox_min.tolist()],
        "bbox_max": [int(x) for x in bbox_max.tolist()],
        "extent_mm": [float(x) for x in extent_mm.tolist()],
        "centroid_voxel": [float(x) for x in centroid_voxel.tolist()],
        "centroid_mm": [float(x) for x in centroid_mm.tolist()],
    }


def compute_class_metrics(class_label_map: np.ndarray, spacing_mm: Tuple[float, float, float]) -> Dict[str, Dict[str, object]]:
    voxel_volume_mm3 = float(np.prod(spacing_mm))
    result: Dict[str, Dict[str, object]] = {}

    for class_label, descriptor in BRATS_CLASS_DEFINITIONS.items():
        mask = class_label_map == class_label
        voxel_count = int(np.count_nonzero(mask))
        volume_mm3 = float(voxel_count * voxel_volume_mm3)
        volume_ml = float(volume_mm3 / 1000.0)

        result[str(class_label)] = {
            "label": int(class_label),
            "key": str(descriptor["key"]),
            "name": str(descriptor["name"]),
            "detected": bool(voxel_count > 0),
            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,
            "volume_ml": volume_ml,
        }

    return result
