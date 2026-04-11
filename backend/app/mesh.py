from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from skimage.measure import marching_cubes


def _downsample_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    target_max_dim: int = 96,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    target_max_dim = max(int(target_max_dim), 1)
    max_dim = max(mask.shape)
    if max_dim <= target_max_dim:
        return mask, spacing

    stride = int(np.ceil(max_dim / target_max_dim))
    reduced = mask[::stride, ::stride, ::stride]
    reduced_spacing = tuple(float(s * stride) for s in spacing)
    return reduced, reduced_spacing


def build_mesh_from_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    target_max_dim: int = 96,
) -> Dict[str, object]:
    if not np.any(mask):
        return {
            "vertices": [],
            "faces": [],
            "vertex_count": 0,
            "face_count": 0,
            "mesh_shape": [int(x) for x in mask.shape],
        }

    mesh_mask, mesh_spacing = _downsample_mask(mask, spacing, target_max_dim=target_max_dim)
    vertices, faces, _, _ = marching_cubes(mesh_mask.astype(np.float32), level=0.5, spacing=mesh_spacing)

    vertices = np.round(vertices, 3)

    return {
        "vertices": vertices.tolist(),
        "faces": faces.astype(int).tolist(),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "mesh_shape": [int(x) for x in mesh_mask.shape],
    }
