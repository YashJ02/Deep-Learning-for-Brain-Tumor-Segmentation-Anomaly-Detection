from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from skimage.filters import gaussian
from skimage.measure import marching_cubes
from skimage.morphology import remove_small_objects
from skimage.transform import resize


def _downsample_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    target_max_dim: int = 128,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    target_max_dim = max(int(target_max_dim), 1)
    max_dim = max(int(x) for x in mask.shape)
    if max_dim <= target_max_dim:
        return mask.astype(bool), spacing

    scale = float(target_max_dim) / float(max_dim)
    new_shape = tuple(max(2, int(round(int(dim) * scale))) for dim in mask.shape)

    reduced = resize(
        mask.astype(np.float32),
        output_shape=new_shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )
    reduced = reduced >= 0.5

    scale_factors = tuple(float(orig) / float(new) for orig, new in zip(mask.shape, new_shape))
    reduced_spacing = tuple(float(s * factor) for s, factor in zip(spacing, scale_factors))
    return reduced.astype(bool), reduced_spacing


def _prepare_mesh_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    voxel_count = int(np.count_nonzero(binary))
    if voxel_count == 0:
        return binary

    # Remove tiny disconnected blobs that make the viewer look like a point cloud.
    min_size = max(24, int(voxel_count * 0.0008))
    cleaned = remove_small_objects(binary, max_size=min_size)
    if np.any(cleaned):
        binary = cleaned

    return binary


def build_mesh_from_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    target_max_dim: int = 128,
) -> Dict[str, object]:
    if not np.any(mask):
        return {
            "vertices": [],
            "faces": [],
            "vertex_count": 0,
            "face_count": 0,
            "mesh_shape": [int(x) for x in mask.shape],
        }

    prepared = _prepare_mesh_mask(mask)
    if not np.any(prepared):
        return {
            "vertices": [],
            "faces": [],
            "vertex_count": 0,
            "face_count": 0,
            "mesh_shape": [int(x) for x in mask.shape],
        }

    mesh_mask, mesh_spacing = _downsample_mask(prepared, spacing, target_max_dim=target_max_dim)
    if not np.any(mesh_mask):
        return {
            "vertices": [],
            "faces": [],
            "vertex_count": 0,
            "face_count": 0,
            "mesh_shape": [int(x) for x in mesh_mask.shape],
        }

    smooth_volume = gaussian(mesh_mask.astype(np.float32), sigma=0.8, preserve_range=True)
    vertices, faces, _, _ = marching_cubes(
        smooth_volume,
        level=0.35,
        spacing=mesh_spacing,
        step_size=1,
        allow_degenerate=False,
    )

    vertices = np.round(vertices, 3)

    return {
        "vertices": vertices.tolist(),
        "faces": faces.astype(int).tolist(),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "mesh_shape": [int(x) for x in mesh_mask.shape],
    }
