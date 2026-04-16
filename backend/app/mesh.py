# -----yash jain------
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
    try:
        cleaned = remove_small_objects(binary, max_size=min_size)
    except TypeError:
        cleaned = remove_small_objects(binary, min_size=min_size)
    if np.any(cleaned):
        binary = cleaned

    return binary


def build_mesh_from_mask(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    target_max_dim: int = 128,
) -> Dict[str, object]:
    def _empty(shape: Tuple[int, ...]) -> Dict[str, object]:
        return {
            "vertices": [],
            "faces": [],
            "vertex_count": 0,
            "face_count": 0,
            "mesh_shape": [int(x) for x in shape],
        }

    if not np.any(mask):
        return _empty(mask.shape)

    prepared = _prepare_mesh_mask(mask)
    if not np.any(prepared):
        return _empty(mask.shape)

    mesh_mask, mesh_spacing = _downsample_mask(prepared, spacing, target_max_dim=target_max_dim)
    if not np.any(mesh_mask):
        return _empty(mesh_mask.shape)

    smooth_volume = gaussian(mesh_mask.astype(np.float32), sigma=0.8, preserve_range=True)

    # Add a thin zero-valued border so surfaces touching volume boundaries
    # remain extractable and marching-cubes level selection stays valid.
    padded_volume = np.pad(smooth_volume, pad_width=1, mode="constant", constant_values=0.0)
    value_min = float(np.min(padded_volume))
    value_max = float(np.max(padded_volume))
    if (not np.isfinite(value_min)) or (not np.isfinite(value_max)) or value_max <= value_min:
        return _empty(mesh_mask.shape)

    epsilon = max(1e-6, (value_max - value_min) * 1e-6)
    level = min(max(0.35, value_min + epsilon), value_max - epsilon)
    if not np.isfinite(level):
        return _empty(mesh_mask.shape)

    try:
        vertices, faces, _, _ = marching_cubes(
            padded_volume,
            level=float(level),
            spacing=mesh_spacing,
            step_size=1,
            allow_degenerate=False,
        )
    except ValueError:
        return _empty(mesh_mask.shape)

    # Marching-cubes ran on padded coordinates, so shift the mesh back.
    vertices = vertices - np.asarray(mesh_spacing, dtype=np.float32)

    vertices = np.round(vertices, 3)

    return {
        "vertices": vertices.tolist(),
        "faces": faces.astype(int).tolist(),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "mesh_shape": [int(x) for x in mesh_mask.shape],
    }
