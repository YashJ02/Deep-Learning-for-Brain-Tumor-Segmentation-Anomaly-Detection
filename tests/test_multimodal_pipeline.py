# -----yash jain------
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.segmentation import _checkpoint_is_multimodal_multiclass, load_multimodal_nifti_volumes
from training.inference import _validate_multimodal_multiclass_config
from training.torch_dataset import multiclass_indices_to_brats_labels, seg_to_multiclass_indices


def _write_nifti(path: Path, array: np.ndarray, spacing: tuple[float, float, float]) -> None:
    image = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
    image.header.set_zooms(spacing)
    nib.save(image, str(path))


def _make_modality_paths(tmp_path: Path, shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in ("flair", "t1", "t1ce", "t2"):
        path = tmp_path / f"sample_{name}.nii.gz"
        _write_nifti(path, np.random.rand(*shape).astype(np.float32), spacing)
        paths[name] = str(path)
    return paths


def test_load_multimodal_volumes_accepts_matching_shape_and_spacing(tmp_path: Path) -> None:
    paths = _make_modality_paths(tmp_path, shape=(8, 9, 10), spacing=(1.0, 1.0, 1.5))

    volume, spacing = load_multimodal_nifti_volumes(paths)

    assert volume.shape == (4, 8, 9, 10)
    assert volume.dtype == np.float32
    assert spacing == (1.0, 1.0, 1.5)


def test_load_multimodal_volumes_rejects_shape_mismatch(tmp_path: Path) -> None:
    paths = _make_modality_paths(tmp_path, shape=(8, 9, 10), spacing=(1.0, 1.0, 1.0))
    t2_path = Path(paths["t2"])
    _write_nifti(t2_path, np.random.rand(7, 9, 10).astype(np.float32), (1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="share the same shape"):
        load_multimodal_nifti_volumes(paths)


def test_load_multimodal_volumes_rejects_spacing_mismatch(tmp_path: Path) -> None:
    paths = _make_modality_paths(tmp_path, shape=(8, 9, 10), spacing=(1.0, 1.0, 1.0))
    flair_path = Path(paths["flair"])
    _write_nifti(flair_path, np.random.rand(8, 9, 10).astype(np.float32), (1.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="share the same voxel spacing"):
        load_multimodal_nifti_volumes(paths)


def test_multimodal_multiclass_checkpoint_config_validation() -> None:
    _validate_multimodal_multiclass_config({"task": "multiclass", "in_channels": 4, "out_channels": 4})

    with pytest.raises(RuntimeError, match="in_channels=4"):
        _validate_multimodal_multiclass_config({"task": "multiclass", "in_channels": 1, "out_channels": 4})

    with pytest.raises(RuntimeError, match="multiclass checkpoints only"):
        _validate_multimodal_multiclass_config({"task": "binary", "in_channels": 4, "out_channels": 1})


def test_backend_checkpoint_compatibility_detector(tmp_path: Path) -> None:
    compatible = tmp_path / "compatible.pt"
    incompatible = tmp_path / "incompatible.pt"

    torch.save({"config": {"task": "multiclass", "in_channels": 4, "out_channels": 4}}, str(compatible))
    torch.save({"config": {"task": "multiclass", "in_channels": 1, "out_channels": 4}}, str(incompatible))

    assert _checkpoint_is_multimodal_multiclass(compatible)
    assert not _checkpoint_is_multimodal_multiclass(incompatible)


def test_class_label_mapping_behavior() -> None:
    segmentation = np.array(
        [
            [0, 1, 2, 3, 4, 9],
            [4, 3, 2, 1, 0, 0],
        ],
        dtype=np.int32,
    )

    indices = seg_to_multiclass_indices(segmentation)
    restored_labels = multiclass_indices_to_brats_labels(indices)

    expected_indices = np.array(
        [
            [0, 1, 2, 3, 3, 0],
            [3, 3, 2, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    expected_labels = np.array(
        [
            [0, 1, 2, 4, 4, 0],
            [4, 4, 2, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(restored_labels, expected_labels)
