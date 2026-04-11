from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import ball


_CHECKPOINT_TASK_CACHE: Dict[str, str] = {}


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels].astype(bool)


def _percentile_normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p1, p99 = np.percentile(volume, [1, 99])
    if p99 <= p1:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)


def load_multimodal_nifti_volumes(modality_paths: Dict[str, str]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    required = ["flair", "t1", "t1ce", "t2"]
    missing = [name for name in required if not modality_paths.get(name)]
    if missing:
        raise ValueError(f"Missing modality files: {missing}")

    first_shape: Optional[Tuple[int, ...]] = None
    first_spacing: Optional[Tuple[float, float, float]] = None
    normalized_channels: List[np.ndarray] = []

    for name in required:
        path = str(modality_paths[name])
        image = nib.load(path)
        data = image.get_fdata(dtype=np.float32)
        spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

        if data.ndim != 3:
            raise ValueError(f"Multimodal upload expects 3D files per modality; {name} has shape {data.shape}")

        if first_shape is None:
            first_shape = tuple(int(x) for x in data.shape)
            first_spacing = spacing
        else:
            if tuple(int(x) for x in data.shape) != first_shape:
                raise ValueError(f"All modality files must share the same shape. {name} shape={data.shape}, expected={first_shape}")
            if first_spacing is not None and not np.allclose(np.array(spacing), np.array(first_spacing), atol=1e-4):
                raise ValueError(f"All modality files must share the same voxel spacing. {name} spacing={spacing}, expected={first_spacing}")

        normalized_channels.append(_percentile_normalize(data))

    stacked = np.stack(normalized_channels, axis=0)
    fused = np.mean(stacked, axis=0).astype(np.float32)
    return fused, tuple(first_spacing if first_spacing is not None else (1.0, 1.0, 1.0))


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


def extract_brain_mask(volume: np.ndarray) -> np.ndarray:
    smooth = ndi.gaussian_filter(volume, sigma=0.8)
    non_zero = smooth[smooth > 0]

    if non_zero.size == 0:
        return np.zeros_like(smooth, dtype=bool)

    threshold = max(float(np.percentile(non_zero, 20)), 0.05)
    mask = smooth > threshold
    mask = ndi.binary_opening(mask, structure=ball(1))
    mask = ndi.binary_closing(mask, structure=ball(2))
    mask = ndi.binary_fill_holes(mask)

    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_component = int(component_sizes.argmax())
    return (labels == largest_component).astype(bool)


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "best.pt"


def default_ensemble_checkpoint_paths() -> List[Path]:
    project_root = Path(__file__).resolve().parents[2]
    kfold_root = project_root / "models" / "kfold"
    return sorted(kfold_root.glob("fold_*/best.pt"))


def _candidate_deep_checkpoints(explicit_checkpoint_path: Optional[str]) -> List[Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidates: List[Path] = []
    if explicit_checkpoint_path:
        candidates.append(Path(explicit_checkpoint_path).resolve())

    candidates.extend(
        [
            (project_root / "models" / "checkpoints_real_multiclass" / "best.pt").resolve(),
            default_checkpoint_path().resolve(),
            (project_root / "models" / "brats_3d_unet_best.pt").resolve(),
        ]
    )

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _checkpoint_task(checkpoint_path: Path) -> str:
    checkpoint = checkpoint_path.resolve()
    key = str(checkpoint)
    if key in _CHECKPOINT_TASK_CACHE:
        return _CHECKPOINT_TASK_CACHE[key]

    if not checkpoint.exists():
        _CHECKPOINT_TASK_CACHE[key] = "missing"
        return "missing"

    task = "unknown"
    try:
        import torch

        payload = torch.load(str(checkpoint), map_location="cpu")
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        config_task = str(config.get("task", "")).strip().lower()
        if config_task in {"binary", "multiclass"}:
            task = config_task
        else:
            out_channels = int(config.get("out_channels", 1))
            task = "multiclass" if out_channels > 1 else "binary"
    except Exception:
        task = "unknown"

    _CHECKPOINT_TASK_CACHE[key] = task
    return task


def _multiclass_only(paths: Sequence[Path]) -> List[Path]:
    return [path for path in paths if path.exists() and _checkpoint_task(path) == "multiclass"]


def _preferred_multiclass_checkpoint(explicit_checkpoint_path: Optional[str]) -> Optional[Path]:
    for candidate in _candidate_deep_checkpoints(explicit_checkpoint_path):
        if candidate.exists() and _checkpoint_task(candidate) == "multiclass":
            return candidate
    return None


def _extract_fold_index(path: Path) -> Optional[int]:
    folder = path.parent.name
    if not folder.startswith("fold_"):
        return None
    suffix = folder.split("fold_", maxsplit=1)[1]
    if suffix.isdigit():
        return int(suffix)
    return None


def available_fold_checkpoints() -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for checkpoint in default_ensemble_checkpoint_paths():
        items.append(
            {
                "fold_index": _extract_fold_index(checkpoint),
                "path": str(checkpoint),
                "task": _checkpoint_task(checkpoint),
            }
        )
    return items


def resolve_ensemble_checkpoint_paths(fold_indices: Sequence[int]) -> List[Path]:
    project_root = Path(__file__).resolve().parents[2]
    fold_root = project_root / "models" / "kfold"

    normalized = sorted(set(int(value) for value in fold_indices))
    selected: List[Path] = []
    missing: List[int] = []

    for index in normalized:
        if index < 0:
            raise ValueError("Fold indices must be non-negative integers")
        checkpoint = fold_root / f"fold_{index}" / "best.pt"
        if checkpoint.exists():
            selected.append(checkpoint)
        else:
            missing.append(index)

    if missing:
        raise FileNotFoundError(f"Missing fold checkpoints for indices: {missing}")

    return selected


def segment_tumor(
    volume: np.ndarray,
    engine: str = "auto",
    checkpoint_path: Optional[str] = None,
    ensemble_checkpoint_paths: Optional[Sequence[Path]] = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, object], Optional[np.ndarray]]:
    engine = (engine or "auto").strip().lower()
    if engine not in {"all", "auto", "deep", "ensemble", "baseline"}:
        raise ValueError("engine must be one of: all, auto, deep, ensemble, baseline")

    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else default_checkpoint_path().resolve()
    ensemble_paths = list(ensemble_checkpoint_paths) if ensemble_checkpoint_paths is not None else default_ensemble_checkpoint_paths()

    if engine == "all":
        multiclass_ensemble_paths = _multiclass_only(ensemble_paths)
        preferred_multiclass_checkpoint = _preferred_multiclass_checkpoint(checkpoint_path)

        if multiclass_ensemble_paths:
            try:
                from training.inference import segment_with_checkpoint_ensemble

                ensemble_mask, details = segment_with_checkpoint_ensemble(
                    volume=volume,
                    checkpoint_paths=multiclass_ensemble_paths,
                    threshold=threshold,
                )
                class_label_map = details.pop("_class_label_map", None)
                return ensemble_mask.astype(bool), {
                    "engine": "ensemble",
                    "checkpoint": None,
                    "fold_indices": [
                        fold_index
                        for fold_index in (_extract_fold_index(path) for path in multiclass_ensemble_paths)
                        if fold_index is not None
                    ],
                    **{key: value for key, value in details.items() if not key.startswith("_")},
                }, class_label_map
            except Exception as exc:
                if preferred_multiclass_checkpoint is None:
                    raise RuntimeError(f"Multiclass ensemble inference failed: {exc}") from exc

        if preferred_multiclass_checkpoint is not None:
            try:
                from training.inference import segment_with_checkpoint

                deep_mask, details = segment_with_checkpoint(
                    volume=volume,
                    checkpoint_path=preferred_multiclass_checkpoint,
                    threshold=threshold,
                )
                class_label_map = details.pop("_class_label_map", None)
                return deep_mask.astype(bool), {
                    "engine": "deep",
                    "checkpoint": str(preferred_multiclass_checkpoint),
                    "fold_indices": [],
                    **{key: value for key, value in details.items() if not key.startswith("_")},
                }, class_label_map
            except Exception as exc:
                raise RuntimeError(f"Multiclass deep-model inference failed: {exc}") from exc

        raise FileNotFoundError(
            "No multiclass checkpoints available for engine='all'. Train or provide multiclass checkpoints first."
        )

    if engine in {"auto", "ensemble"}:
        if len(ensemble_paths) >= 1:
            try:
                from training.inference import segment_with_checkpoint_ensemble

                ensemble_mask, details = segment_with_checkpoint_ensemble(
                    volume=volume,
                    checkpoint_paths=ensemble_paths,
                    threshold=threshold,
                )
                class_label_map = details.pop("_class_label_map", None)
                return ensemble_mask.astype(bool), {
                    "engine": "ensemble",
                    "checkpoint": None,
                    "fold_indices": [
                        fold_index
                        for fold_index in (_extract_fold_index(path) for path in ensemble_paths)
                        if fold_index is not None
                    ],
                    **{key: value for key, value in details.items() if not key.startswith("_")},
                }, class_label_map
            except Exception as exc:
                if engine == "ensemble":
                    raise RuntimeError(f"Ensemble inference failed: {exc}") from exc
        elif engine == "ensemble":
            raise FileNotFoundError(
                "No ensemble checkpoints available. Expected files under models/kfold/fold_*/best.pt"
            )

    if engine in {"auto", "deep"}:
        if checkpoint.exists():
            try:
                from training.inference import segment_with_checkpoint

                deep_mask, details = segment_with_checkpoint(
                    volume=volume,
                    checkpoint_path=checkpoint,
                    threshold=threshold,
                )
                class_label_map = details.pop("_class_label_map", None)
                return deep_mask.astype(bool), {
                    "engine": "deep",
                    "checkpoint": str(checkpoint),
                    "fold_indices": [],
                    **{key: value for key, value in details.items() if not key.startswith("_")},
                }, class_label_map
            except Exception as exc:
                if engine == "deep":
                    raise RuntimeError(f"Deep model inference failed: {exc}") from exc
        elif engine == "deep":
            raise FileNotFoundError(f"Deep-model checkpoint was not found: {checkpoint}")

    baseline_mask = segment_tumor_baseline(volume)
    return baseline_mask.astype(bool), {
        "engine": "baseline",
        "task": "binary",
        "checkpoint": None,
        "fold_indices": [],
        "probability_mean": None,
        "probability_max": None,
    }, None
