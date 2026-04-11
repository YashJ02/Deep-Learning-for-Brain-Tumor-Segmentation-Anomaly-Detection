"""Run multimodal multiclass ensemble inference from K-fold checkpoints."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.inference import load_model_from_checkpoint, segment_with_checkpoint_ensemble
from training.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict BraTS multiclass map from multimodal ensemble checkpoints.")
    parser.add_argument("--flair", type=Path, required=True, help="Input FLAIR .nii or .nii.gz file")
    parser.add_argument("--t1", type=Path, required=True, help="Input T1 .nii or .nii.gz file")
    parser.add_argument("--t1ce", type=Path, required=True, help="Input T1ce .nii or .nii.gz file")
    parser.add_argument("--t2", type=Path, required=True, help="Input T2 .nii or .nii.gz file")
    parser.add_argument("--checkpoint-glob", type=str, default="models/kfold/fold_*/best.pt")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None, help="Optional explicit checkpoint list")
    parser.add_argument("--output-mask", type=Path, default=None, help="Output NIfTI path for predicted class map")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def _load_multimodal_volume(paths: dict[str, Path]) -> tuple[np.ndarray, nib.Nifti1Image]:
    modality_order = ("flair", "t1", "t1ce", "t2")
    channels: list[np.ndarray] = []
    reference_image: nib.Nifti1Image | None = None
    first_shape: tuple[int, ...] | None = None

    for modality in modality_order:
        image = nib.load(str(paths[modality]))
        data = image.get_fdata(dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if data.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI for {modality}, got shape {data.shape}")

        shape = tuple(int(v) for v in data.shape)
        if first_shape is None:
            first_shape = shape
            reference_image = image
        elif shape != first_shape:
            raise ValueError(f"All modality inputs must share shape. {modality}={shape}, expected={first_shape}")

        channels.append(data)

    if reference_image is None:
        raise RuntimeError("Failed to load multimodal inputs")

    return np.stack(channels, axis=0).astype(np.float32), reference_image


def _discover_checkpoints(args: argparse.Namespace) -> List[Path]:
    if args.checkpoints:
        checkpoints = [_resolve_path(path, PROJECT_ROOT) for path in args.checkpoints]
    else:
        pattern = args.checkpoint_glob
        if Path(pattern).is_absolute():
            checkpoints = sorted(Path(path) for path in glob.glob(pattern))
        else:
            checkpoints = sorted(PROJECT_ROOT.glob(pattern))

    checkpoints = [path for path in checkpoints if path.exists()]
    if len(checkpoints) < 2:
        raise FileNotFoundError(
            "Need at least two checkpoint files for ensemble inference. "
            f"Found {len(checkpoints)}"
        )
    return checkpoints


def _validate_checkpoint_config(checkpoint_path: Path, device: str) -> None:
    _, config = load_model_from_checkpoint(checkpoint_path, device=device, use_cache=True)

    in_channels = int(config.get("in_channels", 4))
    out_channels = int(config.get("out_channels", 4))
    task = str(config.get("task", "multiclass")).strip().lower()

    if in_channels != 4:
        raise RuntimeError(f"Expected multimodal checkpoint with in_channels=4, got {in_channels}: {checkpoint_path}")
    if out_channels not in {3, 4}:
        raise RuntimeError(f"Expected multiclass out_channels in {{3, 4}}, got {out_channels}: {checkpoint_path}")
    if task not in {"", "multiclass"}:
        raise RuntimeError(f"Expected multiclass checkpoint task, got {task!r}: {checkpoint_path}")


def main() -> int:
    args = parse_args()

    input_paths = {
        "flair": _resolve_path(args.flair, PROJECT_ROOT),
        "t1": _resolve_path(args.t1, PROJECT_ROOT),
        "t1ce": _resolve_path(args.t1ce, PROJECT_ROOT),
        "t2": _resolve_path(args.t2, PROJECT_ROOT),
    }
    device = resolve_device(args.device)
    checkpoints = _discover_checkpoints(args)
    for checkpoint in checkpoints:
        _validate_checkpoint_config(checkpoint, str(device))

    volume, reference_image = _load_multimodal_volume(input_paths)

    mask, details = segment_with_checkpoint_ensemble(
        volume=volume,
        checkpoint_paths=checkpoints,
        device=device,
        threshold=args.threshold,
        use_amp=True,
    )

    class_label_map = details.get("_class_label_map")
    if not isinstance(class_label_map, np.ndarray):
        raise RuntimeError("Expected multiclass class map from ensemble inference")

    if args.output_mask is None:
        predictions_dir = ensure_dir(PROJECT_ROOT / "models" / "predictions")
        stem = input_paths["flair"].name.replace(".nii.gz", "").replace(".nii", "")
        output_mask = predictions_dir / f"{stem}_ensemble_mask.nii.gz"
    else:
        output_mask = _resolve_path(args.output_mask, PROJECT_ROOT)
        output_mask.parent.mkdir(parents=True, exist_ok=True)

    mask_image = nib.Nifti1Image(class_label_map.astype(np.uint8), reference_image.affine, reference_image.header)
    nib.save(mask_image, str(output_mask))

    spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    voxel_count = int(mask.sum())
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0
    class_counts = {
        "1": int(np.count_nonzero(class_label_map == 1)),
        "2": int(np.count_nonzero(class_label_map == 2)),
        "4": int(np.count_nonzero(class_label_map == 4)),
    }

    print("Ensemble inference complete")
    print(f"Inputs: {input_paths}")
    print(f"Device: {device}")
    print("Task: multiclass")
    print("Input channels: 4")
    print(f"Ensemble size: {details['ensemble_size']}")
    print(f"Predicted class map: {output_mask}")
    print(f"Detected voxels: {voxel_count}")
    print(f"Class voxel counts (BraTS labels): {class_counts}")
    print(f"Estimated volume: {volume_ml:.3f} mL")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
