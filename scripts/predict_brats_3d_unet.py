"""Run multimodal multiclass inference with a trained 3D U-Net checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.inference import load_model_from_checkpoint, predict_multiclass_from_volume
from training.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict BraTS multiclass map from four modality NIfTI inputs.")
    parser.add_argument("--flair", type=Path, required=True, help="Input FLAIR .nii or .nii.gz file")
    parser.add_argument("--t1", type=Path, required=True, help="Input T1 .nii or .nii.gz file")
    parser.add_argument("--t1ce", type=Path, required=True, help="Input T1ce .nii or .nii.gz file")
    parser.add_argument("--t2", type=Path, required=True, help="Input T2 .nii or .nii.gz file")
    parser.add_argument("--checkpoint", type=Path, default=Path("models") / "checkpoints" / "best.pt")
    parser.add_argument("--output-mask", type=Path, default=None, help="Output NIfTI path for predicted class map")
    parser.add_argument("--output-probability", type=Path, default=None, help="Optional output NIfTI path for max probability map")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--target-shape", type=int, nargs=3, default=None)
    return parser.parse_args()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def _validate_checkpoint_config(config: dict[str, object]) -> None:
    in_channels = int(config.get("in_channels", 4))
    out_channels = int(config.get("out_channels", 4))
    task = str(config.get("task", "multiclass")).strip().lower()

    if in_channels != 4:
        raise RuntimeError(f"Expected multimodal checkpoint with in_channels=4, received {in_channels}")
    if out_channels not in {3, 4}:
        raise RuntimeError(f"Expected multiclass out_channels in {{3, 4}}, received {out_channels}")
    if task not in {"", "multiclass"}:
        raise RuntimeError(f"Expected multiclass checkpoint task, received {task!r}")


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


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT

    input_paths = {
        "flair": _resolve_path(args.flair, project_root),
        "t1": _resolve_path(args.t1, project_root),
        "t1ce": _resolve_path(args.t1ce, project_root),
        "t2": _resolve_path(args.t2, project_root),
    }
    checkpoint_path = _resolve_path(args.checkpoint, project_root)
    device = resolve_device(args.device)

    volume, reference_image = _load_multimodal_volume(input_paths)

    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    _validate_checkpoint_config(config)

    if args.target_shape is None:
        target_shape_cfg = config.get("target_shape", [128, 128, 128])
        target_shape = tuple(int(v) for v in target_shape_cfg)
    else:
        target_shape = tuple(args.target_shape)

    class_label_map, _, max_probability, _ = predict_multiclass_from_volume(
        model=model,
        volume=volume,
        device=device,
        target_shape=target_shape,
        threshold=args.threshold,
        use_amp=True,
    )
    mask = class_label_map > 0

    if args.output_mask is None:
        predictions_dir = ensure_dir(project_root / "models" / "predictions")
        stem = input_paths["flair"].name.replace(".nii.gz", "").replace(".nii", "")
        output_mask = predictions_dir / f"{stem}_mask.nii.gz"
    else:
        output_mask = _resolve_path(args.output_mask, project_root)
        output_mask.parent.mkdir(parents=True, exist_ok=True)

    mask_image = nib.Nifti1Image(class_label_map.astype(np.uint8), reference_image.affine, reference_image.header)
    nib.save(mask_image, str(output_mask))

    if args.output_probability is not None:
        probability_path = _resolve_path(args.output_probability, project_root)
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        probability_image = nib.Nifti1Image(max_probability.astype(np.float32), reference_image.affine, reference_image.header)
        nib.save(probability_image, str(probability_path))
    else:
        probability_path = None

    spacing = tuple(float(value) for value in reference_image.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    voxel_count = int(mask.sum())
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0
    class_counts = {
        "1": int(np.count_nonzero(class_label_map == 1)),
        "2": int(np.count_nonzero(class_label_map == 2)),
        "4": int(np.count_nonzero(class_label_map == 4)),
    }

    print("Inference complete")
    print(f"Inputs: {input_paths}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Task: multiclass")
    print("Input channels: 4")
    print(f"Device: {device}")
    print(f"Predicted class map: {output_mask}")
    if probability_path is not None:
        print(f"Probability map: {probability_path}")
    print(f"Detected voxels: {voxel_count}")
    print(f"Class voxel counts (BraTS labels): {class_counts}")
    print(f"Estimated volume: {volume_ml:.3f} mL")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
