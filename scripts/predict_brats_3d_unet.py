"""Run inference with a trained 3D U-Net checkpoint on one NIfTI volume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.inference import load_model_from_checkpoint, predict_binary_mask_from_volume, predict_multiclass_from_volume
from training.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tumor mask from one NIfTI MRI volume.")
    parser.add_argument("--input", type=Path, required=True, help="Input .nii or .nii.gz file")
    parser.add_argument("--checkpoint", type=Path, default=Path("models") / "checkpoints" / "best.pt")
    parser.add_argument("--output-mask", type=Path, default=None, help="Output NIfTI path for predicted mask")
    parser.add_argument("--output-probability", type=Path, default=None, help="Optional output NIfTI path for probability map")
    parser.add_argument("--modality-index", type=int, default=3, help="Modality index for 4D volume input")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--task", type=str, choices=["auto", "binary", "multiclass"], default="auto")
    parser.add_argument("--target-shape", type=int, nargs=3, default=None)
    return parser.parse_args()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def _load_volume(input_path: Path, modality_index: int) -> tuple[np.ndarray, nib.Nifti1Image, int]:
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        index = int(np.clip(modality_index, 0, data.shape[3] - 1))
        volume = data[..., index]
    elif data.ndim == 3:
        index = 0
        volume = data
    else:
        raise ValueError(f"Unsupported NIfTI shape: {data.shape}")

    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return volume, image, index


def _resolve_task(requested: str, config: dict[str, object]) -> str:
    requested = (requested or "auto").strip().lower()
    if requested in {"binary", "multiclass"}:
        return requested

    config_task = str(config.get("task", "")).strip().lower()
    if config_task in {"binary", "multiclass"}:
        return config_task

    return "multiclass" if int(config.get("out_channels", 1)) > 1 else "binary"


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT

    input_path = _resolve_path(args.input, project_root)
    checkpoint_path = _resolve_path(args.checkpoint, project_root)
    device = resolve_device(args.device)

    volume, image, used_modality = _load_volume(input_path, args.modality_index)

    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    task = _resolve_task(args.task, config)
    if args.target_shape is None:
        target_shape_cfg = config.get("target_shape", [128, 128, 128])
        target_shape = tuple(int(v) for v in target_shape_cfg)
    else:
        target_shape = tuple(args.target_shape)

    if task == "binary":
        mask, probability = predict_binary_mask_from_volume(
            model=model,
            volume=volume,
            device=device,
            target_shape=target_shape,
            threshold=args.threshold,
            use_amp=True,
        )
        output_array = mask.astype(np.uint8)
        class_counts: dict[str, int] = {}
    else:
        class_label_map, _, max_probability, _ = predict_multiclass_from_volume(
            model=model,
            volume=volume,
            device=device,
            target_shape=target_shape,
            threshold=args.threshold,
            use_amp=True,
        )
        mask = class_label_map > 0
        probability = max_probability
        output_array = class_label_map.astype(np.uint8)
        class_counts = {
            "1": int(np.count_nonzero(class_label_map == 1)),
            "2": int(np.count_nonzero(class_label_map == 2)),
            "4": int(np.count_nonzero(class_label_map == 4)),
        }

    if args.output_mask is None:
        predictions_dir = ensure_dir(project_root / "models" / "predictions")
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        output_mask = predictions_dir / f"{stem}_mask.nii.gz"
    else:
        output_mask = _resolve_path(args.output_mask, project_root)
        output_mask.parent.mkdir(parents=True, exist_ok=True)

    mask_image = nib.Nifti1Image(output_array, image.affine, image.header)
    nib.save(mask_image, str(output_mask))

    if args.output_probability is not None:
        probability_path = _resolve_path(args.output_probability, project_root)
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        probability_image = nib.Nifti1Image(probability.astype(np.float32), image.affine, image.header)
        nib.save(probability_image, str(probability_path))
    else:
        probability_path = None

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    voxel_count = int(mask.sum())
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0

    print("Inference complete")
    print(f"Input: {input_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {task}")
    print(f"Device: {device}")
    print(f"Used modality index: {used_modality}")
    print(f"Predicted mask: {output_mask}")
    if probability_path is not None:
        print(f"Probability map: {probability_path}")
    print(f"Detected voxels: {voxel_count}")
    if class_counts:
        print(f"Class voxel counts (BraTS labels): {class_counts}")
    print(f"Estimated volume: {volume_ml:.3f} mL")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
