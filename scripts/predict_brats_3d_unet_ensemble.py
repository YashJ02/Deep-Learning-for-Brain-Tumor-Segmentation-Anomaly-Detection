"""Run ensemble inference from K-fold checkpoints on one NIfTI volume."""

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

from training.inference import segment_with_checkpoint_ensemble
from training.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tumor mask from ensemble of K-fold checkpoints.")
    parser.add_argument("--input", type=Path, required=True, help="Input .nii or .nii.gz file")
    parser.add_argument("--checkpoint-glob", type=str, default="models/kfold/fold_*/best.pt")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None, help="Optional explicit checkpoint list")
    parser.add_argument("--output-mask", type=Path, default=None, help="Output NIfTI path for predicted mask")
    parser.add_argument("--channel-index", type=int, default=3, help="Channel index for 4D volume input")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def _load_volume(input_path: Path, channel_index: int) -> tuple[np.ndarray, nib.Nifti1Image, int]:
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        index = int(np.clip(channel_index, 0, data.shape[3] - 1))
        volume = data[..., index]
    elif data.ndim == 3:
        index = 0
        volume = data
    else:
        raise ValueError(f"Unsupported NIfTI shape: {data.shape}")

    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return volume, image, index


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


def main() -> int:
    args = parse_args()

    input_path = _resolve_path(args.input, PROJECT_ROOT)
    device = resolve_device(args.device)
    checkpoints = _discover_checkpoints(args)

    volume, image, used_modality = _load_volume(input_path, args.channel_index)

    mask, details = segment_with_checkpoint_ensemble(
        volume=volume,
        checkpoint_paths=checkpoints,
        device=device,
        threshold=args.threshold,
        use_amp=True,
    )

    task = str(details.get("task", "binary"))
    class_counts: dict[str, int] = {}
    if task == "multiclass" and isinstance(details.get("_class_label_map"), np.ndarray):
        class_label_map = details["_class_label_map"]
        output_array = class_label_map.astype(np.uint8)
        class_counts = {
            "1": int(np.count_nonzero(class_label_map == 1)),
            "2": int(np.count_nonzero(class_label_map == 2)),
            "4": int(np.count_nonzero(class_label_map == 4)),
        }
    else:
        output_array = mask.astype(np.uint8)

    if args.output_mask is None:
        predictions_dir = ensure_dir(PROJECT_ROOT / "models" / "predictions")
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        output_mask = predictions_dir / f"{stem}_ensemble_mask.nii.gz"
    else:
        output_mask = _resolve_path(args.output_mask, PROJECT_ROOT)
        output_mask.parent.mkdir(parents=True, exist_ok=True)

    mask_image = nib.Nifti1Image(output_array, image.affine, image.header)
    nib.save(mask_image, str(output_mask))

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))
    voxel_count = int(mask.sum())
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0

    print("Ensemble inference complete")
    print(f"Input: {input_path}")
    print(f"Device: {device}")
    print(f"Task: {task}")
    print(f"Used channel index: {used_modality}")
    print(f"Ensemble size: {details['ensemble_size']}")
    print(f"Predicted mask: {output_mask}")
    print(f"Detected voxels: {voxel_count}")
    if class_counts:
        print(f"Class voxel counts (BraTS labels): {class_counts}")
    print(f"Estimated volume: {volume_ml:.3f} mL")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
