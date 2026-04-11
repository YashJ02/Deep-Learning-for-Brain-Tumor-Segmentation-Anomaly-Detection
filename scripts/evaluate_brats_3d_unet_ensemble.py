"""Evaluate K-fold ensemble checkpoints on a BraTS split CSV."""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.inference import load_model_from_checkpoint, segment_with_checkpoint_ensemble
from training.torch_dataset import multiclass_indices_to_brats_labels, seg_to_multiclass_indices
from training.utils import ensure_dir, resolve_device, save_json, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ensemble segmentation on BraTS CSV split.")
    parser.add_argument("--csv", type=Path, default=Path("data") / "splits" / "val.csv")
    parser.add_argument("--checkpoint-glob", type=str, default="models/kfold/fold_*/best.pt")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--modality", type=str, choices=["flair", "t1", "t1ce", "t2"], default="t1ce")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def _discover_checkpoints(args: argparse.Namespace) -> List[Path]:
    if args.checkpoints:
        checkpoints = [_resolve_path(path, PROJECT_ROOT) for path in args.checkpoints]
    else:
        if Path(args.checkpoint_glob).is_absolute():
            checkpoints = sorted(Path(path) for path in glob.glob(args.checkpoint_glob))
        else:
            checkpoints = sorted(PROJECT_ROOT.glob(args.checkpoint_glob))

    checkpoints = [path for path in checkpoints if path.exists()]
    if len(checkpoints) < 2:
        raise FileNotFoundError(
            "Need at least two checkpoint files for ensemble evaluation. "
            f"Found {len(checkpoints)}"
        )
    return checkpoints


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    if not rows:
        raise RuntimeError(f"CSV has no rows: {path}")
    return rows


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _infer_task_from_checkpoint(checkpoint_path: Path, device: str) -> str:
    _, config = load_model_from_checkpoint(checkpoint_path, device=device, use_cache=True)
    config_task = str(config.get("task", "")).strip().lower()
    if config_task in {"binary", "multiclass"}:
        return config_task
    return "multiclass" if int(config.get("out_channels", 1)) > 1 else "binary"


def _dice_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = float(np.logical_and(pred, target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def main() -> int:
    args = parse_args()

    csv_path = _resolve_path(args.csv, PROJECT_ROOT)
    report_dir = ensure_dir(_resolve_path(args.report_dir, PROJECT_ROOT))
    device = resolve_device(args.device)

    rows = _read_csv_rows(csv_path)
    checkpoints = _discover_checkpoints(args)
    task = _infer_task_from_checkpoint(checkpoints[0], str(device))

    case_rows: List[Dict[str, float | str]] = []
    dice_scores: List[float] = []
    iou_scores: List[float] = []
    class_dice_scores: Dict[int, List[float]] = {1: [], 2: [], 4: []}
    class_iou_scores: Dict[int, List[float]] = {1: [], 2: [], 4: []}

    for row in rows:
        image_path = Path(row[args.modality])
        seg_path = Path(row["seg"])

        image = nib.load(str(image_path))
        volume = image.get_fdata(dtype=np.float32)
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        target_mask = seg > 0
        target_class_map = multiclass_indices_to_brats_labels(seg_to_multiclass_indices(seg))

        pred_mask, details = segment_with_checkpoint_ensemble(
            volume=volume,
            checkpoint_paths=checkpoints,
            device=device,
            threshold=args.threshold,
            use_amp=True,
        )

        dice, iou = _dice_iou(pred_mask, target_mask)
        pred_voxels = int(pred_mask.sum())
        gt_voxels = int(target_mask.sum())

        class_metrics: Dict[str, Dict[str, float]] = {}
        if task == "multiclass":
            predicted_class_map = details.get("_class_label_map")
            if isinstance(predicted_class_map, np.ndarray):
                for class_label in (1, 2, 4):
                    class_dice, class_iou = _dice_iou(
                        predicted_class_map == class_label,
                        target_class_map == class_label,
                    )
                    class_dice_scores[class_label].append(class_dice)
                    class_iou_scores[class_label].append(class_iou)
                    class_metrics[str(class_label)] = {
                        "dice": class_dice,
                        "iou": class_iou,
                    }

        case_row: Dict[str, float | str] = {
            "case_id": row["case_id"],
            "dice": dice,
            "iou": iou,
            "pred_voxels": float(pred_voxels),
            "gt_voxels": float(gt_voxels),
            "class_metrics": class_metrics,
        }
        case_rows.append(case_row)
        dice_scores.append(dice)
        iou_scores.append(iou)

    summary = {
        "cases": len(case_rows),
        "task": task,
        "mean_dice": _safe_mean(dice_scores),
        "std_dice": _safe_std(dice_scores),
        "mean_iou": _safe_mean(iou_scores),
        "std_iou": _safe_std(iou_scores),
    }

    if task == "multiclass":
        summary["per_class"] = {
            str(class_label): {
                "mean_dice": _safe_mean(class_dice_scores[class_label]),
                "std_dice": _safe_std(class_dice_scores[class_label]),
                "mean_iou": _safe_mean(class_iou_scores[class_label]),
                "std_iou": _safe_std(class_iou_scores[class_label]),
            }
            for class_label in (1, 2, 4)
        }

    report = {
        "csv": str(csv_path),
        "checkpoints": [str(path) for path in checkpoints],
        "device": str(device),
        "task": task,
        "modality": args.modality,
        "threshold": float(args.threshold),
        "summary": summary,
        "cases": case_rows,
    }

    report_path = report_dir / f"eval_ensemble_{utc_timestamp()}.json"
    save_json(report, report_path)

    print("Ensemble evaluation complete")
    print(f"Cases: {summary['cases']}")
    print(f"Mean Dice: {summary['mean_dice']:.4f} (std={summary['std_dice']:.4f})")
    print(f"Mean IoU: {summary['mean_iou']:.4f} (std={summary['std_iou']:.4f})")
    print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
