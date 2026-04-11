"""Evaluate a trained multimodal multiclass 3D U-Net checkpoint on a BraTS split CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import read_split_csv
from training.inference import load_model_from_checkpoint
from training.losses import multiclass_ce_dice_loss
from training.metrics import multiclass_dice_iou_from_logits
from training.torch_dataset import BraTSTorchDataset
from training.utils import ensure_dir, resolve_device, save_json, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a multimodal multiclass BraTS 3D U-Net checkpoint.")
    parser.add_argument("--csv", type=Path, default=Path("data") / "splits" / "val.csv")
    parser.add_argument("--checkpoint", type=Path, default=Path("models") / "checkpoints" / "best.pt")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--target-shape", type=int, nargs=3, default=None)
    return parser.parse_args()


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def _safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _validate_checkpoint_config(config: Dict[str, object]) -> None:
    in_channels = int(config.get("in_channels", 4))
    out_channels = int(config.get("out_channels", 4))
    task = str(config.get("task", "multiclass")).strip().lower()

    if in_channels != 4:
        raise RuntimeError(f"Expected multimodal checkpoint with in_channels=4, received {in_channels}")
    if out_channels not in {3, 4}:
        raise RuntimeError(f"Expected multiclass out_channels in {{3, 4}}, received {out_channels}")
    if task not in {"", "multiclass"}:
        raise RuntimeError(f"Expected multiclass checkpoint task, received {task!r}")


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT

    csv_path = _resolve_path(args.csv, project_root)
    checkpoint_path = _resolve_path(args.checkpoint, project_root)
    report_dir = ensure_dir(_resolve_path(args.report_dir, project_root))

    device = resolve_device(args.device)
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    _validate_checkpoint_config(config)

    if args.target_shape is None:
        target_shape = tuple(int(v) for v in config.get("target_shape", [128, 128, 128]))
    else:
        target_shape = tuple(args.target_shape)

    records = read_split_csv(csv_path)
    dataset = BraTSTorchDataset(
        records=records,
        target_shape=target_shape,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: List[Dict[str, float | str]] = []
    dice_scores: List[float] = []
    iou_scores: List[float] = []
    losses: List[float] = []
    class_dice_scores: Dict[int, List[float]] = {}
    class_iou_scores: Dict[int, List[float]] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True).long()

            logits = model(images)
            loss = float(multiclass_ce_dice_loss(logits, masks, ce_weight=args.ce_weight).item())
            dice, iou, per_class = multiclass_dice_iou_from_logits(logits, masks)

            predictions = torch.argmax(logits, dim=1)
            pred_voxels = int((predictions > 0).sum().item())
            gt_voxels = int((masks > 0).sum().item())

            class_metrics = {
                str(class_index): {
                    "dice": float(metrics["dice"]),
                    "iou": float(metrics["iou"]),
                }
                for class_index, metrics in per_class.items()
            }

            for class_index, metrics in per_class.items():
                class_dice_scores.setdefault(int(class_index), []).append(float(metrics["dice"]))
                class_iou_scores.setdefault(int(class_index), []).append(float(metrics["iou"]))

            case_ids = batch["case_id"]
            if isinstance(case_ids, list):
                case_id = str(case_ids[0])
            else:
                case_id = str(case_ids)

            rows.append(
                {
                    "case_id": case_id,
                    "loss": loss,
                    "dice": dice,
                    "iou": iou,
                    "pred_voxels": float(pred_voxels),
                    "gt_voxels": float(gt_voxels),
                    "class_metrics": class_metrics,
                }
            )
            losses.append(loss)
            dice_scores.append(dice)
            iou_scores.append(iou)

    summary = {
        "cases": len(rows),
        "task": "multiclass",
        "input_channels": 4,
        "mean_loss": _safe_mean(losses),
        "std_loss": _safe_std(losses),
        "mean_dice": _safe_mean(dice_scores),
        "std_dice": _safe_std(dice_scores),
        "mean_iou": _safe_mean(iou_scores),
        "std_iou": _safe_std(iou_scores),
    }

    per_class_summary: Dict[str, Dict[str, float]] = {}
    for class_index in sorted(class_dice_scores):
        per_class_summary[str(class_index)] = {
            "mean_dice": _safe_mean(class_dice_scores.get(class_index, [])),
            "std_dice": _safe_std(class_dice_scores.get(class_index, [])),
            "mean_iou": _safe_mean(class_iou_scores.get(class_index, [])),
            "std_iou": _safe_std(class_iou_scores.get(class_index, [])),
        }
    summary["per_class"] = per_class_summary

    report = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "task": "multiclass",
        "input_channels": 4,
        "target_shape": list(target_shape),
        "class_index_to_brats_label": {"0": 0, "1": 1, "2": 2, "3": 4},
        "summary": summary,
        "cases": rows,
    }

    report_path = report_dir / f"eval_{utc_timestamp()}.json"
    save_json(report, report_path)

    print("Evaluation complete")
    print(f"Cases: {summary['cases']}")
    print(f"Mean Dice: {summary['mean_dice']:.4f} (std={summary['std_dice']:.4f})")
    print(f"Mean IoU: {summary['mean_iou']:.4f} (std={summary['std_iou']:.4f})")
    print(f"Mean Loss: {summary['mean_loss']:.4f}")
    print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
