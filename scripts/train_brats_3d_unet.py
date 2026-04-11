"""Train a CUDA-ready multimodal multiclass 3D U-Net on BraTS."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import read_split_csv
from training.losses import multiclass_ce_dice_loss
from training.metrics import multiclass_dice_iou_from_logits
from training.model import UNet3D, count_parameters
from training.torch_dataset import BraTSTorchDataset
from training.utils import ensure_dir, resolve_device, save_json, set_seed

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal multiclass 3D U-Net on BraTS.")
    parser.add_argument("--train-csv", type=Path, default=Path("data") / "splits" / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=Path("data") / "splits" / "val.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models") / "checkpoints")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--target-shape", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, cuda:0, ...")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint file to resume from.")
    return parser.parse_args()


def _progress(iterable, description: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=description, leave=False)


def _resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else (project_root / path)


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT

    train_csv = _resolve_path(args.train_csv, project_root)
    val_csv = _resolve_path(args.val_csv, project_root)
    checkpoint_dir = ensure_dir(_resolve_path(args.checkpoint_dir, project_root))

    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp or device.type == "cuda")

    train_records = read_split_csv(train_csv)
    val_records = read_split_csv(val_csv)

    train_dataset = BraTSTorchDataset(
        records=train_records,
        target_shape=tuple(args.target_shape),
        augment=True,
    )
    val_dataset = BraTSTorchDataset(
        records=val_records,
        target_shape=tuple(args.target_shape),
        augment=False,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=pin_memory,
    )

    model = UNet3D(in_channels=4, out_channels=4, base_channels=args.base_channels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp and device.type == "cuda")

    start_epoch = 1
    best_dice = 0.0
    history: List[Dict[str, float]] = []

    if args.resume is not None:
        resume_path = _resolve_path(args.resume, project_root)
        checkpoint = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed training from checkpoint: {resume_path}")

    config = {
        "task": "multiclass",
        "in_channels": 4,
        "out_channels": 4,
        "base_channels": int(args.base_channels),
        "target_shape": [int(v) for v in args.target_shape],
        "ce_weight": float(args.ce_weight),
    }

    print("Starting training")
    print("Task: multiclass")
    print("Input channels: 4 (flair, t1, t1ce, t2)")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({memory_gb:.1f} GB)")
    print(f"Train cases: {len(train_dataset)} | Val cases: {len(val_dataset)}")
    print(f"Model parameters: {count_parameters(model):,}")

    best_checkpoint_path = checkpoint_dir / "best.pt"
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    history_path = checkpoint_dir / "history.json"

    total_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        train_loss_total = 0.0
        train_dice_total = 0.0
        train_iou_total = 0.0

        for batch in _progress(train_loader, f"Epoch {epoch}/{args.epochs} [train]"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(images)
                loss = multiclass_ce_dice_loss(logits, masks, ce_weight=args.ce_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += float(loss.item())
            dice_macro, iou_macro, _ = multiclass_dice_iou_from_logits(logits.detach(), masks)
            train_dice_total += dice_macro
            train_iou_total += iou_macro

        scheduler.step()

        train_batches = max(len(train_loader), 1)
        train_loss = train_loss_total / train_batches
        train_dice = train_dice_total / train_batches
        train_iou = train_iou_total / train_batches

        model.eval()
        val_loss_total = 0.0
        val_dice_total = 0.0
        val_iou_total = 0.0

        with torch.no_grad():
            for batch in _progress(val_loader, f"Epoch {epoch}/{args.epochs} [val]"):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True).long()

                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    logits = model(images)
                    loss = multiclass_ce_dice_loss(logits, masks, ce_weight=args.ce_weight)

                val_loss_total += float(loss.item())
                dice_macro, iou_macro, _ = multiclass_dice_iou_from_logits(logits, masks)
                val_dice_total += dice_macro
                val_iou_total += iou_macro

        val_batches = max(len(val_loader), 1)
        val_loss = val_loss_total / val_batches
        val_dice = val_dice_total / val_batches
        val_iou = val_iou_total / val_batches

        epoch_seconds = time.perf_counter() - epoch_start
        lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": float(epoch),
            "learning_rate": lr,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        checkpoint = {
            "epoch": epoch,
            "best_dice": best_dice,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "config": config,
        }
        torch.save(checkpoint, latest_checkpoint_path)

        if val_dice >= best_dice:
            best_dice = val_dice
            checkpoint["best_dice"] = best_dice
            torch.save(checkpoint, best_checkpoint_path)

        save_json({"history": history, "best_dice": best_dice, "config": config}, history_path)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
            f"lr={lr:.6f} time={epoch_seconds:.1f}s"
        )

    total_seconds = time.perf_counter() - total_start

    model_export = project_root / "models" / "brats_3d_unet_best.pt"
    model_export.parent.mkdir(parents=True, exist_ok=True)
    if best_checkpoint_path.exists():
        shutil.copy2(best_checkpoint_path, model_export)

    print("Training complete")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Total training time: {total_seconds / 3600.0:.2f} hours")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    print(f"Exported model: {model_export}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
