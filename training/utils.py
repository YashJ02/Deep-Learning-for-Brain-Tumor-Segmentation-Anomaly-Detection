# -----yash jain------
"""General helpers shared by training and evaluation scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def file_sha256(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fingerprint(csv_path: Path | str) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))

    return {
        "path": str(csv_path.resolve()),
        "sha256": file_sha256(csv_path),
        "rows": int(row_count),
    }


def git_commit(project_root: Path | str) -> str | None:
    root = Path(project_root)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


def environment_metadata(device: torch.device | str) -> Dict[str, Any]:
    resolved_device = str(device)
    metadata: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": resolved_device,
    }

    if torch.cuda.is_available():
        metadata.update(
            {
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cudnn_version": int(torch.backends.cudnn.version() or 0),
            }
        )

    return metadata
