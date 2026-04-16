# -----yash jain------
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .mesh import build_mesh_from_mask
from .metrics import BRATS_CLASS_DEFINITIONS, compute_class_metrics, compute_tumor_metrics
from .segmentation import (
    available_fold_checkpoints,
    default_checkpoint_path,
    extract_brain_mask,
    load_multimodal_nifti_volumes,
    preferred_deep_checkpoint_path,
    resolve_ensemble_checkpoint_paths,
    segment_tumor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_ROOT = PROJECT_ROOT / "data"

DEMO_PATIENT_DISPLAY_NAMES = [
    "Ava Bennett",
    "Liam Carter",
    "Sofia Hayes",
    "Noah Turner",
    "Mia Foster",
    "Ethan Brooks",
    "Isla Morgan",
    "Lucas Reed",
]

app = FastAPI(
    title="BraTS 3D Segmentation Starter API",
    version="1.0.0",
    description="Upload four BraTS NIfTI modalities (FLAIR/T1/T1ce/T2) and get tumor meshes + measurements.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


def _parse_fold_indices(raw_value: str) -> list[int] | None:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return None

    parsed: list[int] = []
    for token in raw_value.split(","):
        item = token.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise ValueError("Fold indices must be non-negative")
        parsed.append(index)

    if not parsed:
        return None
    return sorted(set(parsed))


def _resolve_case_modality_paths(case_dir: Path) -> Dict[str, Path]:
    modalities = ["flair", "t1", "t1ce", "t2"]
    resolved: Dict[str, Path] = {}

    for modality in modalities:
        matches = sorted(case_dir.glob(f"*_{modality}.nii")) + sorted(case_dir.glob(f"*_{modality}.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"Missing modality {modality} in case directory: {case_dir}")
        resolved[modality] = matches[0]

    return resolved


def _demo_case_dirs() -> list[Path]:
    demo_roots = [
        DATA_ROOT / "demo_cases",
        DATA_ROOT / "demo",
    ]

    roots = [path for path in demo_roots if path.exists() and path.is_dir()]
    if not roots:
        return []

    case_dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child in seen:
                continue
            try:
                _resolve_case_modality_paths(child)
            except FileNotFoundError:
                continue
            case_dirs.append(child)
            seen.add(child)

    return case_dirs


def _demo_display_name(case_index: int) -> str:
    base_name = DEMO_PATIENT_DISPLAY_NAMES[case_index % len(DEMO_PATIENT_DISPLAY_NAMES)]
    cycle = case_index // len(DEMO_PATIENT_DISPLAY_NAMES)
    if cycle == 0:
        return base_name
    return f"{base_name} {cycle + 1}"


def _run_segmentation_with_paths(
    modality_paths: Dict[str, str],
    *,
    engine: str,
    threshold: float,
    requested_fold_indices: list[int] | None,
    filename: str,
    upload_mode: str,
    source_files: Dict[str, str],
) -> dict:
    volume, spacing = load_multimodal_nifti_volumes(modality_paths)

    selected_checkpoints = (
        resolve_ensemble_checkpoint_paths(requested_fold_indices)
        if requested_fold_indices is not None
        else None
    )
    mask, inference_info, class_label_map = segment_tumor(
        volume,
        engine=engine,
        threshold=threshold,
        ensemble_checkpoint_paths=selected_checkpoints,
    )
    inference_info["input_mode"] = "multimodal"

    metrics = compute_tumor_metrics(mask, spacing)
    mesh = build_mesh_from_mask(mask, spacing, target_max_dim=140)
    brain_mask = extract_brain_mask(volume)
    brain_mesh = build_mesh_from_mask(brain_mask, spacing, target_max_dim=156)

    class_metrics: dict = {}
    class_meshes: list[dict] = []
    if class_label_map is not None:
        class_metrics = compute_class_metrics(class_label_map, spacing)
        class_colors = {
            "1": "#f97316",
            "2": "#22c55e",
            "4": "#ef4444",
        }
        for class_label, descriptor in BRATS_CLASS_DEFINITIONS.items():
            label_key = str(class_label)
            class_mask = class_label_map == int(class_label)
            class_mesh = build_mesh_from_mask(class_mask, spacing, target_max_dim=132)
            class_meshes.append(
                {
                    "label": int(class_label),
                    "key": str(descriptor["key"]),
                    "name": str(descriptor["name"]),
                    "color": class_colors.get(label_key, "#f59e0b"),
                    "mesh": class_mesh,
                }
            )

    return {
        "status": "ok",
        "input": {
            "filename": filename,
            "source_files": source_files,
            "upload_mode": upload_mode,
            "volume_shape": [int(x) for x in volume.shape],
            "voxel_spacing_mm": [float(x) for x in spacing],
            "modality_mode": "all",
            "engine_requested": engine,
            "threshold": float(threshold),
            "ensemble_folds_requested": requested_fold_indices,
        },
        "inference": inference_info,
        "metrics": metrics,
        "class_metrics": class_metrics,
        "mesh": mesh,
        "brain_mesh": brain_mesh,
        "class_meshes": class_meshes,
    }


@app.get("/api/checkpoints")
def checkpoint_inventory() -> dict:
    deep_checkpoint = preferred_deep_checkpoint_path()
    deep_path = deep_checkpoint if deep_checkpoint is not None else default_checkpoint_path()
    fold_entries = available_fold_checkpoints()

    return {
        "status": "ok",
        "deep": {
            "path": str(deep_path),
            "exists": deep_checkpoint is not None,
        },
        "ensemble": {
            "count": len(fold_entries),
            "folds": fold_entries,
        },
    }


@app.get("/api/demo-patients")
def demo_patients() -> dict:
    candidates = _demo_case_dirs()[:4]
    if not candidates:
        return {"status": "ok", "patients": []}

    patients: list[dict] = []
    for case_index, case_dir in enumerate(candidates):
        paths = _resolve_case_modality_paths(case_dir)
        patients.append(
            {
                "case_id": case_dir.name,
                "display_name": _demo_display_name(case_index),
                "source": str(case_dir.parent.name),
                "modalities": {
                    key: str(path.name)
                    for key, path in paths.items()
                },
            }
        )

    return {"status": "ok", "patients": patients}


@app.post("/api/segment-demo")
async def segment_demo(
    case_id: str = Form(""),
    engine: str = Form("all"),
    threshold: float = Form(0.5),
    ensemble_folds: str = Form(""),
) -> dict:
    case_id = (case_id or "").strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id is required")
    if not (0.0 < float(threshold) < 1.0):
        raise HTTPException(status_code=400, detail="threshold must be in range (0, 1).")

    try:
        requested_fold_indices = _parse_fold_indices(ensemble_folds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ensemble_folds value: {exc}") from exc

    case_dir = next((path for path in _demo_case_dirs() if path.name == case_id), None)
    if case_dir is None:
        raise HTTPException(status_code=404, detail=f"Unknown demo case_id: {case_id}")

    try:
        modality_paths = _resolve_case_modality_paths(case_dir)
        source_files = {key: path.name for key, path in modality_paths.items()}
        return _run_segmentation_with_paths(
            {key: str(path) for key, path in modality_paths.items()},
            engine=engine,
            threshold=float(threshold),
            requested_fold_indices=requested_fold_indices,
            filename=case_id,
            upload_mode="demo-case",
            source_files=source_files,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid demo case: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc


@app.post("/api/segment")
async def segment(
    flair_file: UploadFile | None = File(None),
    t1_file: UploadFile | None = File(None),
    t1ce_file: UploadFile | None = File(None),
    t2_file: UploadFile | None = File(None),
    engine: str = Form("all"),
    threshold: float = Form(0.5),
    ensemble_folds: str = Form(""),
) -> dict:
    modality_uploads = {
        "flair": flair_file,
        "t1": t1_file,
        "t1ce": t1ce_file,
        "t2": t2_file,
    }

    missing_modalities = [
        name
        for name, upload in modality_uploads.items()
        if upload is None or not (upload.filename or "").strip()
    ]
    if missing_modalities:
        raise HTTPException(
            status_code=400,
            detail="Provide all four modality files: flair, t1, t1ce, t2.",
        )

    def _validate_nii_filename(filename: str) -> bool:
        return filename.endswith(".nii") or filename.endswith(".nii.gz")

    upload_mode = "brats-four-file"
    source_files: Dict[str, str] = {}

    for name, upload in modality_uploads.items():
        if upload is None:
            continue
        filename = (upload.filename or "").strip()
        if not _validate_nii_filename(filename):
            raise HTTPException(
                status_code=400,
                detail=f"Only .nii or .nii.gz files are supported. Invalid file for {name}: {filename}",
            )
        source_files[name] = filename

    filename = "multimodal_bundle"
    if not (0.0 < float(threshold) < 1.0):
        raise HTTPException(status_code=400, detail="threshold must be in range (0, 1).")
    try:
        requested_fold_indices = _parse_fold_indices(ensemble_folds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ensemble_folds value: {exc}") from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        uploaded_paths = {}
        for modality_name, upload in modality_uploads.items():
            if upload is None:
                continue
            target_name = f"{modality_name}_{source_files[modality_name]}"
            modality_path = Path(tmp_dir) / target_name
            with modality_path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            uploaded_paths[modality_name] = str(modality_path)

        try:
            return _run_segmentation_with_paths(
                {
                    "flair": uploaded_paths["flair"],
                    "t1": uploaded_paths["t1"],
                    "t1ce": uploaded_paths["t1ce"],
                    "t2": uploaded_paths["t2"],
                },
                engine=engine,
                threshold=float(threshold),
                requested_fold_indices=requested_fold_indices,
                filename=filename,
                upload_mode=upload_mode,
                source_files=source_files,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
