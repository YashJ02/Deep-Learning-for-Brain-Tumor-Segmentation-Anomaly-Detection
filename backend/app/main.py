from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .mesh import build_mesh_from_mask
from .metrics import compute_tumor_metrics
from .segmentation import load_nifti_volume, segment_tumor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="BraTS 3D Segmentation Starter API",
    version="1.0.0",
    description="Upload a NIfTI MRI volume and get a tumor mask mesh + measurements.",
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


@app.post("/api/segment")
async def segment(
    file: UploadFile = File(...),
    modality_index: int = Form(3),
    engine: str = Form("auto"),
    threshold: float = Form(0.5),
) -> dict:
    filename = file.filename or ""
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")
    if not (0.0 < float(threshold) < 1.0):
        raise HTTPException(status_code=400, detail="threshold must be in range (0, 1).")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        with tmp_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        try:
            volume, spacing, used_modality = load_nifti_volume(str(tmp_path), modality_index)
            mask, inference_info = segment_tumor(volume, engine=engine, threshold=threshold)
            metrics = compute_tumor_metrics(mask, spacing)
            mesh = build_mesh_from_mask(mask, spacing)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc

    return {
        "status": "ok",
        "input": {
            "filename": filename,
            "volume_shape": [int(x) for x in volume.shape],
            "voxel_spacing_mm": [float(x) for x in spacing],
            "modality_index": int(used_modality),
            "engine_requested": engine,
            "threshold": float(threshold),
        },
        "inference": inference_info,
        "metrics": metrics,
        "mesh": mesh,
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
