# NeuroScope 3D: BraTS Segmentation Starter

This is a brand-new project focused on 3D brain tumor localization from MRI NIfTI volumes.

## What This App Does

- Accepts `.nii` and `.nii.gz` MRI volume uploads
- Runs a baseline 3D tumor segmentation pipeline
- Computes tumor metrics (volume, centroid, extent, occupancy)
- Generates a rotatable 3D mesh mask in a web UI

## Project Structure

- backend/app/main.py: FastAPI server and API routes
- backend/app/segmentation.py: NIfTI loading and baseline segmentation
- backend/app/metrics.py: tumor measurement calculations
- backend/app/mesh.py: marching-cubes mesh extraction
- frontend/index.html: web interface
- frontend/app.js: API integration and 3D rendering
- frontend/styles.css: styling
- scripts/train_brats_3d_unet_stub.py: starter training scaffold for future model upgrades

## Quick Start

1. Install dependencies:

   pip install -r requirements.txt

2. Run API server:

   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

3. Open browser:

   http://localhost:8000

## API

- GET /api/health
- POST /api/segment
  - form-data:
    - file: `.nii` or `.nii.gz`
    - modality_index: integer (for 4D MRI volumes, default 3)

## Notes

- Current segmentation uses a robust baseline (thresholding + morphology + largest component), not a trained deep model.
- For production-quality segmentation, replace the baseline with a MONAI/nnU-Net model trained on BraTS.
- For true 3D clinical metrics, keep original voxel spacing metadata from NIfTI headers.

## BraTS Citations

If you use BraTS data in research, cite the official papers:

1. Menze et al., IEEE TMI 2015, DOI: 10.1109/TMI.2014.2377694
2. Bakas et al., Scientific Data 2017, DOI: 10.1038/sdata.2017.117
3. Bakas et al., arXiv:1811.02629 (2018)
# Deep Learning for Brain Tumor Detection and Region Visualization

This project provides a desktop GUI application (Tkinter + TensorFlow/Keras) for:

- Binary tumor detection (tumor vs no tumor)
- Multi-class tumor classification (glioma, meningioma, no tumor, pituitary)
- Tumor-region visualization using image processing and watershed segmentation
- Automatic brain health report generation (JSON + TXT)
- Image metadata display (resolution, channels, file size)

## Updated Project Structure

```text
Deep-Learning-for-Brain-Tumor-Segmentation-Anomaly-Detection/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- models/
|   |-- brain_tumor_detector.h5
|   `-- tumor_classifier.h5
|-- reports/                    # auto-generated at runtime
`-- src/
    |-- __init__.py
    |-- main.py
    |-- core/
    |   |-- __init__.py
    |   |-- classification.py
    |   |-- display_tumor.py
    |   |-- predict_tumor.py
    |   `-- reporting.py
    `-- ui/
        |-- __init__.py
        `-- frames.py
```

## Requirements

- Python 3.10 or 3.11 recommended (best TensorFlow compatibility)
- Pip
- Trained model files present in `models/`

Python packages are listed in `requirements.txt`:

- tensorflow
- tf-keras
- numpy
- opencv-python
- Pillow
- imutils

## Installation

1. Clone or download the repository.
2. Open terminal in the project root.
3. Create a virtual environment.
4. Install dependencies.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Application

From the project root:

```bash
python src/main.py
```

## How to Use

1. Click **Browse** and select an MRI image (`.jpg`, `.jpeg`, `.png`).
2. Select an analysis mode: **Detect Tumor**, **Classification**, or **View Tumor Region**.
3. Use **View** / **Back** in the visualization workflow when applicable.
4. A timestamped brain health report is automatically generated in `reports/` after each analysis.

## Module Overview

- `src/main.py`
  - GUI entrypoint.
  - Handles image selection and user actions.
- `src/ui/frames.py`
  - Manages multi-frame window behavior and image rendering.
- `src/core/predict_tumor.py`
  - Binary tumor detection model inference with confidence output.
- `src/core/classification.py`
  - Multi-class tumor type inference with confidence and class probabilities.
- `src/core/display_tumor.py`
  - Preprocessing and intensity-guided tumor-region extraction with contour boundary visualization.
  - Segmentation statistics used for anomaly summary reports.
- `src/core/reporting.py`
  - Brain health report generation and file export (`.json`, `.txt`).

## Model Files

Expected model file locations:

- `models/brain_tumor_detector.h5` (binary detector)
- `models/tumor_classifier.h5` (4-class classifier)

## Model Technical Details

The following details were extracted directly from the `.h5` model artifacts.

### 1) Binary Detector (`brain_tumor_detector.h5`)

- File size: `0.164 MB`
- Keras version: `2.2.4-tf`
- Backend: `tensorflow`
- Input shape: `(240, 240, 3)`
- Output layer: `Dense(1, activation='sigmoid')`
- Loss: `binary_crossentropy`
- Training metric configured: `accuracy`
- Optimizer (saved config): `Adam`
- Approx. parameter count (from saved weights): `11,137`

Inference behavior in code:

- Converts BGR image to grayscale for contour-based ROI extraction.
- Crops ROI using extreme contour points.
- Resizes ROI to `(240, 240)` and normalizes to `[0, 1]`.
- Uses threshold `0.5` on sigmoid output to show Tumor/No Tumor.

### 2) Multi-Class Classifier (`tumor_classifier.h5`)

- File size: `282.336 MB`
- Keras version: `2.11.0`
- Backend: `tensorflow`
- Input shape: `(128, 128, 3)`
- Output layer: `Dense(4, activation='softmax')`
- Loss: `categorical_crossentropy`
- Training metric configured: `categorical accuracy`
- Optimizer (saved config): `Adam` (custom serialized name in model config)
- Approx. parameter count (from saved weights): `40,367,492`

Class index mapping used in app:

- `0 -> Glioma`
- `1 -> Meningioma`
- `2 -> No Tumor`
- `3 -> Pituitary`

## Troubleshooting

- **ModuleNotFoundError**:
  - Run from project root: `python src/main.py`
  - Ensure dependencies are installed in the active environment.

- **Model file not found**:
  - Confirm `.h5` files exist under `models/` with exact names listed above.

- **No prediction before selecting image**:
  - The app now blocks prediction/classification until an image is selected.
  - Reports are written under `reports/` automatically; if report creation fails, verify write permission in project folder.

- **TensorFlow installation issues**:
  - Upgrade pip first: `pip install --upgrade pip`
  - If needed, install a TensorFlow version compatible with your Python version.

## Notes

- This project is intended for educational/research workflows.
- It is not a clinical diagnostic tool.
- Security/authentication and full clinical compliance are out of scope for this academic implementation.

## License

No explicit license file was found in this repository. Add a `LICENSE` file if you want to define usage and distribution permissions.
