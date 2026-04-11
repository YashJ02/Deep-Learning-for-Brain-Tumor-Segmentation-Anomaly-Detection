# NeuroScope 3D: CUDA Brain Tumor Segmentation Pipeline

This project is a full 3D MRI tumor segmentation pipeline for BraTS-style NIfTI data.

It includes:

1. FastAPI web application with 3D mesh visualization.
2. Dataset split generation for BraTS folders.
3. CUDA-ready 3D U-Net training script.
4. Validation and report generation.
5. Single-volume prediction CLI.
6. HPC Slurm templates for Northeastern-style cluster workflows.

## Repository Layout

- `backend/app/main.py`: API routes and web UI hosting.
- `backend/app/segmentation.py`: baseline segmentation + deep checkpoint auto-switch.
- `backend/app/metrics.py`: volumetric metrics from predicted mask.
- `backend/app/mesh.py`: marching-cubes mesh extraction for viewer.
- `frontend/`: browser UI (upload + metrics + interactive 3D mesh).
- `training/model.py`: 3D U-Net architecture.
- `training/torch_dataset.py`: PyTorch dataset loader for BraTS volumes.
- `training/losses.py`: BCE + Dice training loss.
- `training/metrics.py`: Dice and IoU metrics.
- `training/inference.py`: checkpoint loading and 3D inference utilities.
- `scripts/prepare_brats_dataset.py`: generate `all.csv`, `train.csv`, `val.csv`.
- `scripts/train_brats_3d_unet.py`: main CUDA training entrypoint.
- `scripts/evaluate_brats_3d_unet.py`: validation report script.
- `scripts/predict_brats_3d_unet.py`: prediction for one input NIfTI file.
- `hpc/`: Slurm templates and usage notes.

## Installation

Base runtime dependencies (API + preprocessing):

```bash
pip install -r requirements.txt
```

Training dependencies (PyTorch + tqdm):

```bash
pip install -r requirements-train.txt
```

### CUDA PyTorch Note

On many systems you should install PyTorch using the official CUDA wheel command from:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

If needed, install `torch`, `torchvision`, and `torchaudio` first via that command, then install the rest:

```bash
pip install -r requirements.txt
pip install tqdm
```

## Dataset Download

Download BraTS bundle directly into `data/`:

```bash
python scripts/download_brats_dataset.py --output-dir data --force
```

Expected training folder example:

- `data/MICCAI_BraTS2020_TrainingData`

## Full Training Pipeline

### 1) Build train/validation split CSV files

```bash
python scripts/prepare_brats_dataset.py --data-root data/MICCAI_BraTS2020_TrainingData --output-dir data/splits --val-ratio 0.2 --seed 42
```

### 2) Train 3D U-Net (CUDA)

```bash
python scripts/train_brats_3d_unet.py --train-csv data/splits/train.csv --val-csv data/splits/val.csv --epochs 120 --batch-size 1 --num-workers 8 --modality t1ce --target-shape 128 128 128 --amp
```

Outputs:

- `models/checkpoints/best.pt`
- `models/checkpoints/latest.pt`
- `models/checkpoints/history.json`
- `models/brats_3d_unet_best.pt`

### 3) Evaluate best checkpoint

```bash
python scripts/evaluate_brats_3d_unet.py --csv data/splits/val.csv --checkpoint models/checkpoints/best.pt --device auto
```

Output report:

- `reports/eval_*.json`

### 4) Predict one patient volume

```bash
python scripts/predict_brats_3d_unet.py --input data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii --checkpoint models/checkpoints/best.pt --modality-index 0 --threshold 0.5
```

Default output:

- `models/predictions/<input_name>_mask.nii.gz`

## Run the Web App

Start server:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- [http://localhost:8000](http://localhost:8000)

API route:

- `POST /api/segment`

Form fields:

- `file`: `.nii` or `.nii.gz`
- `modality_index`: integer for 4D volumes
- `engine`: `auto`, `deep`, `baseline`
- `threshold`: float in range `(0, 1)`

Inference behavior:

1. `auto`: uses deep model if checkpoint exists, otherwise baseline.
2. `deep`: requires checkpoint, fails if unavailable.
3. `baseline`: uses non-DL threshold+morphology pipeline.

## HPC (Slurm) Usage

Templates:

- `hpc/slurm_train_3d_unet.sh`
- `hpc/slurm_eval_3d_unet.sh`

Submit training:

```bash
sbatch hpc/slurm_train_3d_unet.sh
```

Submit evaluation:

```bash
sbatch hpc/slurm_eval_3d_unet.sh
```

Cluster-specific options (partition/module names/account) should be updated in those scripts.

## Practical Notes

1. For 16 GB GPUs, keep batch size at `1` with `128^3` target shape.
2. For large GPUs (H100), increase workers and evaluate larger patch/target sizes.
3. Save checkpoints frequently and keep `best.pt` for deployment.
4. For strongest research results, run multiple folds on HPC and ensemble outputs.

## Disclaimer

This project is for research and educational use only. It is not a clinical diagnosis tool.

## BraTS Citation

If you use BraTS data in research, cite official BraTS publications:

1. Menze et al., IEEE TMI 2015, DOI: 10.1109/TMI.2014.2377694.
2. Bakas et al., Scientific Data 2017, DOI: 10.1038/sdata.2017.117.
3. Bakas et al., arXiv:1811.02629 (2018).
