#!/bin/bash
#SBATCH --job-name=brats-download
#SBATCH --partition=short
#SBATCH --account=your_nurc_project
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_%j.out

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/Deep-Learning-for-Brain-Tumor-Segmentation-Anomaly-Detection}
cd "$PROJECT_DIR"

mkdir -p logs data

module purge
module load explorer anaconda3/2024.06

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install kagglehub

python scripts/download_brats_dataset.py \
  --dataset-id awsaf49/brats20-dataset-training-validation \
  --output-dir data/MICCAI_BraTS2020_TrainingData
