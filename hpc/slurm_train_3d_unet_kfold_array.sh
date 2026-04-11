#!/bin/bash
#SBATCH --job-name=brats3d-kfold
#SBATCH --partition=gpu
#SBATCH --account=your_nurc_project
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/kfold_train_%A_%a.out

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/Deep-Learning-for-Brain-Tumor-Segmentation-Anomaly-Detection}
cd "$PROJECT_DIR"

mkdir -p logs reports models/kfold data/splits/folds

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt

FOLD_INDEX=${SLURM_ARRAY_TASK_ID}

# Build fold CSV files once using array task 0; other tasks wait for completion.
if [[ ! -f data/splits/folds/fold_0/train.csv ]]; then
  if [[ "${FOLD_INDEX}" == "0" ]]; then
    python scripts/prepare_brats_kfold_dataset.py \
      --data-root data/MICCAI_BraTS2020_TrainingData \
      --output-dir data/splits/folds \
      --n-splits 5 \
      --seed 42
  else
    while [[ ! -f data/splits/folds/fold_0/train.csv ]]; do
      sleep 10
    done
  fi
fi

python scripts/train_brats_3d_unet.py \
  --train-csv data/splits/folds/fold_${FOLD_INDEX}/train.csv \
  --val-csv data/splits/folds/fold_${FOLD_INDEX}/val.csv \
  --checkpoint-dir models/kfold/fold_${FOLD_INDEX} \
  --task multiclass \
  --epochs 150 \
  --batch-size 1 \
  --num-workers 8 \
  --modality t1ce \
  --target-shape 128 128 128 \
  --amp \
  --device auto
