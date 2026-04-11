# HPC Job Templates

This folder contains Slurm templates for training and evaluation.

## Training job

Submit:

```bash
sbatch hpc/slurm_train_3d_unet.sh
```

What it does:

1. Creates `logs/`, `reports/`, `models/checkpoints/`, and `data/splits/`.
2. Loads CUDA and Python modules.
3. Creates a virtual environment and installs training dependencies.
4. Builds train/validation split CSV files.
5. Trains the 3D U-Net checkpoint.

## Evaluation job

Submit:

```bash
sbatch hpc/slurm_eval_3d_unet.sh
```

What it does:

1. Loads modules and activates environment.
2. Evaluates `models/checkpoints/best.pt` on `data/splits/val.csv`.
3. Writes a JSON report into `reports/`.

## Customize for your cluster

- `--partition` and module names vary by cluster.
- If your site uses `conda`, swap the venv lines for your conda environment commands.
- For multi-GPU training, run one fold per job and assign one GPU per job.
