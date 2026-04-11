from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "MICCAI_BraTS2020_TrainingData"
    model_out = project_root / "models" / "brats_3d_unet.pt"

    print("BraTS 3D training stub")
    print(f"Expected dataset root: {data_root}")
    print(f"Planned model output: {model_out}")
    print("\nNext steps:")
    print("1) Install MONAI + PyTorch with GPU support")
    print("2) Build train/val patient split")
    print("3) Add 3D UNet training loop")
    print("4) Export inference-ready model and connect it in backend/app/segmentation.py")


if __name__ == "__main__":
    main()
