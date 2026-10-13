import argparse
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default_config import Config
from src.pipeline.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train UMUD Aponeurosis Segmentation Ensemble")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet34", help="timm backbone or standalone")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_folds=args.folds,
    )

    trainer = ModelTrainer(cfg)
    trainer.run_cv()


if __name__ == "__main__":
    main()
