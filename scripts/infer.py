import argparse
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default_config import Config
from src.pipeline.predictor import TestPredictor


def main():
    parser = argparse.ArgumentParser(description="Run UMUD Inference on Test Ultrasound Cohort")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data")
    parser.add_argument("--checkpoints_dir", type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--submission_path", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--backbone", type=str, default="resnet34")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.checkpoints_dir,
        submission_path=args.submission_path,
        backbone=args.backbone,
    )

    predictor = TestPredictor(cfg)
    predictor.process_test_directory()


if __name__ == "__main__":
    main()
