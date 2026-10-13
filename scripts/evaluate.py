import argparse
import sys
import os
import pandas as pd

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.metrics import compute_umud_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate Predictions Using Official UMUD Score")
    parser.add_argument("--solution", type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission CSV")
    parser.add_argument("--tau_pa", type=float, default=6.0)
    parser.add_argument("--tau_fl", type=float, default=12.0)
    parser.add_argument("--tau_mt", type=float, default=3.0)
    args = parser.parse_args()

    sol_df = pd.read_csv(args.solution)
    sub_df = pd.read_csv(args.submission)

    score = compute_umud_score(
        sol_df, sub_df,
        tau_pa=args.tau_pa,
        tau_fl=args.tau_fl,
        tau_mt=args.tau_mt
    )
    print("==========================================")
    print(f"Official UMUD Evaluation Score: {score:.5f}")
    print("==========================================")


if __name__ == "__main__":
    main()
