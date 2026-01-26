"""
Runner script to apply all feature creation functions to preprocessed data.

Usage:
    python run_feature_creation.py
    python run_feature_creation.py --input path/to/input.csv --output path/to/output.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent))

from elo_rating import build_elo_features
from fatigue import build_fatigue_features
from h2h import build_h2h_features
from tournament_history import build_tournament_history_features
from round_level_stats import build_round_level_features
from form_score import build_form_score_features


def run_feature_creation(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Load data, apply all feature creation functions, and save results."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=["Date"])
    print(f"Loaded {len(df)} matches")

    initial_cols = set(df.columns)

    # Apply feature creation functions in order
    # ELO must be first since form_score depends on blended_elo
    print("Building ELO features...")
    df = build_elo_features(df)

    print("Building fatigue features...")
    df = build_fatigue_features(df)

    print("Building H2H features...")
    df = build_h2h_features(df)

    print("Building tournament history features...")
    df = build_tournament_history_features(df)

    print("Building round-level stats features...")
    df = build_round_level_features(df)

    print("Building form score features...")
    df = build_form_score_features(df)

    # Report new columns
    new_cols = set(df.columns) - initial_cols
    print(f"\nAdded {len(new_cols)} new feature columns:")
    for col in sorted(new_cols):
        print(f"  - {col}")

    print(f"\nSaving data with features to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Done! Total columns: {len(df.columns)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Add features to preprocessed tennis match data")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(__file__).parent.parent / "preprocessing" / "data" / "atp_matches_preprocessed.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "feature_creation" / "data" / "atp_matches_with_features.csv",
        help="Path to output CSV file"
    )
    args = parser.parse_args()

    run_feature_creation(args.input, args.output)


if __name__ == "__main__":
    main()
