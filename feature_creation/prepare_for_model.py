"""
Prepare feature CSV for model training.

Transforms:
1. Convert hand to is_right_handed (R/U -> 1, L -> 0)
2. One-hot encode surface and tourney_level
3. Rename winner_/loser_ to player1_/player2_
4. Randomly swap player1/player2 in half the rows to avoid positional bias
5. Add target column (1 if player1 won, 0 if player2 won)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def convert_hand_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for prefix in ['winner', 'loser']:
        col = f'{prefix}_hand'
        if col in out.columns:
            out[f'{prefix}_is_right_handed'] = out[col].isin(['R', 'U']).astype(int)
            out = out.drop(columns=[col])

    return out


def one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if 'surface' in out.columns:
        out = pd.get_dummies(out, columns=['surface'], prefix='surface')

    if 'tournament_level' in out.columns:
        out = pd.get_dummies(out, columns=['tournament_level'], prefix='tournament_level')

    if 'indoor_or_outdoor' in out.columns:
        out['outdoor'] = (out['indoor_or_outdoor'] == 'Outdoor').astype(int)
        out = out.drop(columns=['indoor_or_outdoor'])

    return out


def get_column_pairs(df: pd.DataFrame) -> list:
    winner_cols = [c for c in df.columns if c.startswith('winner_')]
    pairs = []

    for w_col in winner_cols:
        suffix = w_col[7:]  # Remove 'winner_' prefix
        l_col = f'loser_{suffix}'
        if l_col in df.columns:
            pairs.append((w_col, l_col, suffix))

    return pairs


def prepare_for_model(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:

    out = df.copy()

    out = convert_hand_to_binary(out)

    out = one_hot_encode_categoricals(out)

    if 'AvgW' in out.columns:
        out = out.rename(columns={'AvgW': 'winner_bet_odds'})
    if 'AvgL' in out.columns:
        out = out.rename(columns={'AvgL': 'loser_bet_odds'})

    pairs = get_column_pairs(out)

    # Determine which rows to swap (random half)
    np.random.seed(seed)
    n_rows = len(out)
    swap_mask = np.random.rand(n_rows) < 0.5

    # Create target column (before swapping, winner is always player1)
    # After swap: if swapped, player2 is the winner so target=0
    out['target'] = (~swap_mask).astype(int)

    # Rename and swap columns
    for w_col, l_col, suffix in pairs:
        p1_col = f'player1_{suffix}'
        p2_col = f'player2_{suffix}'

        winner_vals = out[w_col].values.copy()
        loser_vals = out[l_col].values.copy()

        p1_vals = np.where(swap_mask, loser_vals, winner_vals)
        p2_vals = np.where(swap_mask, winner_vals, loser_vals)

        out[p1_col] = p1_vals
        out[p2_col] = p2_vals

        out = out.drop(columns=[w_col, l_col])

    # Handle non-paired columns (keep columns that don't have winner_/loser_ prefix)
    # These are already handled by not being in pairs
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare feature CSV for model training")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(__file__).parent.parent / "feature_creation" / "data" / "atp_matches_with_features.csv",
        help="Path to input CSV file with features"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "feature_creation" / "data" / "atp_matches_model_ready.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} matches")

    print("Preparing data for model...")
    df = prepare_for_model(df, seed=args.seed)

    print(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)

    print(f"Done! Shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
