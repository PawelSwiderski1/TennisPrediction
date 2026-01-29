import pandas as pd
import numpy as np
from elo_rating import build_elo_features

DATA_PATH = "../preprocessing/data/atp_matches_preprocessed.csv"


def rank_expected(rank_w, rank_l) -> float:
    if pd.isna(rank_w) or pd.isna(rank_l):
        return 0.5
    # Log ratio of ranks (lower rank = better, so l/w)
    log_ratio = np.log(rank_l / rank_w)
    return 1.0 / (1.0 + np.exp(-0.5 * log_ratio))


def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"Total matches: {len(df)}")

    print("\nBuilding Elo features...")
    df = build_elo_features(df)

    # Ground truth: winner always won (y=1)
    y_true = np.ones(len(df))

    # Elo prediction (probability that winner wins)
    elo_pred = df["elo_pwin"].values
    surface_elo_pred = df["surface_elo_pwin"].values
    blended_elo_pred = df["blended_elo_pwin"].values

    # Rank-based prediction
    rank_pred = np.array([
        rank_expected(row["winner_rank"], row["loser_rank"])
        for _, row in df.iterrows()
    ])

    # Filter out matches with missing rank data for fair comparison
    valid_rank = ~(df["winner_rank"].isna() | df["loser_rank"].isna())
    print(f"Matches with valid ranks: {valid_rank.sum()}")

    # Top 50 players only (both players ranked < 50)
    top50_mask = valid_rank & (df["winner_rank"] < 50) & (df["loser_rank"] < 50)
    print(f"Matches with both players in top 50: {top50_mask.sum()}")

    print("\n" + "=" * 50)
    print("BRIER SCORES (lower is better)")
    print("=" * 50)

    print("\n--- All matches ---")
    print(f"Elo:              {brier_score(y_true, elo_pred):.4f}")
    print(f"Surface Elo:      {brier_score(y_true, surface_elo_pred):.4f}")
    print(f"Blended Elo:      {brier_score(y_true, blended_elo_pred):.4f}")
    print(f"ATP Rank:         {brier_score(y_true[valid_rank], rank_pred[valid_rank]):.4f}")

    # Compare on same subset (valid ranks only)
    print("\n--- Matches with valid ranks only ---")
    print(f"Elo:              {brier_score(y_true[valid_rank], elo_pred[valid_rank]):.4f}")
    print(f"Surface Elo:      {brier_score(y_true[valid_rank], surface_elo_pred[valid_rank]):.4f}")
    print(f"Blended Elo:      {brier_score(y_true[valid_rank], blended_elo_pred[valid_rank]):.4f}")
    print(f"ATP Rank:         {brier_score(y_true[valid_rank], rank_pred[valid_rank]):.4f}")

    # Top 50 only
    print("\n--- Top 50 players only (both players ranked < 50) ---")
    print(f"Elo:              {brier_score(y_true[top50_mask], elo_pred[top50_mask]):.4f}")
    print(f"Surface Elo:      {brier_score(y_true[top50_mask], surface_elo_pred[top50_mask]):.4f}")
    print(f"Blended Elo:      {brier_score(y_true[top50_mask], blended_elo_pred[top50_mask]):.4f}")
    print(f"ATP Rank:         {brier_score(y_true[top50_mask], rank_pred[top50_mask]):.4f}")

    # Top 50 by surface
    print("\n--- Top 50 by surface ---")
    for surface in df["surface"].unique():
        mask = (df["surface"] == surface) & top50_mask
        if mask.sum() == 0:
            continue
        print(f"\n{surface} ({mask.sum()} matches):")
        print(f"  Elo:         {brier_score(y_true[mask], elo_pred[mask]):.4f}")
        print(f"  Surface Elo: {brier_score(y_true[mask], surface_elo_pred[mask]):.4f}")
        print(f"  Blended Elo: {brier_score(y_true[mask], blended_elo_pred[mask]):.4f}")
        print(f"  ATP Rank:    {brier_score(y_true[mask], rank_pred[mask]):.4f}")

    # Top 50 by tournament level
    print("\n--- Top 50 by tournament level ---")
    for level in df["tournament_level"].unique():
        mask = (df["tournament_level"] == level) & top50_mask
        if mask.sum() == 0:
            continue
        print(f"\n{level} ({mask.sum()} matches):")
        print(f"  Elo:         {brier_score(y_true[mask], elo_pred[mask]):.4f}")
        print(f"  Surface Elo: {brier_score(y_true[mask], surface_elo_pred[mask]):.4f}")
        print(f"  Blended Elo: {brier_score(y_true[mask], blended_elo_pred[mask]):.4f}")
        print(f"  ATP Rank:    {brier_score(y_true[mask], rank_pred[mask]):.4f}")

    # By surface
    print("\n--- By surface (valid ranks) ---")
    for surface in df["surface"].unique():
        mask = (df["surface"] == surface) & valid_rank
        if mask.sum() == 0:
            continue
        print(f"\n{surface} ({mask.sum()} matches):")
        print(f"  Elo:         {brier_score(y_true[mask], elo_pred[mask]):.4f}")
        print(f"  Surface Elo: {brier_score(y_true[mask], surface_elo_pred[mask]):.4f}")
        print(f"  Blended Elo: {brier_score(y_true[mask], blended_elo_pred[mask]):.4f}")
        print(f"  ATP Rank:    {brier_score(y_true[mask], rank_pred[mask]):.4f}")

    # By tournament level
    print("\n--- By tournament level (valid ranks) ---")
    for level in df["tournament_level"].unique():
        mask = (df["tournament_level"] == level) & valid_rank
        if mask.sum() == 0:
            continue
        print(f"\n{level} ({mask.sum()} matches):")
        print(f"  Elo:         {brier_score(y_true[mask], elo_pred[mask]):.4f}")
        print(f"  Surface Elo: {brier_score(y_true[mask], surface_elo_pred[mask]):.4f}")
        print(f"  Blended Elo: {brier_score(y_true[mask], blended_elo_pred[mask]):.4f}")
        print(f"  ATP Rank:    {brier_score(y_true[mask], rank_pred[mask]):.4f}")


if __name__ == "__main__":
    main()
