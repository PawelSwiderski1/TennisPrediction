import numpy as np
import pandas as pd
from collections import defaultdict


def build_h2h_features(
    df: pd.DataFrame,
    *,
    winner_col: str = "winner_id",
    loser_col: str = "loser_id",
    date_col: str = "Date",
    surface_col: str = "surface",
) -> pd.DataFrame:
    """Add head-to-head win counts (overall and surface-specific)."""
    out = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(out)

    # Track H2H record: (player_a, player_b) -> wins for player_a
    # We use sorted tuple as key so (A, B) and (B, A) map to same matchup
    h2h_record = defaultdict(lambda: defaultdict(int))

    # Track surface-specific H2H: surface -> (player_a, player_b) -> wins for player_a
    h2h_surface_record = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Output arrays
    winner_h2h_wins = np.zeros(n, dtype=np.int32)
    loser_h2h_wins = np.zeros(n, dtype=np.int32)
    winner_h2h_surface_wins = np.zeros(n, dtype=np.int32)
    loser_h2h_surface_wins = np.zeros(n, dtype=np.int32)

    for i, row in out.iterrows():
        w = row[winner_col]
        l = row[loser_col]
        surface = str(row[surface_col]).lower()

        # Get H2H record before this match
        winner_h2h_wins[i] = h2h_record[w][l]
        loser_h2h_wins[i] = h2h_record[l][w]

        # Get surface-specific H2H record before this match
        winner_h2h_surface_wins[i] = h2h_surface_record[surface][w][l]
        loser_h2h_surface_wins[i] = h2h_surface_record[surface][l][w]

        # Update records after this match (winner won)
        h2h_record[w][l] += 1
        h2h_surface_record[surface][w][l] += 1

    out["winner_h2h_wins"] = winner_h2h_wins
    out["loser_h2h_wins"] = loser_h2h_wins
    out["winner_h2h_surface_wins"] = winner_h2h_surface_wins
    out["loser_h2h_surface_wins"] = loser_h2h_surface_wins

    return out
