import numpy as np
import pandas as pd
from collections import defaultdict

DECAY_FACTOR = 0.85


def calc_fatigue_score(player_history: list, current_date) -> float:
    """
    Calculate fatigue score based on recent match history.

    Score is sum of: 0.85^((current_date - match_date).days - 1) * minutes
    for all previous matches.
    """
    total = 0.0
    for match_date, minutes in player_history:
        days_diff = (current_date - match_date).days
        if days_diff > 0:
            total += (DECAY_FACTOR ** (days_diff - 1)) * minutes
    return total


def build_fatigue_features(
    df: pd.DataFrame,
    *,
    winner_col: str = "winner_id",
    loser_col: str = "loser_id",
    date_col: str = "Date",
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    """
    Add fatigue score features for winners and losers.

    Features added:
        - winner_fatigue_score: accumulated fatigue for winner before match
        - loser_fatigue_score: accumulated fatigue for loser before match
    """
    out = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(out)

    # Track match history per player: player_id -> [(date, minutes), ...]
    player_history = defaultdict(list)

    # Output arrays
    winner_fatigue = np.zeros(n, dtype=np.float64)
    loser_fatigue = np.zeros(n, dtype=np.float64)

    for i, row in out.iterrows():
        w = row[winner_col]
        l = row[loser_col]
        match_date = row[date_col]
        minutes = row[minutes_col] if pd.notna(row[minutes_col]) else 0.0

        # Calculate fatigue before this match
        winner_fatigue[i] = calc_fatigue_score(player_history[w], match_date)
        loser_fatigue[i] = calc_fatigue_score(player_history[l], match_date)

        # Record this match for both players
        player_history[w].append((match_date, minutes))
        player_history[l].append((match_date, minutes))

    out["winner_fatigue_score"] = winner_fatigue
    out["loser_fatigue_score"] = loser_fatigue

    return out
