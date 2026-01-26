import numpy as np
import pandas as pd
from collections import defaultdict

# Round to numeric value mapping (higher = better result)
ROUND_VALUES = {
    "r128": 1,
    "r64": 2,
    "r32": 3,
    "r16": 4,
    "qf": 5,
    "sf": 6,
    "f": 7,
}
CHAMPION_VALUE = 8  # Winning the final


def get_round_value(round_name: str) -> int:
    return ROUND_VALUES.get(str(round_name).lower(), 0)


def build_tournament_history_features(
    df: pd.DataFrame,
    *,
    winner_col: str = "winner_id",
    loser_col: str = "loser_id",
    date_col: str = "Date",
    tourney_name_col: str = "tournament_location",
    round_col: str = "round",
) -> pd.DataFrame:
    """
    Add tournament history features for winners and losers.

    Features added:
        - winner_best_result_tournament_history: best result winner achieved in this tournament before
        - loser_best_result_tournament_history: best result loser achieved in this tournament before
        - winner_last_result_tournament_history: winner's result from last time in this tournament
        - loser_last_result_tournament_history: loser's result from last time in this tournament
        - winner_average_result_tournament_history: winner's average result in this tournament
        - loser_average_result_tournament_history: loser's average result in this tournament

    Results are numeric: 1=R128, 2=R64, 3=R32, 4=R16, 5=QF, 6=SF, 7=F, 8=Champion
    """
    out = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(out)

    # Track tournament results per player: (player, tourney_name) -> [results]
    player_tourney_results = defaultdict(list)

    # Track current tournament edition results to determine final result
    # (player, tourney_name, year) -> best_round_reached_so_far
    current_edition_best = defaultdict(int)

    # Output arrays
    winner_best = np.zeros(n, dtype=np.float64)
    loser_best = np.zeros(n, dtype=np.float64)
    winner_last = np.zeros(n, dtype=np.float64)
    loser_last = np.zeros(n, dtype=np.float64)
    winner_avg = np.zeros(n, dtype=np.float64)
    loser_avg = np.zeros(n, dtype=np.float64)

    # First pass: determine each player's final result in each tournament edition
    # We need to know the final result before we can use it as history
    edition_results = defaultdict(dict)  # (tourney_name, year) -> {player: result}

    for _, row in df.iterrows():
        w = row[winner_col]
        l = row[loser_col]
        tourney_name = row[tourney_name_col]
        match_date = row[date_col]
        round_val = get_round_value(row[round_col])
        year = match_date.year if hasattr(match_date, 'year') else pd.to_datetime(match_date).year

        edition_key = (tourney_name, year)

        # Winner advances, so their result is at least this round
        # If they win the final, they're champion
        if str(row[round_col]).lower() == "f":
            edition_results[edition_key][w] = CHAMPION_VALUE
        else:
            edition_results[edition_key][w] = max(
                edition_results[edition_key].get(w, 0),
                round_val + 1  # They advance to next round
            )

        # Loser is eliminated at this round
        edition_results[edition_key][l] = max(
            edition_results[edition_key].get(l, 0),
            round_val
        )

    # Build historical results per player per tournament
    # Sort editions by year to build chronological history
    all_editions = sorted(edition_results.keys(), key=lambda x: x[1])

    # (player, tourney_name) -> [(year, result), ...]
    player_history = defaultdict(list)
    for tourney_name, year in all_editions:
        for player, result in edition_results[(tourney_name, year)].items():
            player_history[(player, tourney_name)].append((year, result))

    # Second pass: compute features for each match
    for i, row in out.iterrows():
        w = row[winner_col]
        l = row[loser_col]
        tourney_name = row[tourney_name_col]
        match_date = row[date_col]
        current_year = match_date.year if hasattr(match_date, 'year') else pd.to_datetime(match_date).year

        # Get historical results before current year
        w_history = [r for y, r in player_history.get((w, tourney_name), []) if y < current_year]
        l_history = [r for y, r in player_history.get((l, tourney_name), []) if y < current_year]

        # Winner features
        if w_history:
            winner_best[i] = max(w_history)
            winner_last[i] = w_history[-1]  # Most recent (list is sorted by year)
            winner_avg[i] = sum(w_history) / len(w_history)
        else:
            winner_best[i] = 0
            winner_last[i] = 0
            winner_avg[i] = 0

        # Loser features
        if l_history:
            loser_best[i] = max(l_history)
            loser_last[i] = l_history[-1]
            loser_avg[i] = sum(l_history) / len(l_history)
        else:
            loser_best[i] = 0
            loser_last[i] = 0
            loser_avg[i] = 0

    out["winner_best_result_tournament_history"] = winner_best
    out["loser_best_result_tournament_history"] = loser_best
    out["winner_last_result_tournament_history"] = winner_last
    out["loser_last_result_tournament_history"] = loser_last
    out["winner_average_result_tournament_history"] = winner_avg
    out["loser_average_result_tournament_history"] = loser_avg

    return out
