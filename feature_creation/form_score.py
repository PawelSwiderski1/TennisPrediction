import numpy as np
import pandas as pd
from collections import defaultdict

DEFAULT_N = 4


def calculate_expected_set_margin(pred: float, surface: str) -> float:
    surface_lower = str(surface).lower()
    if surface_lower == "clay":
        return pred * 1.825 - 0.913
    elif surface_lower == "hard":
        return pred * 1.810 - 0.915
    else:  # grass or other
        return pred * 1.813 - 0.917


def calculate_expected_game_margin(pred: float, surface: str) -> float:
    surface_lower = str(surface).lower()
    if surface_lower == "clay":
        return pred * 1.084 - 0.542
    elif surface_lower == "hard":
        return pred * 1.045 - 0.522
    else:  # grass or other
        return pred * 0.922 - 0.461


def pad_history(values: list, n: int, default=0.0) -> list:
    padded = [default] * (n - len(values)) + values
    return padded


def build_form_score_features(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    winner_col: str = "winner_id",
    loser_col: str = "loser_id",
    date_col: str = "Date",
    surface_col: str = "surface",
    tourney_col: str = "tournament_id",
    best_of_col: str = "best_of",
    comment_col: str = "Comment",
    winner_elo_col: str = "winner_blended_elo",
    loser_elo_col: str = "loser_blended_elo",
    w_sets_col: str = "Wsets",
    l_sets_col: str = "Lsets",
    w_game_cols: tuple = ("W1", "W2", "W3", "W4", "W5"),
    l_game_cols: tuple = ("L1", "L2", "L3", "L4", "L5"),
) -> pd.DataFrame:
    """
    Add form score features tracking last N matches for each player.
    Walkovers are excluded from form history.
    """
    out = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    num_rows = len(out)

    # Check which columns exist
    has_sets = w_sets_col in out.columns and l_sets_col in out.columns
    has_games = all(c in out.columns for c in w_game_cols) and all(c in out.columns for c in l_game_cols)
    has_elo = winner_elo_col in out.columns and loser_elo_col in out.columns

    player_histories = defaultdict(list)

    output_cols = {
        f"{role}_{feat}": [None] * num_rows
        for role in ["winner", "loser"]
        for feat in [
            "last_preds", "last_base_perfs",
            "last_set_margin_norm", "last_game_margin_norm", "last_margin_surplus",
            "last_best_of_3",
            "last_days_since", "last_opponent_elo",
            "last_same_surface", "last_same_tournament"
        ]
    }

    for i, row in out.iterrows():
        w = row[winner_col]
        l = row[loser_col]
        match_date = row[date_col]
        surface = str(row[surface_col])
        tournament = str(row[tourney_col])
        best_of = row[best_of_col] if best_of_col in out.columns else 3
        is_walkover = str(row.get(comment_col, "")).lower() == "walkover"

        # Calculate set and game margins
        if has_sets:
            w_sets = row[w_sets_col] if pd.notna(row[w_sets_col]) else 0
            l_sets = row[l_sets_col] if pd.notna(row[l_sets_col]) else 0
            winner_set_margin = (w_sets - l_sets) / (2 if best_of == 3 else 3)
        else:
            winner_set_margin = 0.0

        if has_games:
            w_games = sum(row[c] if pd.notna(row.get(c, np.nan)) else 0 for c in w_game_cols)
            l_games = sum(row[c] if pd.notna(row.get(c, np.nan)) else 0 for c in l_game_cols)
            winner_game_margin = (w_games - l_games) / (12 if best_of == 3 else 18)
        else:
            winner_game_margin = 0.0

        # Get ELO ratings
        if has_elo:
            w_elo = row[winner_elo_col] if pd.notna(row.get(winner_elo_col, np.nan)) else 1500.0
            l_elo = row[loser_elo_col] if pd.notna(row.get(loser_elo_col, np.nan)) else 1500.0
        else:
            w_elo, l_elo = 1500.0, 1500.0

        # Process each role (winner and loser)
        for role, player_id, opponent_elo in [("winner", w, l_elo), ("loser", l, w_elo)]:
            # Get recent history (excluding walkovers)
            history = player_histories[player_id]
            recent = []
            for m in reversed(history):
                if m["date"] < match_date and not m.get("is_walkover", False):
                    recent.append(m)
                    if len(recent) == n:
                        break
            recent = list(reversed(recent))

            # Extract and pad features
            output_cols[f"{role}_last_preds"][i] = pad_history([m["pred_prob"] for m in recent], n)
            output_cols[f"{role}_last_base_perfs"][i] = pad_history([m["base_perf"] for m in recent], n)
            output_cols[f"{role}_last_set_margin_norm"][i] = pad_history([m["set_margin"] for m in recent], n)
            output_cols[f"{role}_last_game_margin_norm"][i] = pad_history([m["game_margin"] for m in recent], n)
            output_cols[f"{role}_last_margin_surplus"][i] = pad_history([m["margin_surplus"] for m in recent], n)
            output_cols[f"{role}_last_best_of_3"][i] = pad_history([m["best_of_3"] for m in recent], n)
            output_cols[f"{role}_last_opponent_elo"][i] = pad_history([m["opponent_elo"] for m in recent], n)

            # Days since each match
            days_since = []
            for m in recent:
                try:
                    days = (match_date - m["date"]).days
                except:
                    days = 0
                days_since.append(days)
            output_cols[f"{role}_last_days_since"][i] = pad_history(days_since, n, default=0)

            same_surface = [1 if str(m["surface"]).lower() == surface.lower() else 0 for m in recent]
            output_cols[f"{role}_last_same_surface"][i] = pad_history(same_surface, n, default=0)

            same_tournament = [1 if m["tournament"] == tournament else 0 for m in recent]
            output_cols[f"{role}_last_same_tournament"][i] = pad_history(same_tournament, n, default=0)

            elo_diff = w_elo - l_elo
            base_prob = 1 / (1 + 10 ** (-elo_diff / 400))
            pred_prob = base_prob if role == "winner" else 1 - base_prob

            outcome = 1.0 if role == "winner" else 0.0
            base_perf = outcome - pred_prob

            if role == "winner":
                set_margin = winner_set_margin
                game_margin = winner_game_margin
            else:
                set_margin = -winner_set_margin
                game_margin = -winner_game_margin

            expected_set = calculate_expected_set_margin(pred_prob, surface)
            expected_game = calculate_expected_game_margin(pred_prob, surface)
            actual_total = set_margin + 0.25 * game_margin
            expected_total = expected_set + 0.25 * expected_game
            margin_surplus = actual_total - expected_total

            player_histories[player_id].append({
                "date": match_date,
                "pred_prob": pred_prob,
                "base_perf": base_perf,
                "is_winner": outcome,
                "set_margin": set_margin,
                "game_margin": game_margin,
                "margin_surplus": margin_surplus,
                "best_of_3": 1.0 if best_of == 3 else 0.0,
                "surface": surface,
                "tournament": tournament,
                "opponent_elo": opponent_elo,
                "is_walkover": is_walkover,
            })

    for col_name, values in output_cols.items():
        out[col_name] = values

    return out
