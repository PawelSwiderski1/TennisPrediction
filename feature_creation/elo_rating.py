import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import timedelta

ELO_BASE = 1500.0
ELO_SCALE = 400.0

# Absence penalty settings
ABSENCE_THRESHOLD_WEEKS = 8
ABSENCE_PENALTY_PER_WEEK = 4.0
ABSENCE_MIN_RATING = 1700  # only apply penalty if rating > this

# Initial Elo based on ATP rank thresholds: (max_rank, initial_elo)
RANK_TO_INITIAL_ELO = [
    (10, 2000),
    (50, 1900),
    (100, 1800),
    (200, 1700),
    (300, 1600),
    (400, 1500),
]
DEFAULT_INITIAL_ELO = 1400

# K-factor multiplier by tournament level
TOURNAMENT_K_MULTIPLIER = {
    "grand slam": 1.2,
    "masters 1000": 1.1,
    "atp500": 1.0,
    "atp250": 0.9,
}

# Blended rating settings
SURFACE_WEIGHT_LOW = 0.2    # weight when < threshold matches
SURFACE_WEIGHT_HIGH = 0.4   # weight when >= threshold matches
SURFACE_MATCH_THRESHOLD = 10


def initial_elo_from_rank(rank) -> float:
    if rank is None or pd.isna(rank):
        return ELO_BASE
    for max_rank, elo in RANK_TO_INITIAL_ELO:
        if rank <= max_rank:
            return elo
    return DEFAULT_INITIAL_ELO


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / ELO_SCALE))


def dynamic_k_factor(match_dates: list, current_date, lookback_days: int) -> float:
    cutoff = current_date - timedelta(days=lookback_days)
    recent = sum(1 for d in match_dates if d > cutoff)
    return 210 / ((recent + 5) ** 0.5)


def blended_rating(overall_rating, surface_rating, num_surface_matches):
    weight = SURFACE_WEIGHT_HIGH if num_surface_matches >= SURFACE_MATCH_THRESHOLD else SURFACE_WEIGHT_LOW
    return weight * surface_rating + (1 - weight) * overall_rating


def calc_absence_penalty(rating, last_match, current_date):
    if last_match is None:
        return 0.0
    if rating <= ABSENCE_MIN_RATING:
        return 0.0
    weeks_absent = (current_date - last_match).days / 7.0
    if weeks_absent <= ABSENCE_THRESHOLD_WEEKS:
        return 0.0
    weeks_over = weeks_absent - ABSENCE_THRESHOLD_WEEKS
    max_penalty = rating - ABSENCE_MIN_RATING  # don't go below min rating
    return min(weeks_over * ABSENCE_PENALTY_PER_WEEK, max_penalty)


def _update_elo(ratings, matches, w, l, match_date, is_walkover,
                lookback_days, k_multiplier=1.0):
    rw, rl = ratings[w], ratings[l]
    pwin = elo_expected(rw, rl)

    if not is_walkover:
        k_w = dynamic_k_factor(matches[w], match_date, lookback_days) * k_multiplier
        k_l = dynamic_k_factor(matches[l], match_date, lookback_days) * k_multiplier
        ratings[w] = rw + k_w * (1.0 - pwin)
        ratings[l] = rl - k_l * (1.0 - pwin)

    matches[w].append(match_date)
    matches[l].append(match_date)

    return rw, rl, pwin


def build_elo_features(
    df: pd.DataFrame,
    *,
    winner_col: str = "winner_id",
    loser_col: str = "loser_id",
    winner_rank_col: str = "winner_rank",
    loser_rank_col: str = "loser_rank",
    date_col: str = "Date",
    surface_col: str = "surface",
    tourney_level_col: str = "tournament_level",
    comment_col: str = "Comment",
) -> pd.DataFrame:
    out = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(out)

    # Global ratings (initialized on first appearance based on rank)
    ratings = {}
    matches = defaultdict(list)
    last_played = {}

    # Surface-specific ratings: surface -> player -> rating
    surf_ratings = defaultdict(dict)
    surf_matches = defaultdict(lambda: defaultdict(list))

    # Track surface match counts per player
    surf_match_count = defaultdict(lambda: defaultdict(int))

    # Output arrays
    cols = {k: np.empty(n, dtype=np.float64) for k in
            ["winner_elo", "loser_elo", "elo_pwin",
             "winner_surface_elo", "loser_surface_elo", "surface_elo_pwin",
             "winner_blended_elo", "loser_blended_elo", "blended_elo_pwin"]}

    all_surfaces = ["hard", "clay", "grass"]

    for i, row in out.iterrows():
        w, l = row[winner_col], row[loser_col]
        w_rank, l_rank = row[winner_rank_col], row[loser_rank_col]
        date = row[date_col]
        surface = str(row[surface_col]).lower()
        tourney_level = str(row[tourney_level_col]).lower()
        is_wo = str(row.get(comment_col, "Completed")).lower() == "walkover"
        k_mult = TOURNAMENT_K_MULTIPLIER.get(tourney_level, 1.0)

        # Initialize ratings on first appearance
        if w not in ratings:
            ratings[w] = initial_elo_from_rank(w_rank)
            for s in all_surfaces:
                surf_ratings[s][w] = initial_elo_from_rank(w_rank)
        if l not in ratings:
            ratings[l] = initial_elo_from_rank(l_rank)
            for s in all_surfaces:
                surf_ratings[s][l] = initial_elo_from_rank(l_rank)

        # Apply global absence penalty to general AND all surface ratings
        for player in [w, l]:
            penalty = calc_absence_penalty(ratings[player], last_played.get(player), date)
            if penalty > 0:
                ratings[player] -= penalty
                for s in all_surfaces:
                    if player in surf_ratings[s]:
                        surf_ratings[s][player] = max(
                            ABSENCE_MIN_RATING,
                            surf_ratings[s][player] - penalty
                        )

        # Update last played
        last_played[w] = date
        last_played[l] = date

        # Blended Elo (compute before updates, using current ratings)
        w_blended = blended_rating(ratings[w], surf_ratings[surface][w], surf_match_count[surface][w])
        l_blended = blended_rating(ratings[l], surf_ratings[surface][l], surf_match_count[surface][l])
        cols["winner_blended_elo"][i] = w_blended
        cols["loser_blended_elo"][i] = l_blended
        cols["blended_elo_pwin"][i] = elo_expected(w_blended, l_blended)

        # Global Elo
        cols["winner_elo"][i], cols["loser_elo"][i], cols["elo_pwin"][i] = \
            _update_elo(ratings, matches, w, l, date, is_wo, 365, k_mult)

        # Surface Elo
        cols["winner_surface_elo"][i], cols["loser_surface_elo"][i], cols["surface_elo_pwin"][i] = \
            _update_elo(surf_ratings[surface], surf_matches[surface], w, l, date, is_wo, 800, k_mult)

        # Update surface match counts
        surf_match_count[surface][w] += 1
        surf_match_count[surface][l] += 1

    for col, arr in cols.items():
        out[col] = arr
    out["elo_diff"] = out["winner_elo"] - out["loser_elo"]
    out["surface_elo_diff"] = out["winner_surface_elo"] - out["loser_surface_elo"]
    out["blended_elo_diff"] = out["winner_blended_elo"] - out["loser_blended_elo"]

    return out
