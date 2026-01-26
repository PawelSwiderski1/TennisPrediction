import numpy as np
import pandas as pd
from collections import defaultdict

# Tournament level hierarchy (higher value = higher prestige)
LEVEL_HIERARCHY = {
    "grand slam": 3,
    "masters 1000": 2,
    "atp500": 1,
    "atp250": 1,
}
DEFAULT_LEVEL = 0


def get_level_value(level: str) -> int:
    """Convert tournament level to numeric value."""
    return LEVEL_HIERARCHY.get(str(level).lower(), DEFAULT_LEVEL)


def build_round_level_features(
    df: pd.DataFrame,
    *,
    winner_col: str = "winner_id",
    loser_col: str = "loser_id",
    date_col: str = "Date",
    round_col: str = "round",
    level_col: str = "tournament_level",
) -> pd.DataFrame:
    """
    Add round-level performance features for winners and losers.

    Features added:
        - winner_round_level_appearances: appearances in this round at this level or higher
        - loser_round_level_appearances: appearances in this round at this level or higher
        - winner_round_level_win_pct: win % in this round at this level or higher
        - loser_round_level_win_pct: win % in this round at this level or higher

    Optimized by maintaining running counts per player/round/level combination.
    """
    out = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(out)

    # Running counts per player -> round -> level -> {appearances, wins}
    # This allows O(1) lookup and update per match
    player_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"apps": 0, "wins": 0})))

    # Cache for (player, round, min_level) -> (appearances, wins) to avoid recalculating sums
    # Key: (player, round, level_value), Value: (total_apps, total_wins)
    cache = {}

    # Output arrays
    winner_apps = np.zeros(n, dtype=np.int32)
    loser_apps = np.zeros(n, dtype=np.int32)
    winner_win_pct = np.zeros(n, dtype=np.float64)
    loser_win_pct = np.zeros(n, dtype=np.float64)

    # All levels for summing
    all_levels = list(LEVEL_HIERARCHY.keys())

    def get_stats_at_or_above(player, round_name, min_level_val):
        """Get total appearances and wins at round for levels >= min_level_val."""
        cache_key = (player, round_name, min_level_val)
        if cache_key in cache:
            return cache[cache_key]

        total_apps = 0
        total_wins = 0
        for lvl in all_levels:
            if get_level_value(lvl) >= min_level_val:
                stats = player_stats[player][round_name][lvl]
                total_apps += stats["apps"]
                total_wins += stats["wins"]

        cache[cache_key] = (total_apps, total_wins)
        return total_apps, total_wins

    def invalidate_cache(player, round_name):
        """Invalidate cache entries for player/round when stats update."""
        for min_lvl in LEVEL_HIERARCHY.values():
            cache_key = (player, round_name, min_lvl)
            if cache_key in cache:
                del cache[cache_key]

    for i, row in out.iterrows():
        w = row[winner_col]
        l = row[loser_col]
        round_name = str(row[round_col]).lower()
        level = str(row[level_col]).lower()
        level_val = get_level_value(level)

        # Get stats BEFORE this match
        w_apps, w_wins = get_stats_at_or_above(w, round_name, level_val)
        l_apps, l_wins = get_stats_at_or_above(l, round_name, level_val)

        winner_apps[i] = w_apps
        loser_apps[i] = l_apps
        winner_win_pct[i] = w_wins / w_apps if w_apps > 0 else 0.0
        loser_win_pct[i] = l_wins / l_apps if l_apps > 0 else 0.0

        # Update stats AFTER this match
        # Invalidate cache first
        invalidate_cache(w, round_name)
        invalidate_cache(l, round_name)

        # Winner: appeared and won
        player_stats[w][round_name][level]["apps"] += 1
        player_stats[w][round_name][level]["wins"] += 1

        # Loser: appeared but lost
        player_stats[l][round_name][level]["apps"] += 1

    out["winner_round_level_appearances"] = winner_apps
    out["loser_round_level_appearances"] = loser_apps
    out["winner_round_level_win_pct"] = winner_win_pct
    out["loser_round_level_win_pct"] = loser_win_pct

    return out
