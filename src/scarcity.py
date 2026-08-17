"""
Positional scarcity — empirical replacement level.

Replacement level is the per-game output of the last startable player at a
position. It is NOT a fixed rank; it falls out of the league's roster
requirements. Getting this right is the whole ballgame in a 2QB league.

Run:  python src/scarcity.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from league_config import (
    NUM_TEAMS, STARTERS, FLEX_ELIGIBLE, BENCH_SPOTS, DISCRETIONARY_MONEY,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                    "player_seasons.parquet")

# How the flex slot splits across eligible positions in practice. Flex is
# overwhelmingly filled by RB and WR; TE flex starts are uncommon.
FLEX_SPLIT = {"RB": 0.40, "WR": 0.55, "TE": 0.05}

# Bench spots are not evenly distributed either — managers hoard RBs and WRs.
BENCH_SPLIT = {"QB": 0.15, "RB": 0.35, "WR": 0.38, "TE": 0.12}


def replacement_ranks(include_bench: bool = False) -> dict[str, int]:
    """
    The rank at each position where replacement level sits.

    With `include_bench=False` this is the strict starter requirement, which
    is the right baseline for valuing starters. Including bench demand pushes
    replacement deeper and is closer to how a real draft empties the pool.
    """
    ranks: dict[str, float] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        starters = STARTERS.get(pos, 0) * NUM_TEAMS
        flex = 0.0
        if pos in FLEX_ELIGIBLE:
            flex = STARTERS["FLEX"] * NUM_TEAMS * FLEX_SPLIT[pos]
        bench = BENCH_SPLIT[pos] * BENCH_SPOTS * NUM_TEAMS if include_bench else 0.0
        ranks[pos] = starters + flex + bench
    return {k: int(round(v)) for k, v in ranks.items()}


def replacement_level(df: pd.DataFrame, seasons: list[int],
                      ranks: dict[str, int], min_games: int = 6) -> pd.Series:
    """Average PPG at the replacement rank, across several seasons."""
    out = {}
    for pos, rank in ranks.items():
        vals = []
        for yr in seasons:
            pool = (df[(df.season == yr) & (df.position == pos) &
                       (df.games >= min_games)]
                    .nlargest(rank + 5, "fantasy_points")["ppg"].values)
            if len(pool) >= rank:
                vals.append(pool[rank - 1])
        out[pos] = float(np.mean(vals)) if vals else np.nan
    return pd.Series(out)


def points_above_replacement(df: pd.DataFrame, season: int,
                             baseline: pd.Series) -> pd.DataFrame:
    """
    Value above replacement, in points.

    Multiplied by games played rather than a fixed 17, so a player who was
    elite but missed time is credited for what he actually delivered.
    """
    d = df[(df.season == season) & (df.games >= 6)].copy()
    d["replacement_ppg"] = d["position"].map(baseline)
    d["par"] = (d["ppg"] - d["replacement_ppg"]) * d["games"]
    return d.sort_values("par", ascending=False)


def main() -> None:
    df = pd.read_parquet(DATA)
    seasons = [2022, 2023, 2024, 2025]

    starter_ranks = replacement_ranks(include_bench=False)
    deep_ranks = replacement_ranks(include_bench=True)

    print("Replacement rank by position (this league's roster rules):")
    for pos in ("QB", "RB", "WR", "TE"):
        print(f"  {pos}: starters-only {starter_ranks[pos]:>3}   "
              f"with bench demand {deep_ranks[pos]:>3}")

    baseline = replacement_level(df, seasons, deep_ranks)
    print(f"\nReplacement level PPG ({seasons[0]}-{seasons[-1]}):")
    for pos, val in baseline.items():
        print(f"  {pos}: {val:.1f}")

    par = points_above_replacement(df, 2025, baseline)
    print("\n2025 top 15 by points above replacement:")
    cols = ["player_display_name", "position", "team", "games", "ppg", "par"]
    print(par.head(15)[cols].to_string(index=False, float_format="%.1f"))

    # Preview of Phase 4: dollars follow from PAR share of the pool.
    pool = par[par["par"] > 0].head(150)
    total_par = pool["par"].sum()
    print(f"\nIf 2025 had been perfectly foreseen, $/PAR "
          f"= {DISCRETIONARY_MONEY / total_par:.2f}")
    print("Implied prices (Phase 4 does this properly with projections):")
    preview = pool.head(8).copy()
    preview["price"] = (preview["par"] / total_par * DISCRETIONARY_MONEY + 1).round(0)
    print(preview[["player_display_name", "position", "par", "price"]]
          .to_string(index=False, float_format="%.0f"))


if __name__ == "__main__":
    main()
