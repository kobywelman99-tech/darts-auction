"""
Phase 1: build the player-season history table.

Output: one row per player-season, containing DARTS fantasy points, the
scoring breakdown, usage rates, age, and experience. This is the foundation
every later layer reads from.

Run:  python src/build_history.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from ingest import load_weekly_range, load_players
from scoring import score_offense, score_components
from league_config import HISTORY_START_SEASON, HISTORY_END_SEASON, REGULAR_SEASON_WEEKS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

# Counting stats summed to the season level.
SUM_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions", "passing_air_yards",
    "passing_first_downs", "passing_2pt_conversions",
    "rushing_yards", "rushing_tds", "rushing_first_downs", "carries",
    "rushing_2pt_conversions", "rushing_fumbles_lost",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_first_downs", "receiving_yards_after_catch",
    "receiving_2pt_conversions", "receiving_fumbles_lost",
    "sack_fumbles_lost", "special_teams_tds",
]

# Rate stats averaged across games played.
MEAN_COLS = ["target_share", "air_yards_share", "wopr"]


def age_at_season(birth_date: pd.Series, season: pd.Series) -> pd.Series:
    """
    Age on September 1 of the given season.

    Anchoring to a fixed date matters: comparing a player's 'age 27 season'
    to another's is meaningless if one turned 27 in January and the other
    in December.
    """
    birth = pd.to_datetime(birth_date, errors="coerce")
    ref = pd.to_datetime(season.astype(str) + "-09-01")
    return (ref - birth).dt.days / 365.25


def build() -> pd.DataFrame:
    print("Pulling weekly stats from nflverse...")
    weekly = load_weekly_range(HISTORY_START_SEASON, HISTORY_END_SEASON)

    print("Pulling player master file...")
    players = load_players()

    # Regular season only. Playoff games would inflate totals for players on
    # good teams, and our league's season ends in Week 17 anyway.
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    weekly = weekly[weekly["position"].isin(FANTASY_POSITIONS)].copy()

    weekly["fantasy_points"] = score_offense(weekly)

    # A "game played" = a week with a recorded stat line. See README caveat.
    weekly["games"] = 1

    sum_cols = [c for c in SUM_COLS if c in weekly.columns]
    mean_cols = [c for c in MEAN_COLS if c in weekly.columns]

    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})
    agg["fantasy_points"] = "sum"
    agg["games"] = "sum"

    season_df = (
        weekly.groupby(["player_id", "season"], as_index=False)
        .agg({**agg, "player_display_name": "first", "position": "first"})
    )

    # Team as of the player's last game that season (handles midseason trades).
    last_team = (
        weekly.sort_values("week")
        .groupby(["player_id", "season"], as_index=False)["team"]
        .last()
    )
    season_df = season_df.merge(last_team, on=["player_id", "season"], how="left")

    # Scoring breakdown, so we can separate TD luck from real production.
    comp = score_components(season_df)
    season_df["pts_from_td"] = comp["pts_from_td"]
    season_df["pts_non_td"] = season_df["fantasy_points"] - season_df["pts_from_td"]

    # Biographical data
    bio = players[["gsis_id", "birth_date", "rookie_season",
                   "draft_year", "draft_round", "draft_pick"]].rename(
        columns={"gsis_id": "player_id"}
    )
    season_df = season_df.merge(bio, on="player_id", how="left")

    season_df["age"] = age_at_season(season_df["birth_date"], season_df["season"])
    season_df["experience"] = season_df["season"] - season_df["rookie_season"]

    # Per-game rates — the honest way to compare a 17-game season to an
    # 11-game one. Season totals punish injured players twice.
    season_df["ppg"] = season_df["fantasy_points"] / season_df["games"]
    season_df["games_missed"] = REGULAR_SEASON_WEEKS + 3 - season_df["games"]

    for col, per_game in [("targets", "targets_pg"), ("carries", "carries_pg"),
                          ("receptions", "receptions_pg")]:
        if col in season_df.columns:
            season_df[per_game] = season_df[col] / season_df["games"]

    # Efficiency: player-owned, stable year to year.
    with np.errstate(divide="ignore", invalid="ignore"):
        season_df["yards_per_target"] = (
            season_df["receiving_yards"] / season_df["targets"].replace(0, np.nan)
        )
        season_df["yards_per_carry"] = (
            season_df["rushing_yards"] / season_df["carries"].replace(0, np.nan)
        )
        season_df["catch_rate"] = (
            season_df["receptions"] / season_df["targets"].replace(0, np.nan)
        )
        # TD rate per opportunity — the noisiest thing in fantasy, and the
        # thing that most often makes a player look like a breakout.
        opportunities = (
            season_df["targets"].fillna(0) + season_df["carries"].fillna(0)
        ).replace(0, np.nan)
        season_df["td_rate"] = (
            (season_df["rushing_tds"].fillna(0) + season_df["receiving_tds"].fillna(0))
            / opportunities
        )

    season_df = season_df.sort_values(["season", "fantasy_points"],
                                      ascending=[True, False])
    return season_df


def main() -> None:
    df = build()
    os.makedirs(OUT_DIR, exist_ok=True)

    parquet_path = os.path.join(OUT_DIR, "player_seasons.parquet")
    csv_path = os.path.join(OUT_DIR, "player_seasons.csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"\nBuilt {len(df):,} player-seasons "
          f"({HISTORY_START_SEASON}-{HISTORY_END_SEASON})")
    print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
