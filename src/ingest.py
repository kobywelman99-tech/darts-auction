"""
Data ingestion from nflverse.

Pulls weekly player stats and the player master file, caching everything to
disk so we only hit the network once per season file.
"""

from __future__ import annotations

import os
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
WEEKLY_URL = BASE + "/stats_player/stats_player_week_{season}.parquet"
PLAYERS_URL = BASE + "/players/players.parquet"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def load_weekly(season: int, refresh: bool = False) -> pd.DataFrame:
    """Weekly player stat lines for one season."""
    path = _cache_path(f"weekly_{season}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    df = pd.read_parquet(WEEKLY_URL.format(season=season))
    df.to_parquet(path, index=False)
    return df


def load_weekly_range(start: int, end: int, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for season in range(start, end + 1):
        df = load_weekly(season, refresh=refresh)
        frames.append(df)
        print(f"  {season}: {len(df):,} player-weeks")
    return pd.concat(frames, ignore_index=True)


def load_players(refresh: bool = False) -> pd.DataFrame:
    """
    Player master file: birth date, rookie season, draft capital.

    Draft capital matters because it is a durable signal of how much
    opportunity a team will keep handing a player through early struggles.
    """
    path = _cache_path("players.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    df = pd.read_parquet(PLAYERS_URL)
    df.to_parquet(path, index=False)
    return df
