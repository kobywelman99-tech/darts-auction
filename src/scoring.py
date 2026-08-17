"""
Scoring engine.

Converts raw nflverse stat lines into DARTS fantasy points.

Why compute this ourselves instead of using nflverse's built-in
`fantasy_points_ppr` column: that column assumes full PPR, 4-point passing
TDs, -2 per interception and standard yardage. Our league is HALF PPR with
-2 interceptions. Using the prebuilt column would silently misprice every
pass-catcher in the pool.
"""

from __future__ import annotations

import pandas as pd

from league_config import SCORING


# Columns pulled from nflverse weekly stats that feed offensive scoring.
STAT_COLUMNS = [
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_2pt_conversions",
    "rushing_yards",
    "rushing_tds",
    "rushing_2pt_conversions",
    "rushing_fumbles_lost",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_2pt_conversions",
    "receiving_fumbles_lost",
    "sack_fumbles_lost",
    "special_teams_tds",
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column as floats, or zeros if the column is absent."""
    if name in df.columns:
        return df[name].fillna(0).astype(float)
    return pd.Series(0.0, index=df.index)


def score_offense(df: pd.DataFrame) -> pd.Series:
    """
    Compute DARTS fantasy points for offensive players.

    Accepts either weekly or season-aggregated rows — the math is identical
    because every scoring rule is linear.
    """
    s = SCORING

    passing = (
        _col(df, "passing_yards") / s["passing_yards_per_point"]
        + _col(df, "passing_tds") * s["passing_td"]
        + _col(df, "passing_interceptions") * s["interception"]
    )

    rushing = (
        _col(df, "rushing_yards") / s["rushing_yards_per_point"]
        + _col(df, "rushing_tds") * s["rushing_td"]
    )

    receiving = (
        _col(df, "receptions") * s["reception"]
        + _col(df, "receiving_yards") / s["receiving_yards_per_point"]
        + _col(df, "receiving_tds") * s["receiving_td"]
    )

    returns = _col(df, "special_teams_tds") * s["return_td"]

    two_point = (
        _col(df, "passing_2pt_conversions")
        + _col(df, "rushing_2pt_conversions")
        + _col(df, "receiving_2pt_conversions")
    ) * s["two_point_conversion"]

    # Only fumbles actually LOST cost points. Fumbles recovered by own team
    # are free, so we deliberately do not use `fumbles_total`.
    fumbles = (
        _col(df, "rushing_fumbles_lost")
        + _col(df, "receiving_fumbles_lost")
        + _col(df, "sack_fumbles_lost")
    ) * s["fumble_lost"]

    return passing + rushing + receiving + returns + two_point + fumbles


def score_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break scoring into its parts.

    Useful for diagnosis: it lets us see how much of a player's total came
    from touchdowns (noisy, regresses hard) versus yardage and receptions
    (stable, driven by opportunity).
    """
    s = SCORING

    out = pd.DataFrame(index=df.index)
    out["pts_pass_yards"] = _col(df, "passing_yards") / s["passing_yards_per_point"]
    out["pts_pass_td"] = _col(df, "passing_tds") * s["passing_td"]
    out["pts_int"] = _col(df, "passing_interceptions") * s["interception"]
    out["pts_rush_yards"] = _col(df, "rushing_yards") / s["rushing_yards_per_point"]
    out["pts_rush_td"] = _col(df, "rushing_tds") * s["rushing_td"]
    out["pts_receptions"] = _col(df, "receptions") * s["reception"]
    out["pts_rec_yards"] = _col(df, "receiving_yards") / s["receiving_yards_per_point"]
    out["pts_rec_td"] = _col(df, "receiving_tds") * s["receiving_td"]
    out["pts_return_td"] = _col(df, "special_teams_tds") * s["return_td"]
    out["pts_two_point"] = (
        _col(df, "passing_2pt_conversions")
        + _col(df, "rushing_2pt_conversions")
        + _col(df, "receiving_2pt_conversions")
    ) * s["two_point_conversion"]
    out["pts_fumbles"] = (
        _col(df, "rushing_fumbles_lost")
        + _col(df, "receiving_fumbles_lost")
        + _col(df, "sack_fumbles_lost")
    ) * s["fumble_lost"]

    out["fantasy_points"] = out.sum(axis=1)

    # All touchdown points, isolated. This is the single most over-extrapolated
    # quantity in fantasy football and gets regressed in the projection phase.
    out["pts_from_td"] = (
        out["pts_pass_td"] + out["pts_rush_td"]
        + out["pts_rec_td"] + out["pts_return_td"]
    )
    return out
