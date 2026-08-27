"""
Phase 3: the opportunity layer.

THE CORE IDEA
-------------
Fantasy production = opportunity x efficiency.

Phases 1-2 handle efficiency: what a player does per touch, and how that
changes with age. Those are player-owned properties, estimated from history.

This phase handles opportunity: how many touches he gets. That is team-owned,
resets every offseason, and CANNOT be derived from historical data. It has to
be entered by a human who has read the current news.

So this module is deliberately NOT clever. It reads a hand-maintained CSV.
The judgment lives in the CSV; the arithmetic lives here.

WHY A CSV AND NOT CODE
----------------------
Depth charts move daily until kickoff. A CSV can be edited in thirty seconds
when a starter goes down in the third preseason game. Code cannot. The file
is the part of the model you own and update; everything else is machinery.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HISTORY = os.path.join(DATA_DIR, "processed", "player_seasons.parquet")
OPP_DIR = os.path.join(DATA_DIR, "opportunity")

# ---------------------------------------------------------------------------
# Role tiers.
#
# These exist to make the CSV fast to fill in. Assign a tier from the news,
# and it implies a default share of the backfield/target pie. Override the
# numeric columns directly whenever you know better than the default.
# ---------------------------------------------------------------------------
ROLE_TIERS = {
    # RB tiers — share of team RB touches
    # Medians from 2025 actual data (players ≥10 games, classified by role):
    #   bell_cow 0.65, lead_committee 0.54, even_committee 0.41, rank36-48 0.30
    "bell_cow":        {"touch_share": 0.65, "job_security": 0.85},
    "lead_committee":  {"touch_share": 0.54, "job_security": 0.65},
    "even_committee":  {"touch_share": 0.41, "job_security": 0.50},
    "passing_down":    {"touch_share": 0.25, "job_security": 0.60},
    "early_down":      {"touch_share": 0.30, "job_security": 0.55},
    "handcuff":        {"touch_share": 0.18, "job_security": 0.35},
    "backup":          {"touch_share": 0.10, "job_security": 0.25},

    # Pass-catcher tiers — share of team targets
    # Medians from 2025 actual data (TEAM_TARGETS_PER_GAME = 30.3):
    #   alpha_wr 0.248, wr2 0.181, wr3 0.124, te1 0.196, te2 0.141
    "alpha_wr":        {"target_share": 0.25, "job_security": 0.90},
    "wr2":             {"target_share": 0.18, "job_security": 0.70},
    "wr3":             {"target_share": 0.12, "job_security": 0.50},
    "slot":            {"target_share": 0.15, "job_security": 0.60},
    "featured_te":     {"target_share": 0.30, "job_security": 0.85},  # McBride/Bowers tier: 9+ tgt/g
    "te1":             {"target_share": 0.20, "job_security": 0.75},  # solid starter: ~6 tgt/g
    "te2":             {"target_share": 0.14, "job_security": 0.35},  # #2 TE: ~4 tgt/g

    # QB tiers
    "starting_qb":     {"job_security": 0.85},
    "contested_qb":    {"job_security": 0.50},
    "backup_qb":       {"job_security": 0.15},
}

# Columns the opportunity file must contain.
REQUIRED_COLUMNS = [
    "player_name",      # display name, joined against history
    "team",             # 2026 team
    "position",
    "role_tier",        # one of ROLE_TIERS
    "touch_share",      # 0-1, share of team RB touches. Blank = tier default
    "target_share",     # 0-1, share of team targets. Blank = tier default
    "job_security",     # 0-1 confidence he holds the role. Blank = tier default
    "team_changed",     # 1 if new team for 2026
    "new_play_caller",  # 1 if the team's play-caller changed
    "situation_note",   # free text, for you
    "confidence",       # high / medium / low — how sure is the whole row
    "last_updated",
]

# League-average team volume, used to turn shares into counts.
# Refined from actual team data in a later pass; these are reasonable priors.
TEAM_RB_TOUCHES_PER_GAME = 27.0
TEAM_TARGETS_PER_GAME = 30.0  # 2025 actual: 30.3 (was 34.0)


def opportunity_path(position: str) -> str:
    return os.path.join(OPP_DIR, f"{position.lower()}_2026.csv")


def make_skeleton(position: str, min_games: int = 4,
                  seasons_back: int = 2) -> pd.DataFrame:
    """
    Generate a starter CSV listing players who might matter in 2026.

    Pulled from recent history so you aren't typing 200 names from scratch.
    Rookies are NOT here — they have no history — and must be added by hand.
    """
    hist = pd.read_parquet(HISTORY)
    recent = hist[
        (hist.season >= hist.season.max() - seasons_back + 1)
        & (hist.position == position.upper())
        & (hist.games >= min_games)
    ]

    latest = (recent.sort_values("season")
              .groupby("player_display_name", as_index=False)
              .last())

    out = pd.DataFrame({
        "player_name": latest["player_display_name"],
        "team": latest["team"],          # 2025 team — VERIFY, many changed
        "position": position.upper(),
        "role_tier": "",
        "touch_share": np.nan,
        "target_share": np.nan,
        "job_security": np.nan,
        "team_changed": 0,
        "new_play_caller": 0,
        "situation_note": "",
        "confidence": "",
        "last_updated": "",
    })

    # Sort by recent production so the players who matter are at the top.
    order = latest.set_index("player_display_name")["fantasy_points"]
    out["_sort"] = out["player_name"].map(order)
    out = out.sort_values("_sort", ascending=False).drop(columns="_sort")
    return out.reset_index(drop=True)


def load_opportunity(position: str) -> pd.DataFrame:
    """Read the hand-maintained file and fill defaults from role tiers."""
    path = opportunity_path(position)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No opportunity file at {path}. "
            f"Run: python src/opportunity.py --skeleton {position}"
        )
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    for field in ("touch_share", "target_share", "job_security"):
        defaults = df["role_tier"].map(
            lambda t: ROLE_TIERS.get(t, {}).get(field, np.nan)
        )
        df[field] = df[field].fillna(defaults)

    return df


def project_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn shares into per-game counts.

    Deliberately simple. The hard part of this phase is deciding the shares,
    not the arithmetic that follows from them.
    """
    out = df.copy()
    out["proj_carries_pg"] = out["touch_share"].fillna(0) * TEAM_RB_TOUCHES_PER_GAME
    out["proj_targets_pg"] = out["target_share"].fillna(0) * TEAM_TARGETS_PER_GAME

    # Risk-adjusted volume. A player with a 50% chance of holding a big role
    # is not worth the same as one certain to hold a smaller role, even if the
    # expected touches match — the auction punishes variance more than it
    # rewards it, because a bust costs you the roster spot too.
    out["risk_adj_carries_pg"] = out["proj_carries_pg"] * out["job_security"].fillna(0.5)
    out["risk_adj_targets_pg"] = out["proj_targets_pg"] * out["job_security"].fillna(0.5)
    return out


def audit(position: str) -> None:
    """Report what still needs filling in. Run this often."""
    df = pd.read_csv(opportunity_path(position))
    total = len(df)
    filled = df["role_tier"].notna() & (df["role_tier"].astype(str) != "")
    print(f"{position.upper()}: {filled.sum()}/{total} rows have a role_tier")

    bad = df.loc[filled, "role_tier"][
        ~df.loc[filled, "role_tier"].isin(ROLE_TIERS)
    ]
    if len(bad):
        print(f"  Unrecognized tiers: {sorted(bad.unique())}")

    unfilled = df.loc[~filled, "player_name"].head(15).tolist()
    if unfilled:
        print(f"  Next to fill: {', '.join(unfilled)}")


if __name__ == "__main__":
    import sys

    os.makedirs(OPP_DIR, exist_ok=True)

    if "--skeleton" in sys.argv:
        pos = sys.argv[sys.argv.index("--skeleton") + 1]
        path = opportunity_path(pos)
        if os.path.exists(path):
            print(f"{path} already exists — refusing to overwrite your edits.")
        else:
            make_skeleton(pos).to_csv(path, index=False)
            print(f"Wrote skeleton: {path}")
    elif "--audit" in sys.argv:
        audit(sys.argv[sys.argv.index("--audit") + 1])
    else:
        print(__doc__)
        print("Usage:")
        print("  python src/opportunity.py --skeleton RB")
        print("  python src/opportunity.py --audit RB")
