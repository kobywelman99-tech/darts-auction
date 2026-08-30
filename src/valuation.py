"""
Phase 4 — Auction dollar values.

PAR-TO-DOLLARS: WHY $1,850
--------------------------
10 teams × $200 = $2,000 in the pool.
150 players get drafted × $1 minimum bid = $150 locked in before bidding starts.
That leaves $1,850 in discretionary money — the only portion that responds to
player value.

Each player's price:  $1  +  (player_PAR / total_PAR)  ×  $1,850
Sum across 150:       150 × $1  +  $1,850  =  $2,000  ✓

SELECTION: POSITION QUOTAS, NOT GLOBAL PAR RANK
------------------------------------------------
We pick the top N players *within each position* using bench-inclusive ranks
from scarcity.py (QB 28, RB 42, WR 54, TE 16, DEF 10 = 150).  Selecting
globally by PAR lets one position crowd out another when replacement-level
gaps are unequal — in a 2QB league, every QB above 13.1 ppg looks valuable
by PAR, so a global sort drafts 44 QBs and only 33 WRs.  The quota enforces
what roster construction actually demands.

DEFENSES
--------
10 DEF slots are reserved at $1 each.  DST scoring isn't modeled yet, so
no PAR-based price is possible.  The 140 offensive players price against the
remainder: $2,000 − (10 × $1 DEF) − (140 × $1 minimums) = $1,850.

PROJECTION SOURCE (SWAPPABLE)
------------------------------
compute_values() accepts any projection function.  That function receives the
full history DataFrame and must return one row per player with at minimum:
  player_display_name, position, team, age, proj_ppg

The default is placeholder_projections(), which uses 2025 actual ppg.
Swap it for Phase 3 output — the pricing code stays untouched.

Run:  python src/valuation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from league_config import (
    BUDGET_PER_TEAM,
    DISCRETIONARY_MONEY,
    MIN_BID,
    NUM_TEAMS,
    TOTAL_PLAYERS_DRAFTED,
)
from scarcity import replacement_level, replacement_ranks

HISTORY = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "player_seasons.parquet"
)
RECENT_SEASONS = [2022, 2023, 2024, 2025]  # window for computing replacement level
PROJ_GAMES = 17                             # full-season assumption; injury risk = Phase 3
DEF_SLOTS = 10                              # reserved at $1 each; not priced via PAR
STD_1QB_DRAFTED = 15                        # approx QBs drafted in a standard 10-team 1QB league

# Seasons used to estimate per-touch efficiency (separate from replacement-level window)
EFFICIENCY_SEASONS = [2023, 2024, 2025]
EFFICIENCY_MIN_GAMES = 6   # lower bar: want more data for per-touch rates

# ── Efficiency pipeline settings ────────────────────────────────────────────
# Recency: year weights multiplied by games played within each season.
# 2025 counts 3×, 2024 2×, 2023 1×.  Games-weighting within a year means
# a 6-game injury season still outweighs a 16-game healthy season two years back.
YEAR_WEIGHTS = {2023: 1.0, 2024: 2.0, 2025: 3.0}

# TD regression: Bayesian shrinkage toward the positional median.
# k is the equivalent "prior sample size": large k → heavy regression toward mean.
# k=50 targets ≈ a fringe starter's seasonal volume; k=100 carries ≈ same for RBs.
# QB passing: k=300 ≈ one season of backup-starter attempts.
TD_REG_K_TARGETS  = 50    # receiving TD regression prior weight (targets)
TD_REG_K_CARRIES  = 100   # rushing TD regression prior weight (carries)
TD_REG_K_PASS_ATT = 300   # QB passing TD regression prior weight (attempts)

# Age curve: applied AFTER recency-weighted efficiency.
# Recency weighting answers "what is his efficiency NOW."
# The age curve projects that estimate one year forward to 2026.
# Scaling at 50% avoids amplifying both signals for a young ascending player.
AGE_CURVE_SCALE = 0.50


# ---------------------------------------------------------------------------
# PROJECTION SOURCES
# ---------------------------------------------------------------------------

def placeholder_projections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder: each player's 2025 season ppg as their 2026 projection.

    Filtering to 2025 (rather than each player's most recent season) drops
    retired players automatically — they have no 2025 entry.

    CONTRACT (any replacement function must satisfy this):
      - Input:  full history DataFrame (player_seasons.parquet)
      - Output: one row per player, no duplicate player_display_name within position
      - Required columns: player_display_name, position, team, age, proj_ppg
    """
    recent = df[(df.season == 2025) & (df.games >= 6)].copy()
    return (
        recent
        .rename(columns={"ppg": "proj_ppg"})
        [["player_display_name", "position", "team", "age", "proj_ppg"]]
        .reset_index(drop=True)
    )


def opportunity_projections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 3 projection: opportunity CSVs → projected ppg.

    Model: proj_ppg = opportunity × efficiency

    job_security does NOT enter the projection formula — it belongs in
    draft_state.py as a ceiling discount on the bid price. Applying it here
    would double-count risk because efficiency rates are computed from games
    the player actually played (survival already baked in).

    OPPORTUNITY (from data/opportunity/{pos}_2026.csv)
      role_tier  →  touch_share (RB) or target_share (WR/TE)
      shares × league-average team volume = raw projected touches/targets per game

    EFFICIENCY (from 2023-2025 history)
      pts_per_carry  = season rushing pts / carries
      pts_per_target = season receiving pts / targets
      Averaged across qualifying seasons (≥6 games), weighted by games played.
      For players with no history (rookies), uses positional-median efficiency.

    RB RECEIVING
      The opportunity CSV only models rushing volume (touch_share).
      A back's receiving follows his role: his historical targets_pg is scaled
      by job_security so it shrinks with role uncertainty.

    FALLBACK
      Players not in any opportunity file (QBs, unrostered depth) fall back to
      placeholder_projections (2025 actual ppg).
      Players in the opportunity file but with no role_tier filled also fall back.
    """
    from opportunity import load_opportunity, project_volume

    # ── 1. Per-touch efficiency from recent history ─────────────────────────
    eff_df = df[
        df.season.isin(EFFICIENCY_SEASONS) & (df.games >= EFFICIENCY_MIN_GAMES)
    ].copy()

    # ── 1a. TD regression — split each season into non-TD and TD-rate parts ─
    # Non-TD components are stable (yards per target, catch rate, yards per carry).
    # TD rate is the noisiest signal in fantasy; regress it toward the positional
    # median using Bayesian shrinkage: λ = k/(k+n), so more targets → less regression.

    eff_df["_non_td_rec_pts"] = (
        eff_df["receiving_yards"] / 10
        + eff_df["receptions"] * 0.5
        - eff_df["receiving_fumbles_lost"] * 2
    )
    eff_df["_rec_td_rate"] = np.where(
        eff_df["targets"] > 10,
        eff_df["receiving_tds"] / eff_df["targets"],
        np.nan,
    )
    eff_df["_non_td_rush_pts"] = (
        eff_df["rushing_yards"] / 10
        - eff_df["rushing_fumbles_lost"] * 2
    )
    eff_df["_rush_td_rate"] = np.where(
        eff_df["carries"] > 10,
        eff_df["rushing_tds"] / eff_df["carries"],
        np.nan,
    )

    # Positional median TD rates (the Bayesian prior)
    pos_rec_td_prior  = eff_df.groupby("position")["_rec_td_rate"].median()
    pos_rush_td_prior = eff_df.groupby("position")["_rush_td_rate"].median()

    _rec_td_prior_col  = eff_df["position"].map(pos_rec_td_prior).fillna(0.0)
    _rush_td_prior_col = eff_df["position"].map(pos_rush_td_prior).fillna(0.0)

    lam_rec  = TD_REG_K_TARGETS / (TD_REG_K_TARGETS + eff_df["targets"].clip(lower=1))
    lam_rush = TD_REG_K_CARRIES / (TD_REG_K_CARRIES + eff_df["carries"].clip(lower=1))

    eff_df["_rec_td_rate_reg"] = np.where(
        eff_df["_rec_td_rate"].notna(),
        (1 - lam_rec)  * eff_df["_rec_td_rate"]  + lam_rec  * _rec_td_prior_col,
        np.nan,
    )
    eff_df["_rush_td_rate_reg"] = np.where(
        eff_df["_rush_td_rate"].notna(),
        (1 - lam_rush) * eff_df["_rush_td_rate"] + lam_rush * _rush_td_prior_col,
        np.nan,
    )

    # Regressed per-touch efficiency = non-TD component + regressed TD contribution
    eff_df["pts_per_carry"] = np.where(
        eff_df["carries"] > 10,
        eff_df["_non_td_rush_pts"] / eff_df["carries"]
        + eff_df["_rush_td_rate_reg"] * 6,
        np.nan,
    )
    eff_df["pts_per_target"] = np.where(
        eff_df["targets"] > 10,
        eff_df["_non_td_rec_pts"] / eff_df["targets"]
        + eff_df["_rec_td_rate_reg"] * 6,
        np.nan,
    )

    # ── 1b. Recency-weighted mean (year weight × games played) ───────────────
    # 2025 counts 3×, 2024 2×, 2023 1×.  A 6-game injury season from 2025 still
    # counts more than a 16-game 2023 season (6×3=18 > 16×1=16), which is by
    # design: recent form beats old form, even abbreviated.
    def _wavg(g):
        raw_w = g["games"] * g["season"].map(YEAR_WEIGHTS).fillna(1.0)
        def wmean(col):
            valid = g[col].notna()
            if not valid.any():
                return np.nan
            w = raw_w[valid]
            return float((g.loc[valid, col] * w).sum() / w.sum())
        return pd.Series({
            "pts_per_carry":   wmean("pts_per_carry"),
            "pts_per_target":  wmean("pts_per_target"),
            "hist_targets_pg": wmean("targets_pg"),
            "age":             wmean("age"),
        })

    eff = eff_df.groupby("player_display_name").apply(_wavg).reset_index()

    # Pull latest position/team for the merge
    latest_meta = (
        df[df.season == 2025]
        .sort_values("season")
        .groupby("player_display_name")[["position", "team", "age"]]
        .last()
        .reset_index()
    )
    eff = eff.merge(latest_meta, on="player_display_name", suffixes=("_hist", ""))

    # Positional median efficiency — fallback for rookies
    pos_median = (
        eff_df.groupby("position")
        .agg(med_ppc=("pts_per_carry", "median"), med_ppt=("pts_per_target", "median"))
        .to_dict("index")
    )

    # ── 2. Load opportunity files and compute volume ─────────────────────────
    opp_parts = []
    for pos in ("RB", "WR", "TE"):
        try:
            opp = load_opportunity(pos)
            opp_parts.append(project_volume(opp))
        except FileNotFoundError:
            pass

    if not opp_parts:
        return placeholder_projections(df)

    opp_all = pd.concat(opp_parts, ignore_index=True)

    # Only use rows where a role_tier was actually filled in
    opp_filled = opp_all[opp_all["role_tier"].notna() & (opp_all["role_tier"] != "")]

    # ── 3. Join efficiency to opportunity volume ─────────────────────────────
    # Merge only the efficiency scalars; keep position/team/age from opportunity CSV.
    eff_cols = eff[["player_display_name", "pts_per_carry", "pts_per_target",
                    "hist_targets_pg", "age"]]
    merged = opp_filled.merge(eff_cols, left_on="player_name",
                              right_on="player_display_name", how="left")

    # Canonical position from opportunity CSV
    merged_pos = merged["position"].str.upper()

    # Fill missing efficiency with positional medians (rookies / limited history)
    for col, med_key in [("pts_per_carry", "med_ppc"), ("pts_per_target", "med_ppt")]:
        missing = merged[col].isna()
        merged.loc[missing, col] = merged_pos[missing].map(
            lambda p: pos_median.get(p, {}).get(med_key, 0.0)
        )

    # ── 4. Compute proj_ppg ──────────────────────────────────────────────────
    # Use raw projected volume (no job_security discount here). job_security
    # belongs only in draft_state.py as a ceiling discount — applying it here
    # would double-count it because efficiency was measured from games actually
    # played (i.e., the player already survived to play those snaps).
    #
    # RBs: rushing volume from opportunity layer; receiving from historical rate.
    # WR/TE: receiving volume from opportunity layer; no rushing component.
    raw_carries  = merged["proj_carries_pg"].fillna(0.0)
    raw_targets  = merged["proj_targets_pg"].fillna(0.0)

    rb_mask = merged_pos == "RB"
    rb_rec  = merged["hist_targets_pg"].fillna(0.0)
    effective_targets = np.where(rb_mask, rb_rec, raw_targets)

    merged["proj_ppg"] = (
        raw_carries * merged["pts_per_carry"].fillna(0.0)
        + effective_targets * merged["pts_per_target"].fillna(0.0)
    )

    merged["_pos"]  = merged["position"].str.upper()
    merged["_team"] = merged["team"]
    merged["_age"]  = merged["age"]   # comes only from eff_cols join; NaN for rookies

    # ── 4b. Age curve adjustment ─────────────────────────────────────────────
    # Recency-weighted efficiency answers "what is this player's efficiency NOW."
    # The age curve projects that one more year forward to 2026.
    # Scaled at AGE_CURVE_SCALE (50%) so a young ascending player doesn't get
    # both recency boost AND full curve delta pushing in the same direction.
    from age_curves import fit_age_curves as _fit_curves
    _curves = _fit_curves(df)

    def _age_delta(pos: str, age) -> float:
        if pd.isna(age):
            return 0.0
        curve = _curves.get(pos)
        if curve is None:
            return 0.0
        return float(curve.expected_delta_ppg(float(age)))

    merged["age_curve_delta"] = [
        _age_delta(p, a)
        for p, a in zip(merged["_pos"], merged["_age"])
    ]
    merged["proj_ppg"] = (
        merged["proj_ppg"] + merged["age_curve_delta"] * AGE_CURVE_SCALE
    ).clip(lower=0)

    opp_proj = merged[[
        "player_name", "_pos", "_team", "_age", "proj_ppg", "role_tier", "age_curve_delta"
    ]].rename(columns={
        "player_name": "player_display_name",
        "_pos":  "position",
        "_team": "team",
        "_age":  "age",
    }).copy()
    opp_proj = opp_proj[opp_proj["proj_ppg"] > 0].reset_index(drop=True)

    # ── 5. QB projections (separate path — pass attempts not in season parquet) ─
    qb_proj = _project_qb_opportunity(df)

    # ── 6. Fallback: placeholder for unmodeled QBs + unfilled opp rows ────────
    placeholder = placeholder_projections(df)
    covered = set(opp_proj["player_display_name"].str.lower())
    if not qb_proj.empty:
        covered |= set(qb_proj["player_display_name"].str.lower())
    fallback = placeholder[~placeholder["player_display_name"].str.lower().isin(covered)]
    fallback = fallback.copy()
    fallback["age_curve_delta"] = 0.0   # no age adjustment for placeholder fallback

    parts = [opp_proj]
    if not qb_proj.empty:
        parts.append(qb_proj)
    parts.append(fallback)

    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates(subset="player_display_name", keep="first")
    result["age_curve_delta"] = result.get("age_curve_delta", pd.Series(dtype=float)).fillna(0.0)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weekly consistency metrics (for WR bucket classification)
# ---------------------------------------------------------------------------

def _compute_consistency() -> pd.DataFrame:
    """
    Score each WR/TE game from the last two seasons and compute:
      weekly_std    standard deviation of per-game fantasy points
      floor_rate    fraction of games scoring >= 8 ppg (startable threshold)
      weekly_median median per-game score

    These are used in mock_draft.py to sort WRs into bidding buckets:
    alpha (high floor, target certainty), upside (high variance), floor (reliable).

    Returns a DataFrame keyed by player_display_name. Players with fewer than
    8 recorded games are excluded — sample is too small to be meaningful.
    """
    from scoring import score_offense

    FLOOR_PPG = 8.0
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

    frames = []
    for yr in [2024, 2025]:
        path = os.path.join(data_dir, f"weekly_{yr}.parquet")
        if not os.path.exists(path):
            continue
        wk = pd.read_parquet(path)
        if "season_type" in wk.columns:
            wk = wk[wk["season_type"] == "REG"]
        if "position" in wk.columns:
            wk = wk[wk["position"].isin(["WR", "TE"])]
        frames.append(wk)

    if not frames:
        return pd.DataFrame(columns=["player_display_name", "weekly_std",
                                     "floor_rate", "weekly_median"])

    df = pd.concat(frames, ignore_index=True)
    df["game_pts"] = score_offense(df)

    agg = (
        df.groupby("player_display_name")
        .agg(
            weekly_std=("game_pts", "std"),
            weekly_median=("game_pts", "median"),
            floor_rate=("game_pts", lambda x: (x >= FLOOR_PPG).mean()),
            _n=("game_pts", "count"),
        )
        .reset_index()
    )
    return agg[agg["_n"] >= 8].drop(columns="_n").reset_index(drop=True)


# ---------------------------------------------------------------------------
# QB efficiency (reads from raw weekly files — season parquet lacks pass attempts)
# ---------------------------------------------------------------------------

def _compute_qb_efficiency() -> pd.DataFrame:
    """
    Per-game QB efficiency from 2023-2025 weekly data, with:
      - Season-level TD regression (Bayesian shrinkage toward QB median)
      - Recency weighting (2025:3, 2024:2, 2023:1 × games in season)

    Returns pts_per_pass_att and pts_per_rush for each QB.
    Medians are the fallback for rookies in _project_qb_opportunity.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    frames = []
    for yr in [2023, 2024, 2025]:
        path = os.path.join(data_dir, f"weekly_{yr}.parquet")
        if not os.path.exists(path):
            continue
        wk = pd.read_parquet(path)
        if "season_type" in wk.columns:
            wk = wk[wk["season_type"] == "REG"]
        if "position" in wk.columns:
            wk = wk[wk["position"] == "QB"]
        wk["_yr"] = yr
        frames.append(wk)

    if not frames:
        return pd.DataFrame(columns=["player_display_name", "pts_per_pass_att", "pts_per_rush"])

    all_weeks = pd.concat(frames, ignore_index=True)
    all_weeks = all_weeks[all_weeks["attempts"] > 10].copy()

    # ── Season-level aggregation ─────────────────────────────────────────────
    # Work at season level so TD regression uses the full seasonal sample,
    # then apply year weights when combining seasons.
    season = all_weeks.groupby(["player_display_name", "_yr"]).agg(
        attempts     = ("attempts",              "sum"),
        carries      = ("carries",               "sum"),
        pass_yds     = ("passing_yards",         "sum"),
        pass_tds     = ("passing_tds",           "sum"),
        pass_ints    = ("passing_interceptions", "sum"),
        rush_yds     = ("rushing_yards",         "sum"),
        rush_tds     = ("rushing_tds",           "sum"),
        rush_fum     = ("rushing_fumbles_lost",  "sum"),
        games        = ("week",                  "count"),
    ).reset_index().rename(columns={"_yr": "season"})

    season = season[season["attempts"] >= 100].copy()
    if season.empty:
        return pd.DataFrame(columns=["player_display_name", "pts_per_pass_att", "pts_per_rush"])

    # ── TD regression for QB passing ─────────────────────────────────────────
    # Passing TDs are the noisiest QB stat (r≈0.5 year-over-year).
    # Regress per-attempt TD rate toward the QB population median.
    season["_pass_td_rate"] = season["pass_tds"] / season["attempts"]
    pos_pass_td_prior = float(season["_pass_td_rate"].median())
    lam_pass = TD_REG_K_PASS_ATT / (TD_REG_K_PASS_ATT + season["attempts"])
    season["_pass_td_rate_reg"] = (
        (1 - lam_pass) * season["_pass_td_rate"] + lam_pass * pos_pass_td_prior
    )

    # Non-TD passing points per attempt (yards − INTs; stable year-over-year)
    season["_non_td_pass_ppa"] = (
        season["pass_yds"] / 25 - season["pass_ints"] * 2
    ) / season["attempts"]

    season["pts_per_pass_att"] = (
        season["_non_td_pass_ppa"] + season["_pass_td_rate_reg"] * 4
    )

    # ── TD regression for QB rushing ─────────────────────────────────────────
    rush_mask = season["carries"] > 10
    season["_rush_td_rate"] = np.where(
        rush_mask, season["rush_tds"] / season["carries"], np.nan
    )
    pos_rush_td_prior = float(season["_rush_td_rate"].median())
    lam_rush = TD_REG_K_CARRIES / (TD_REG_K_CARRIES + season["carries"].clip(lower=1))
    season["_rush_td_rate_reg"] = np.where(
        rush_mask,
        (1 - lam_rush) * season["_rush_td_rate"].fillna(pos_rush_td_prior)
        + lam_rush * pos_rush_td_prior,
        pos_rush_td_prior,
    )

    season["_non_td_rush_ppr"] = np.where(
        rush_mask,
        (season["rush_yds"] / 10 - season["rush_fum"] * 2) / season["carries"],
        np.nan,
    )
    season["pts_per_rush"] = np.where(
        rush_mask,
        season["_non_td_rush_ppr"] + season["_rush_td_rate_reg"] * 6,
        np.nan,
    )

    # ── Recency-weighted aggregation across seasons ──────────────────────────
    season["_yr_wt"]    = season["season"].map(YEAR_WEIGHTS).fillna(1.0)
    season["_total_wt"] = season["games"] * season["_yr_wt"]

    def _qb_wavg(g):
        def wmean(col):
            valid = g[col].notna()
            if not valid.any():
                return np.nan
            w = g.loc[valid, "_total_wt"]
            return float((g.loc[valid, col] * w).sum() / w.sum())
        return pd.Series({
            "pts_per_pass_att": wmean("pts_per_pass_att"),
            "pts_per_rush":     wmean("pts_per_rush"),
        })

    agg = season.groupby("player_display_name").apply(_qb_wavg).reset_index()
    return agg.reset_index(drop=True)


def _project_qb_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project QB ppg from qb_2026.csv using QB-specific efficiency.

    Returns same schema as opp_proj (player_display_name, position, team, age,
    proj_ppg, role_tier).  Returns empty DataFrame if no QB file exists.
    """
    from opportunity import load_opportunity, project_volume

    try:
        qb_opp = load_opportunity("QB")
    except FileNotFoundError:
        return pd.DataFrame()

    qb_vol = project_volume(qb_opp)
    qb_filled = qb_vol[qb_vol["role_tier"].notna() & (qb_vol["role_tier"].astype(str) != "")].copy()
    if qb_filled.empty:
        return pd.DataFrame()

    qb_eff = _compute_qb_efficiency()
    if qb_eff.empty:
        return pd.DataFrame()

    med_ppa = float(qb_eff["pts_per_pass_att"].median())
    med_ppr_series = qb_eff["pts_per_rush"].dropna()
    med_ppr = float(med_ppr_series.median()) if not med_ppr_series.empty else 0.5

    merged = qb_filled.merge(
        qb_eff, left_on="player_name", right_on="player_display_name", how="left"
    )
    merged["proj_ppg"] = (
        merged["proj_pass_att_pg"].fillna(0) * merged["pts_per_pass_att"].fillna(med_ppa)
        + merged["proj_rush_att_pg"].fillna(0) * merged["pts_per_rush"].fillna(med_ppr)
    )

    # Pull age and team from latest season data (NaN for rookies with no history)
    latest_qb = (
        df[(df.season == 2025) & (df.position == "QB")]
        .groupby("player_display_name")[["age"]]
        .last()
        .reset_index()
        .rename(columns={"age": "_age_hist"})
    )
    merged = merged.merge(latest_qb, left_on="player_name", right_on="player_display_name", how="left")

    # ── Age curve adjustment for QBs ─────────────────────────────────────────
    from age_curves import fit_age_curves as _fit_curves_qb
    _qb_curves = _fit_curves_qb(df)
    qb_curve = _qb_curves.get("QB")

    def _qb_age_delta(age) -> float:
        if qb_curve is None or pd.isna(age):
            return 0.0
        return float(qb_curve.expected_delta_ppg(float(age)))

    merged["age_curve_delta"] = merged["_age_hist"].apply(_qb_age_delta)
    merged["proj_ppg"] = (
        merged["proj_ppg"] + merged["age_curve_delta"] * AGE_CURVE_SCALE
    ).clip(lower=0)

    result = merged[merged["proj_ppg"] > 0].copy()
    return result[[
        "player_name", "position", "team", "proj_ppg", "role_tier", "_age_hist", "age_curve_delta",
    ]].rename(columns={
        "player_name": "player_display_name",
        "_age_hist": "age",
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rounding — integers that sum to exactly the right total
# ---------------------------------------------------------------------------

def _largest_remainder_round(raw: pd.Series, target: int) -> pd.Series:
    """
    Round a Series of floats to integers that sum to exactly target.

    1. Floor everyone.
    2. Give one extra dollar to the (target − floor_sum) players with the
       largest fractional remainders.

    Exact by construction — no off-by-one errors from plain rounding.
    """
    floors = raw.apply(np.floor).astype(int)
    deficit = target - floors.sum()
    if deficit < 0:
        raise ValueError(
            f"Floored prices already exceed target: {floors.sum()} > {target}"
        )
    fractions = raw - floors
    bonus_idx = fractions.nlargest(int(deficit)).index
    floors.loc[bonus_idx] += 1
    return floors


# ---------------------------------------------------------------------------
# Core valuation pipeline
# ---------------------------------------------------------------------------

def compute_values(
    proj_fn=None,
    min_bid: int = MIN_BID,
) -> pd.DataFrame:
    """
    Convert projected ppg → PAR → auction dollars.

    Parameters
    ----------
    proj_fn : callable(df) -> DataFrame, optional
        Projection source.  Defaults to placeholder_projections.
        Swap for Phase 3 output when the opportunity layer is populated.
    min_bid : int
        Minimum bid per player.  Default from league_config.

    Returns
    -------
    DataFrame with all projected players.  Column `is_drafted` marks the
    150 drafted slots (140 offense + 10 DEF).  Sorted by price descending.
    """
    if proj_fn is None:
        proj_fn = opportunity_projections

    df = pd.read_parquet(HISTORY)

    # ── 1. Replacement levels, bench-inclusive ──────────────────────────────
    deep_ranks = replacement_ranks(include_bench=True)
    repl = replacement_level(df, RECENT_SEASONS, deep_ranks)

    # ── 2. Projections ──────────────────────────────────────────────────────
    proj = proj_fn(df).copy()
    proj["replacement_ppg"] = proj["position"].map(repl)

    missing = proj["replacement_ppg"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} players have no replacement level. "
            "Check that all positions in the projection are in scarcity.py."
        )

    # ── 3. PAR: points above replacement for a full PROJ_GAMES season ───────
    proj["par"] = (
        (proj["proj_ppg"] - proj["replacement_ppg"]) * PROJ_GAMES
    ).clip(lower=0)

    # ── 4. Position-quota selection ─────────────────────────────────────────
    # Take the top N by PAR *within* each position.  This enforces the
    # roster distribution instead of letting PAR-gap differences between
    # positions dictate who gets drafted.
    off_parts = []
    for pos, quota in deep_ranks.items():
        pos_pool = proj[proj.position == pos].nlargest(quota, "par")
        off_parts.append(pos_pool)
    off_top = pd.concat(off_parts).copy()
    off_rest = proj[~proj.index.isin(off_top.index)].copy()

    n_off = len(off_top)    # should be 140 (28 + 42 + 54 + 16)
    total_dollars = NUM_TEAMS * BUDGET_PER_TEAM   # 2000

    # ── 5. Price the 140 offensive players ──────────────────────────────────
    # Money accounting:
    #   Total:                        $2,000
    #   DEF (10 × $1):                   $10   ← no discretionary
    #   Offensive minimums (140 × $1):  $140
    #   Offensive discretionary:       $1,850   (= 2000 − 10 − 140)
    off_disc = total_dollars - DEF_SLOTS * min_bid - n_off * min_bid

    total_par = off_top["par"].sum()
    if total_par == 0:
        raise ValueError(
            "Total offensive PAR is zero — all projections at or below replacement."
        )

    off_top["raw_price"] = min_bid + off_top["par"] / total_par * off_disc

    # Offensive prices must sum to $1,990 (= $2,000 − 10 DEF × $1)
    off_target = total_dollars - DEF_SLOTS * min_bid
    off_top["price"] = _largest_remainder_round(off_top["raw_price"], off_target)
    off_top["is_drafted"] = True

    # ── 6. DEF: all 32 NFL team DSTs, $1 each ───────────────────────────────
    # 10 reserved slots in the draft pool (is_drafted=True); all 32 are loggable
    # in the live draft assistant.  DST scoring is not modeled, so no PAR-based
    # price is possible — $1 for every team.
    NFL_TEAMS = sorted([
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN", "WAS",
    ])
    n_dst = len(NFL_TEAMS)
    def_rows = pd.DataFrame({
        "player_display_name": [f"{t} DST" for t in NFL_TEAMS],
        "position":            ["DEF"] * n_dst,
        "team":                NFL_TEAMS,
        "age":                 [np.nan] * n_dst,
        "proj_ppg":            [np.nan] * n_dst,
        "replacement_ppg":     [np.nan] * n_dst,
        "par":                 [0.0]  * n_dst,
        "raw_price":           [float(min_bid)] * n_dst,
        "price":               [min_bid] * n_dst,
        "is_drafted":          [True] * n_dst,
        "role_tier":           [np.nan] * n_dst,
    })

    off_rest["par"]        = 0.0
    off_rest["price"]      = min_bid
    off_rest["is_drafted"] = False

    # ── 7. Assertion — not optional ─────────────────────────────────────────
    price_sum = off_top["price"].sum() + DEF_SLOTS * min_bid
    assert price_sum == total_dollars, (
        f"BUG: prices sum to ${price_sum}, expected ${total_dollars}. "
        "Check _largest_remainder_round."
    )

    board = (
        pd.concat([off_top, def_rows, off_rest], ignore_index=True)
        .sort_values("price", ascending=False)
        .reset_index(drop=True)
    )

    # Merge weekly consistency for WR/TE bucket classification in mock_draft.
    # Defaults for players with no weekly history (rookies, QBs, DEF):
    #   weekly_std = 5.5  (near-median variance — neither alpha nor extreme upside)
    #   floor_rate = 0.40 (below the startable threshold most weeks)
    #   weekly_median = 6.0
    consistency = _compute_consistency()
    if not consistency.empty:
        board = board.merge(consistency, on="player_display_name", how="left")
    board["weekly_std"]    = board.get("weekly_std",    pd.Series(dtype=float)).fillna(5.5)
    board["floor_rate"]    = board.get("floor_rate",    pd.Series(dtype=float)).fillna(0.40)
    board["weekly_median"] = board.get("weekly_median", pd.Series(dtype=float)).fillna(6.0)

    return board


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df_hist = pd.read_parquet(HISTORY)
    deep_ranks = replacement_ranks(include_bench=True)
    repl = replacement_level(df_hist, RECENT_SEASONS, deep_ranks)

    board = compute_values()
    drafted_off = board[board.is_drafted & (board.position != "DEF")]

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 75)
    print("DARTS 2026 AUCTION BOARD  (Phase 3: opportunity-layer projections)")
    print("=" * 75)
    print()
    print("Replacement levels  (bench-inclusive, 2022-2025 average, min 10 games):")
    for pos in ("QB", "RB", "WR", "TE"):
        print(f"  {pos}: {repl[pos]:.1f} ppg  [drafted rank {deep_ranks[pos]}]")

    # ── Top 40 ───────────────────────────────────────────────────────────────
    print("\nTop 40 by price:")
    print(
        drafted_off.head(40)[
            ["player_display_name", "position", "team", "age", "proj_ppg", "par", "price"]
        ].to_string(index=False, float_format="%.1f")
    )

    # ── Position summary ─────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("\nPosition summary:")
    print(
        f"\n{'Pos':<5} {'N':>4}  {'AvgPPG':>8}  {'AvgPAR':>8}  "
        f"{'AvgPrice':>9}  {'TotalSpend':>11}  {'TopPrice':>9}"
    )
    print("─" * 62)
    total_spend = 0
    for pos in ("QB", "RB", "WR", "TE"):
        sub = drafted_off[drafted_off.position == pos]
        if sub.empty:
            continue
        spend = sub.price.sum()
        total_spend += spend
        print(
            f"{pos:<5} {len(sub):>4}  {sub.proj_ppg.mean():>8.1f}  "
            f"{sub.par.mean():>8.1f}  {sub.price.mean():>9.1f}  "
            f"${spend:>10,}  ${sub.price.max():>8}"
        )
    print(
        f"{'DEF':<5} {DEF_SLOTS:>4}  {'—':>8}  {'—':>8}  {'1.0':>9}  "
        f"${DEF_SLOTS:>10,}  ${'1':>8}"
    )
    total_spend += DEF_SLOTS
    print("─" * 62)
    print(f"{'TOTAL':<5} {len(drafted_off) + DEF_SLOTS:>4}  {'':>56}  ${total_spend:>8,}")

    # Slot confirmation and QB share
    slot_str = " + ".join(
        f"{pos}:{len(drafted_off[drafted_off.position == pos])}"
        for pos in ("QB", "RB", "WR", "TE")
    )
    print(f"\n  Slots: {slot_str} + DEF:{DEF_SLOTS} = {len(drafted_off) + DEF_SLOTS} total")

    qb_spend = drafted_off[drafted_off.position == "QB"]["price"].sum()
    print(f"  QB share of total spend: ${qb_spend} / $2,000 = {qb_spend / 2000 * 100:.1f}%")

    # ── QB sanity check ──────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("\nQB SANITY CHECK")
    print(
        "2QB league: replacement near QB22, not QB13.  "
        "Top QBs should cost more here than in standard 1QB auction values."
    )
    print()

    qbs = drafted_off[drafted_off.position == "QB"].sort_values("price", ascending=False).head(12)
    print(
        qbs[["player_display_name", "age", "proj_ppg", "replacement_ppg", "par", "price"]]
        .to_string(index=False, float_format="%.1f")
    )

    top_qb_price    = qbs["price"].iloc[0] if not qbs.empty else 0
    top_skill_price = drafted_off[drafted_off.position.isin(["RB", "WR"])]["price"].max()

    print(f"\n  Top QB ${top_qb_price}  vs.  top RB/WR ${top_skill_price}")

    # 1QB replacement derived from the same data — no magic numbers
    std_repl_qb = replacement_level(
        df_hist, RECENT_SEASONS, {"QB": STD_1QB_DRAFTED}
    )["QB"]
    our_repl    = repl["QB"]
    extra_par   = (std_repl_qb - our_repl) * PROJ_GAMES

    print(
        f"\n  Our QB replacement:      {our_repl:.1f} ppg  (bench-inclusive QB{deep_ranks['QB']})"
    )
    print(
        f"  Standard 1QB replacement: {std_repl_qb:.1f} ppg  "
        f"(empirical QB{STD_1QB_DRAFTED} in this data)"
    )
    print(
        f"  2QB premium: ~{extra_par:.0f} PAR points per QB per season "
        f"({std_repl_qb - our_repl:.1f} ppg × {PROJ_GAMES} games)"
    )
    print(
        "  Every QB in this league should be priced ~$10-20 higher than "
        "standard 1QB auction rankings."
    )

    if top_qb_price < top_skill_price * 0.80:
        print(
            f"\n  ⚠  WARNING: Top QB (${top_qb_price}) is well below the top skill player "
            f"(${top_skill_price}).  For a 2QB league this is almost certainly wrong. "
            f"Our QB replacement is {our_repl:.1f} ppg vs. {std_repl_qb:.1f} ppg in 1QB — "
            "if those are close, bench-inclusive QB ranks may not be loading correctly."
        )
    else:
        print("\n  ✓  QB pricing looks consistent with 2QB replacement level.")

    # ── Budget check ─────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print(
        f"\nTotal draft spend: $2,000  "
        f"(${total_spend - TOTAL_PLAYERS_DRAFTED * MIN_BID} discretionary "
        f"+ ${TOTAL_PLAYERS_DRAFTED * MIN_BID} minimums)"
    )
    print("Assertion passed: prices sum to exactly $2,000.  ✓")


if __name__ == "__main__":
    main()
