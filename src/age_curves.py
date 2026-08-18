"""
Phase 2 — Age curves and trajectory detection.

How this works
--------------
We use the *delta method*: instead of averaging ppg by age across players
(which suffers from survivorship bias — only the best players survive to 35),
we compute year-over-year changes for the same player and average those.
The output is: "at age X, a player at this position should change by Y ppg."

Separate curves are fit for pts_non_td and pts_from_td because TDs are noisy
enough to swamp the age signal if left combined.

TD regression: each player's pts_from_td is pulled toward the positional mean
before projecting forward, because year-to-year TD correlation is low (~0.3).
The player's actual pts_non_td (efficiency-based production) is unchanged.

Team changes are flagged on the trajectory label so Phase 3 knows which
players need situational review rather than a pure age-curve projection.

Run:  python src/age_curves.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — adjust here if you want to tune the model
# ---------------------------------------------------------------------------
MIN_GAMES = 8        # both seasons in a delta pair must meet this
TD_LAMBDA = 0.50     # how far to pull pts_from_td toward the positional mean
                     # 0 = no regression, 1 = replace with positional mean
POSITIONS = ("QB", "RB", "WR", "TE")
RECENT_SEASONS = [2022, 2023, 2024, 2025]  # window for computing TD mean baseline

# Trajectory thresholds, in ppg vs. age-curve expectation
GROWTH_THRESHOLD = 1.0          # >1.0 ppg above curve → growth
ON_CURVE_BAND = 1.5             # within ±1.5 → on_curve
SHARP_DECLINE_THRESHOLD = 3.0   # >3.0 below curve → sharp_decline
TD_VARIANCE_GAP = 0.75          # pts_from_td_pg must be this far below pos mean
                                 # to count as "down year driven by TD variance"

# Multi-season baseline settings
MAX_LOOKBACK = 3                     # how many prior seasons to blend
RECENCY_WEIGHTS = (1.0, 0.5, 0.25)  # weight for 1, 2, 3 seasons back (halving)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                    "player_seasons.parquet")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                        "trajectory_2025.parquet")


# ---------------------------------------------------------------------------
# Step 1 — build consecutive-season pairs (the raw material for the curve)
# ---------------------------------------------------------------------------

def _build_pairs(df: pd.DataFrame, min_games: int = MIN_GAMES) -> pd.DataFrame:
    """
    For each player, pair adjacent seasons where both have >= min_games.

    We only include truly consecutive seasons (no gap years), because a player
    returning from a missed year carries injury noise that would distort the
    age signal.
    """
    qualifying = (df[df.games >= min_games]
                  .sort_values(["player_id", "season"])
                  .copy())

    records = []
    for pid, grp in qualifying.groupby("player_id", sort=False):
        rows = grp.reset_index(drop=True)
        for i in range(len(rows) - 1):
            r0 = rows.iloc[i]
            r1 = rows.iloc[i + 1]
            if int(r1.season) != int(r0.season) + 1:
                continue   # gap year — skip

            records.append({
                "player_id": pid,
                "player_display_name": r0.player_display_name,
                "position": r0.position,
                "season_from": int(r0.season),
                "season_to": int(r1.season),
                "age": float(r0.age),           # age at the *start* season
                "games_from": int(r0.games),
                "games_to": int(r1.games),
                "team_changed": int(r0.team != r1.team),
                # The deltas we want to model
                "delta_ppg": float(r1.ppg - r0.ppg),
                "delta_non_td_pg": float(
                    r1.pts_non_td / r1.games - r0.pts_non_td / r0.games
                ),
                "delta_td_pg": float(
                    r1.pts_from_td / r1.games - r0.pts_from_td / r0.games
                ),
                # Efficiency metrics (position-appropriate; NaN is OK here)
                "delta_ypt": float(r1.yards_per_target - r0.yards_per_target)
                             if not (np.isnan(r1.yards_per_target)
                                     or np.isnan(r0.yards_per_target))
                             else np.nan,
                "delta_ypc": float(r1.yards_per_carry - r0.yards_per_carry)
                             if not (np.isnan(r1.yards_per_carry)
                                     or np.isnan(r0.yards_per_carry))
                             else np.nan,
                "delta_catch_rate": float(r1.catch_rate - r0.catch_rate)
                                    if not (np.isnan(r1.catch_rate)
                                            or np.isnan(r0.catch_rate))
                                    else np.nan,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 2 — fit position-specific quadratic age curves
# ---------------------------------------------------------------------------

@dataclass
class AgeCurve:
    """
    Quadratic model of year-over-year change at each age.

    Coefficients follow np.polyval convention: highest degree first.
    expected_delta(age) = coefs[0]*age² + coefs[1]*age + coefs[2]

    age_min / age_max are the 5th/95th percentile of the training data.
    Predictions outside this range are extrapolation and should be treated
    skeptically — flag them rather than trust them.
    """
    position: str
    ppg_coefs: np.ndarray
    non_td_coefs: np.ndarray
    td_coefs: np.ndarray
    age_min: float
    age_max: float
    n_pairs: int

    def expected_delta_ppg(self, age: float) -> float:
        age_clipped = float(np.clip(age, self.age_min, self.age_max))
        return float(np.polyval(self.ppg_coefs, age_clipped))

    def expected_delta_non_td(self, age: float) -> float:
        age_clipped = float(np.clip(age, self.age_min, self.age_max))
        return float(np.polyval(self.non_td_coefs, age_clipped))

    def is_extrapolating(self, age: float) -> bool:
        return age < self.age_min or age > self.age_max


def fit_age_curves(df: pd.DataFrame,
                   min_games: int = MIN_GAMES) -> dict[str, AgeCurve]:
    """
    Fit quadratic age curves using the delta method.

    Observations are weighted by the harmonic mean of games played in both
    seasons — a 16-game season paired with a 16-game season counts more than
    an 8-game season paired with a 12-game season.
    """
    pairs = _build_pairs(df, min_games)
    curves: dict[str, AgeCurve] = {}

    for pos in POSITIONS:
        sub = pairs[(pairs.position == pos)
                    & pairs.delta_ppg.notna()
                    & pairs.age.notna()].copy()

        if len(sub) < 15:
            print(f"  WARNING: {pos} only has {len(sub)} pairs — "
                  "curve is unreliable at the tails")

        # Harmonic mean of games: penalises pairs where one season is short
        weights = 2.0 / (1.0 / sub.games_from + 1.0 / sub.games_to)

        ppg_coefs = np.polyfit(sub.age, sub.delta_ppg, deg=2, w=weights)
        non_td_coefs = np.polyfit(sub.age, sub.delta_non_td_pg.fillna(0),
                                  deg=2, w=weights)
        td_coefs = np.polyfit(sub.age, sub.delta_td_pg.fillna(0),
                              deg=2, w=weights)

        curves[pos] = AgeCurve(
            position=pos,
            ppg_coefs=ppg_coefs,
            non_td_coefs=non_td_coefs,
            td_coefs=td_coefs,
            age_min=float(sub.age.quantile(0.05)),
            age_max=float(sub.age.quantile(0.95)),
            n_pairs=len(sub),
        )

    return curves


# ---------------------------------------------------------------------------
# Step 3 — multi-season weighted baseline
# ---------------------------------------------------------------------------

def _multi_season_baseline(
    prior_seasons: pd.DataFrame,
    target_season: int,
    curve: AgeCurve,
    max_lookback: int = MAX_LOOKBACK,
    recency_weights: tuple[float, ...] = RECENCY_WEIGHTS,
) -> float:
    """
    Games-weighted, recency-weighted baseline PPG prediction for target_season.

    For each of the last max_lookback qualifying prior seasons, project that
    season's PPG forward to target_season by compounding year-by-year expected
    curve deltas, then combine with weight = games × recency_weight.

    RECENCY_WEIGHTS = (1.0, 0.5, 0.25) means a 17-game season from last year
    gets weight 17.0, the same 17-game season from two years ago gets 8.5, and
    from three years ago gets 4.25.  An 8-game injury year from last season
    gets weight 8.0 — less than a healthy full season from two years ago (8.5).
    This is the key fix: injury-shortened seasons no longer monopolise the
    baseline just because they were the most recent.

    prior_seasons must already be filtered to games >= min_games.
    """
    lookback = (prior_seasons
                .sort_values("season", ascending=False)
                .head(max_lookback)
                .reset_index(drop=True))
    if lookback.empty:
        return np.nan

    weighted_sum = 0.0
    weight_sum = 0.0
    for i, season_row in lookback.iterrows():
        steps_forward = target_season - int(season_row.season)
        projected = float(season_row.ppg)
        age = float(season_row.age)
        for _ in range(steps_forward):
            projected += curve.expected_delta_ppg(age)
            age += 1.0

        rw = recency_weights[i] if i < len(recency_weights) else recency_weights[-1]
        w = float(season_row.games) * rw
        weighted_sum += w * projected
        weight_sum += w

    return weighted_sum / weight_sum if weight_sum > 0 else np.nan


# ---------------------------------------------------------------------------
# Step 4 — TD regression toward positional mean
# ---------------------------------------------------------------------------

def _positional_td_means(df: pd.DataFrame,
                          seasons: list[int],
                          min_games: int = MIN_GAMES) -> dict[str, float]:
    """
    Mean pts_from_td per game by position over recent seasons.
    This is the anchor we regress individual player TD rates toward.
    """
    recent = df[(df.season.isin(seasons)) & (df.games >= min_games)].copy()
    recent["td_pg"] = recent.pts_from_td / recent.games
    return recent.groupby("position")["td_pg"].mean().to_dict()


def td_regressed_ppg(row: pd.Series,
                     pos_td_means: dict[str, float],
                     td_lambda: float = TD_LAMBDA) -> float:
    """
    Pull a player's pts_from_td toward the positional mean before projecting.

    Example: a WR who scored 8 TDs in a year where 4 is typical will have
    roughly half of that excess pulled back.  His efficiency-based production
    (pts_non_td) is untouched — we only regress the noisy part.
    """
    if row.games == 0:
        return float(row.ppg)
    actual_td_pg = row.pts_from_td / row.games
    pos_mean = pos_td_means.get(row.position, actual_td_pg)
    adjusted_td_pg = (1.0 - td_lambda) * actual_td_pg + td_lambda * pos_mean
    non_td_pg = row.pts_non_td / row.games
    return float(non_td_pg + adjusted_td_pg)


# ---------------------------------------------------------------------------
# Step 4 — efficiency trend per player
# ---------------------------------------------------------------------------

def _get_efficiency_value(row: pd.Series, pos: str) -> float:
    """Position-appropriate efficiency metric for a single season."""
    if pos in ("WR", "TE"):
        return float(row.yards_per_target)
    if pos == "RB":
        return float(row.yards_per_carry)
    # QB: non-TD points per game captures passing yards + rushing without TD noise
    return float(row.pts_non_td / row.games) if row.games > 0 else np.nan


def _efficiency_trend(player_seasons: pd.DataFrame, pos: str) -> str:
    """
    Compare the most recent season's efficiency to the player's prior career.

    We use a 1-SD threshold: if the recent season is more than one standard
    deviation below the player's own career average, we call it 'declining'.
    One bad year inside the noise band is 'stable'.

    Returns 'declining', 'stable', or 'insufficient_data'.
    """
    qualified = player_seasons.sort_values("season").reset_index(drop=True)
    if len(qualified) < 2:
        return "insufficient_data"

    recent_val = _get_efficiency_value(qualified.iloc[-1], pos)
    prior_vals = [
        _get_efficiency_value(qualified.iloc[i], pos)
        for i in range(len(qualified) - 1)
    ]
    prior_vals = [v for v in prior_vals if not np.isnan(v)]

    if not prior_vals or np.isnan(recent_val):
        return "insufficient_data"

    prior_mean = np.mean(prior_vals)
    prior_std = np.std(prior_vals) if len(prior_vals) > 1 else 0.0

    if prior_std < 0.01:
        # No meaningful variation in prior seasons — can't meaningfully compare
        return "stable"

    z = (recent_val - prior_mean) / prior_std
    return "declining" if z < -1.0 else "stable"


# ---------------------------------------------------------------------------
# Step 5 — trajectory label
# ---------------------------------------------------------------------------

def _trajectory_label(
    age_adj: float,
    eff_trend: str,
    td_gap: float,
    team_changed: bool,
) -> str:
    """
    Assign one of five trajectory labels.

    age_adj:     actual ppg minus what the age curve predicted from last year
    eff_trend:   'stable' or 'declining' from _efficiency_trend
    td_gap:      player's pts_from_td_pg minus positional mean — negative means
                 the player scored fewer TDs than average this year
    team_changed: True if the player switched teams between seasons

    Priority order:
    1. TD variance — underperformed the curve, but efficiency held and TDs
       were the culprit (strong regression-to-mean candidate)
    2. Growth — meaningfully outperformed the curve
    3. On-curve — within the noise band
    4. Mild / sharp decline — underperformed and efficiency is also declining
    """
    # TD variance: the down year is mostly explained by below-average TDs,
    # and the player's efficiency stats didn't decline — likely to revert
    td_variance = (
        age_adj < -ON_CURVE_BAND
        and eff_trend != "declining"
        and td_gap < -TD_VARIANCE_GAP
    )
    if td_variance:
        return "td_variance_team_change" if team_changed else "td_variance"

    if age_adj >= GROWTH_THRESHOLD:
        return "growth"
    if age_adj >= -ON_CURVE_BAND:
        return "on_curve"
    if age_adj >= -SHARP_DECLINE_THRESHOLD:
        return "mild_decline"
    return "sharp_decline"


# ---------------------------------------------------------------------------
# Step 6 — build the full trajectory table
# ---------------------------------------------------------------------------

def build_trajectory_table(
    df: pd.DataFrame,
    curves: dict[str, AgeCurve],
    recent_season: int = 2025,
    min_games: int = MIN_GAMES,
    td_lambda: float = TD_LAMBDA,
) -> pd.DataFrame:
    """
    Enrich each qualifying player from recent_season with:
      - ppg_td_regressed   — ppg with pts_from_td pulled toward positional mean
      - ppg_curve_projected — naive 2026 baseline (TD-regressed + age curve delta)
      - age_adjusted_ppg   — how far the player landed vs. their age-curve
                             expectation coming into 2025
      - efficiency_trend   — stable / declining / insufficient_data
      - trajectory_signal  — five-category label for Phase 3 to act on
      - team_changed       — whether the player switched teams for this season

    Players without a qualifying prior season get NaN for age_adjusted_ppg
    and 'insufficient_data' for trajectory_signal.  Phase 3 should treat them
    as unknowns and rely more heavily on the opportunity layer.
    """
    pos_td_means = _positional_td_means(df, RECENT_SEASONS, min_games)

    qualifying = (df[(df.season == recent_season) & (df.games >= min_games)]
                  .copy())
    prior_all = df[(df.season < recent_season) & (df.games >= min_games)]

    records = []
    for _, row in qualifying.iterrows():
        pos = row.position
        curve = curves.get(pos)

        # ---- multi-season weighted baseline ----
        prior_for_player = (prior_all[prior_all.player_id == row.player_id]
                            .sort_values("season"))

        age_adj = np.nan
        team_changed = False
        extrapolating = False

        if curve is not None and not prior_for_player.empty:
            baseline = _multi_season_baseline(prior_for_player, recent_season, curve)
            if not np.isnan(baseline):
                age_adj = float(row.ppg) - baseline
                most_recent_prior = prior_for_player.iloc[-1]
                team_changed = (row.team != most_recent_prior.team)
                extrapolating = curve.is_extrapolating(float(most_recent_prior.age))

        # ---- efficiency trend across all qualifying seasons ----
        all_for_player = (df[(df.player_id == row.player_id)
                             & (df.games >= min_games)])
        eff_trend = _efficiency_trend(all_for_player, pos)

        # ---- TD gap: how far was this player from the positional TD mean? ----
        actual_td_pg = (float(row.pts_from_td) / row.games
                        if row.games > 0 else np.nan)
        pos_mean_td = pos_td_means.get(pos, np.nan)
        td_gap = (actual_td_pg - pos_mean_td
                  if not np.isnan(actual_td_pg) and not np.isnan(pos_mean_td)
                  else 0.0)

        # ---- trajectory label ----
        if np.isnan(age_adj):
            traj = "insufficient_data"
        else:
            traj = _trajectory_label(age_adj, eff_trend, td_gap, team_changed)

        # ---- TD-regressed ppg ----
        ppg_td_reg = td_regressed_ppg(row, pos_td_means, td_lambda)

        # ---- naive 2026 age-curve projection ----
        # TD-regressed ppg + expected change at their current age
        if curve is not None:
            next_delta = curve.expected_delta_ppg(float(row.age))
            ppg_projected = ppg_td_reg + next_delta
        else:
            ppg_projected = ppg_td_reg

        records.append({
            "player_id": row.player_id,
            "player_display_name": row.player_display_name,
            "position": row.position,
            "team": row.team,
            "season": int(row.season),
            "age": float(row.age),
            "games": int(row.games),
            "ppg": float(row.ppg),
            "pts_from_td_pg": actual_td_pg,
            "pts_non_td_pg": (float(row.pts_non_td) / row.games
                              if row.games > 0 else np.nan),
            "ppg_td_regressed": ppg_td_reg,
            "ppg_curve_projected": ppg_projected,
            "age_adjusted_ppg": age_adj,
            "efficiency_trend": eff_trend,
            "trajectory_signal": traj,
            "team_changed": team_changed,
            "td_gap_vs_pos_mean": td_gap,
            "age_curve_extrapolating": extrapolating,
        })

    return (pd.DataFrame(records)
            .sort_values(["position", "ppg"], ascending=[True, False])
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Main — print a readable summary and save the trajectory table
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_parquet(DATA)

    print("=" * 60)
    print("PHASE 2 — AGE CURVES (delta method, quadratic, per position)")
    print("=" * 60)
    print(f"\nMin games filter: {MIN_GAMES}  |  TD regression λ: {TD_LAMBDA}")
    print(f"TD λ=0.5 means half of the excess/deficit vs. positional mean")
    print(f"is pulled back before projecting forward.\n")

    curves = fit_age_curves(df)

    print("\nExpected year-over-year Δppg by age:")
    print(f"{'Age':>4}", end="")
    for pos in POSITIONS:
        print(f"  {pos:>7}", end="")
    print()

    for age in range(22, 37):
        print(f"{age:>4}", end="")
        for pos in POSITIONS:
            c = curves[pos]
            if c.age_min <= age <= c.age_max:
                delta = c.expected_delta_ppg(age)
                print(f"  {delta:>+7.2f}", end="")
            else:
                print(f"  {'(extrap)':>7}", end="")
        print()

    print(f"\n{'Position':<6} {'Pairs':>6} {'Age range':>12}")
    for pos in POSITIONS:
        c = curves[pos]
        print(f"{pos:<6} {c.n_pairs:>6} "
              f"  {c.age_min:.0f}–{c.age_max:.0f}")

    print("\n" + "=" * 60)
    print("2025 TRAJECTORY TABLE")
    print("=" * 60)

    traj = build_trajectory_table(df, curves)

    print(f"\nTrajectory signal distribution:")
    print(traj.trajectory_signal.value_counts().to_string())

    display_cols = [
        "player_display_name", "position", "team", "age",
        "ppg", "ppg_td_regressed", "ppg_curve_projected",
        "age_adjusted_ppg", "efficiency_trend", "trajectory_signal",
        "team_changed",
    ]

    for signal in ("growth", "td_variance", "td_variance_team_change",
                   "mild_decline", "sharp_decline"):
        subset = traj[traj.trajectory_signal == signal].head(6)
        if subset.empty:
            continue
        print(f"\n--- {signal.upper()} ---")
        print(subset[display_cols].to_string(index=False, float_format="%.1f"))

    # TD regression impact: show where it moves the needle most
    traj["td_reg_delta"] = traj.ppg_td_regressed - traj.ppg
    movers = traj.reindex(traj.td_reg_delta.abs().nlargest(8).index)
    print("\n--- BIGGEST TD REGRESSION ADJUSTMENTS ---")
    print(movers[["player_display_name", "position", "ppg",
                  "ppg_td_regressed", "td_reg_delta"]]
          .to_string(index=False, float_format="%.1f"))

    traj.drop(columns=["td_reg_delta"]).to_parquet(OUT_PATH, index=False)
    print(f"\nTrajectory table saved → {OUT_PATH}")
    print(f"({len(traj)} players, 2025 season, >= {MIN_GAMES} games)")


if __name__ == "__main__":
    main()
