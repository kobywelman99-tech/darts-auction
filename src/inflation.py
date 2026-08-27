"""
Phase 4.5 — Live draft-night inflation multiplier.

WHAT THIS IS
------------
During the auction, as expensive players are taken early, the remaining
money-per-PAR ratio shifts. If the market is spending faster than the model
predicts, the late-pick bargains become more extreme.

The multiplier compares the actual remaining $/PAR to the model's baseline:

    baseline_rate   = DISCRETIONARY_MONEY / total_drafted_PAR      (from valuation)
    remaining_money = league_budget_left - remaining_roster_minimums
    remaining_PAR   = sum of PAR for all undrafted players

    inflation = remaining_$/PAR / baseline_$/PAR

Interpretation:
  inflation > 1.0 → market is spending FASTER than model → players cost more
  inflation < 1.0 → market is spending SLOWER → late bargains are real

PURCHASING POWER (the number that actually matters on draft night)
------------------------------------------------------------------
A leaguewide multiplier of 0.07 is useless if everyone including you is
broke. What matters is your budget relative to the room's average:

    avg_per_team   = league_remaining_money / teams_still_with_spots
    my_power       = my_remaining_money / avg_per_team

Above 1.0 = you can outbid the average remaining team. This is when you act.
Below 1.0 = you are budget-constrained; others have more leverage.

INFLATION-ADJUSTED PRICES
--------------------------
Only the discretionary portion of a model price inflates.
The $1 minimum is fixed:

    inflation_price = 1 + (raw_price - 1) * multiplier

BACKTEST
--------
Replaying draft_2025.csv pick-by-pick shows how both numbers evolved.
Run: python src/inflation.py --backtest

LIVE USE
--------
Pass already_drafted as a list of (player_display_name, salary_paid) tuples,
plus my_remaining_money, to get the full picture.

Run:  python src/inflation.py
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from league_config import (
    BUDGET_PER_TEAM,
    MIN_BID,
    NUM_TEAMS,
    TOTAL_PLAYERS_DRAFTED,
)
from valuation import compute_values, DEF_SLOTS

DRAFT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "drafts")

TOTAL_MONEY = NUM_TEAMS * BUDGET_PER_TEAM          # $2,000
TOTAL_MINIMUMS = TOTAL_PLAYERS_DRAFTED * MIN_BID   # $150


def _baseline_rate(board: pd.DataFrame) -> float:
    """Baseline $/PAR: discretionary money divided by total drafted PAR."""
    drafted_off = board[board.is_drafted & (board.position != "DEF")]
    total_par = drafted_off["par"].sum()
    discretionary = TOTAL_MONEY - TOTAL_MINIMUMS
    return discretionary / total_par if total_par > 0 else np.nan


def inflation_multiplier(
    board: pd.DataFrame,
    already_drafted: Sequence[tuple[str, int]],
    teams_still_bidding: int | None = None,
    my_remaining_money: int | None = None,
) -> dict:
    """
    Compute the inflation multiplier and your personal purchasing power.

    Parameters
    ----------
    board : DataFrame from compute_values()
    already_drafted : list of (player_display_name, salary_paid)
        All players taken so far leaguewide, with actual dollar paid.
    teams_still_bidding : int, optional
        Teams that still have roster spots. Defaults to NUM_TEAMS.
    my_remaining_money : int, optional
        Your current remaining budget. If provided, my_purchasing_power
        is computed; otherwise it is NaN.

    Returns
    -------
    dict with keys:
      multiplier          — league $/PAR relative to model baseline
      baseline_rate       — model $/PAR
      current_rate        — current league $/PAR
      money_spent         — total dollars spent leaguewide so far
      picks_made          — players drafted so far
      remaining_money     — league discretionary money left
      remaining_par       — PAR of undrafted model players
      my_purchasing_power — my_remaining / avg_remaining_per_team
                            > 1.0 means you can outbid the room
      adjusted_board      — board with inflation_price column added
    """
    if teams_still_bidding is None:
        teams_still_bidding = NUM_TEAMS

    drafted_names = {name.lower().strip() for name, _ in already_drafted}
    money_spent = sum(salary for _, salary in already_drafted)
    picks_made = len(already_drafted)

    # Remaining slots leaguewide
    remaining_slots = TOTAL_PLAYERS_DRAFTED - picks_made
    remaining_minimums = remaining_slots * MIN_BID

    remaining_money = TOTAL_MONEY - money_spent - remaining_minimums
    remaining_money = max(remaining_money, 0)

    # Undrafted players on the model board
    board_lower = board.copy()
    board_lower["_name_lower"] = board_lower["player_display_name"].str.lower().str.strip()
    undrafted = board_lower[
        board_lower["is_drafted"] &
        (board_lower["position"] != "DEF") &
        ~board_lower["_name_lower"].isin(drafted_names)
    ].copy()

    remaining_par = undrafted["par"].sum()
    baseline = _baseline_rate(board)

    if remaining_par > 0 and remaining_money > 0:
        current_rate = remaining_money / remaining_par
        multiplier = current_rate / baseline
    else:
        current_rate = np.nan
        multiplier = np.nan

    # Purchasing power: my budget relative to the room's average
    if my_remaining_money is not None and teams_still_bidding > 0 and remaining_money > 0:
        avg_remaining = remaining_money / teams_still_bidding
        my_purchasing_power = my_remaining_money / avg_remaining
    else:
        my_purchasing_power = np.nan

    # Inflation-adjusted prices for remaining players.
    # Only the discretionary portion (raw_price - 1) inflates; the $1
    # minimum is fixed regardless of market conditions.
    adj_board = board.copy()
    if not np.isnan(multiplier):
        adj_board["inflation_price"] = (
            1 + (adj_board["raw_price"] - 1) * multiplier
        ).round(0).astype("Int64")
        adj_board.loc[adj_board["inflation_price"] < 1, "inflation_price"] = 1
    else:
        adj_board["inflation_price"] = adj_board["price"].astype("Int64")

    return {
        "multiplier": round(multiplier, 3) if not np.isnan(multiplier) else np.nan,
        "baseline_rate": round(baseline, 3),
        "current_rate": round(current_rate, 3) if not np.isnan(current_rate) else np.nan,
        "money_spent": money_spent,
        "picks_made": picks_made,
        "remaining_money": remaining_money,
        "remaining_par": round(remaining_par, 1),
        "my_purchasing_power": (
            round(my_purchasing_power, 3) if not np.isnan(my_purchasing_power) else np.nan
        ),
        "adjusted_board": adj_board,
    }


# ---------------------------------------------------------------------------
# Backtest against draft_2025.csv
# ---------------------------------------------------------------------------

def backtest(season: int = 2025, report_at: Sequence[int] = (20, 50, 75, 100)) -> None:
    """
    Replay a historical draft pick-by-pick and report the inflation multiplier
    and purchasing power at specified pick numbers.

    Hypothetical team: spent $0 in the first 20 skill picks, keeping the full
    $200 budget intact. This shows how large the purchasing power advantage
    gets and when it peaks.
    """
    path = os.path.join(DRAFT_DIR, f"draft_{season}.csv")
    draft = pd.read_csv(path)
    skill_draft = draft[draft["position"].isin(["QB", "RB", "WR", "TE"])].sort_values("pick")

    board = compute_values()
    baseline = _baseline_rate(board)

    # Hypothetical team that passed on the first 20 picks entirely
    HYPO_BUDGET = BUDGET_PER_TEAM   # started with $200, spent $0

    print(f"\nBacktest: {season} draft  (inflation + purchasing power replay)")
    print(f"Model baseline: ${baseline:.2f} per PAR point")
    print(f"Hypothetical team: sat out first 20 picks, kept full ${HYPO_BUDGET} budget\n")
    print(
        f"{'Pick':>5}  {'Player':<26}  {'Pos':>3}  {'Paid':>5}  "
        f"{'Spent':>7}  {'LgLeft':>7}  {'Mult':>6}  {'MyBudget':>9}  {'MyPower':>8}"
    )
    print("─" * 88)

    running = []
    report_set = set(report_at)
    cumulative_pick = 0

    # Hypo team's remaining budget: $200 until pick 20, then shrinks as they
    # must spend their money too. For simplicity we hold it at $200 throughout —
    # this is the maximum purchasing power scenario (they spent nothing yet).
    hypo_remaining = HYPO_BUDGET

    for _, row in skill_draft.iterrows():
        cumulative_pick += 1
        running.append((row["player"], int(row["salary"])))

        if cumulative_pick in report_set or cumulative_pick <= 5:
            result = inflation_multiplier(
                board, running,
                my_remaining_money=hypo_remaining,
            )
            power = result["my_purchasing_power"]
            power_str = f"{power:.2f}x" if not np.isnan(power) else "  —  "
            print(
                f"{cumulative_pick:>5}  {row['player']:<26}  {row['position']:>3}  "
                f"${row['salary']:>4}  "
                f"${result['money_spent']:>6,}  ${result['remaining_money']:>6,}  "
                f"{result['multiplier']:>6.3f}  ${hypo_remaining:>8,}  {power_str:>8}"
            )

    result = inflation_multiplier(board, running)
    print(f"\n  Skill picks in {season} draft: {cumulative_pick} of {len(draft)} total picks")
    print(f"  Total skill spend: ${result['money_spent']:,}")
    print(f"  Total draft spend: ${draft['salary'].sum():,}")
    print(
        f"\n  At pick 20, a team with $200 left had {_power_at_pick(board, skill_draft, 20, HYPO_BUDGET):.2f}x "
        f"the average team's purchasing power."
    )
    print(
        f"  At pick 50, that same $200 gave {_power_at_pick(board, skill_draft, 50, HYPO_BUDGET):.2f}x "
        f"purchasing power."
    )


def _power_at_pick(board, skill_draft, pick_num, my_budget):
    """Helper: compute purchasing power after pick_num skill picks."""
    rows = skill_draft.head(pick_num)
    drafted = [(r["player"], int(r["salary"])) for _, r in rows.iterrows()]
    result = inflation_multiplier(board, drafted, my_remaining_money=my_budget)
    return result["my_purchasing_power"]


# ---------------------------------------------------------------------------
# Main — demo with pick-50 snapshot from 2025 draft
# ---------------------------------------------------------------------------

def main() -> None:
    import sys as _sys

    if "--backtest" in _sys.argv:
        backtest()
        return

    board = compute_values()

    # Demo: simulate the first 10 picks of the 2025 draft and check multiplier
    demo_picks = [
        ("Ja'Marr Chase", 60),
        ("Puka Nacua", 35),
        ("Bijan Robinson", 57),
        ("Justin Jefferson", 48),
        ("Josh Allen", 60),
        ("Saquon Barkley", 68),
        ("Patrick Mahomes", 37),
        ("Joe Burrow", 46),
        ("Jahmyr Gibbs", 50),
        ("CeeDee Lamb", 49),
    ]

    my_budget = 120   # demo: spent $80 in first 10 picks
    result = inflation_multiplier(board, demo_picks, my_remaining_money=my_budget)
    print("Demo: after picks 1-10 of the 2025 draft (my remaining budget: $120)")
    print(f"  League spent:        ${result['money_spent']:,}")
    print(f"  League remaining:    ${result['remaining_money']:,}")
    print(f"  Remaining PAR:       {result['remaining_par']:.0f}")
    print(f"  Baseline rate:       ${result['baseline_rate']:.2f}/PAR")
    print(f"  Current rate:        ${result['current_rate']:.2f}/PAR")
    print(f"  Inflation mult:      {result['multiplier']:.3f}x")
    print(f"  My purchasing power: {result['my_purchasing_power']:.2f}x  "
          f"({'above' if result['my_purchasing_power'] > 1 else 'below'} avg)")
    print()
    print("Top 10 by inflation-adjusted price (remaining players):")
    adj = result["adjusted_board"]
    remaining = adj[
        adj["is_drafted"] &
        (adj["position"] != "DEF") &
        ~adj["player_display_name"].str.lower().isin({n.lower() for n, _ in demo_picks})
    ].nlargest(10, "inflation_price")[
        ["player_display_name", "position", "price", "inflation_price", "par"]
    ]
    print(remaining.to_string(index=False))
    print()
    print("Run with --backtest to replay the full 2025 draft pick-by-pick.")


if __name__ == "__main__":
    main()
