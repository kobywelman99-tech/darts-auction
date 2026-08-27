"""
Phase 4.5 — Market prices and positional spending patterns.

Market price is LEAGUE BEHAVIOR data, not a projection. It tells you what
this league paid for a player last year. It is a prior about bidding culture,
not an estimate of 2026 production. It must never feed back into the
valuation model — it lives as a separate comparison column only.

WHAT IS VALID SIGNAL HERE
--------------------------
Player-level gaps (hindsight_gap) are CONTAMINATED: the 2025 draft prices
were set before the 2025 season, but the model uses 2025 actual ppg. So a
large positive hindsight_gap just means the player outperformed his price —
it is not a forward-looking target list.

What IS structurally valid: POSITIONAL spending shares. The league's tendency
to underspend on TEs and overspend on RBs is stable year-over-year and does
not depend on individual outcomes. That is the headline output here.

Run:  python src/market.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from valuation import compute_values

DRAFT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "drafts")

# ---------------------------------------------------------------------------
# Name normalization — handles OCR artifacts and common display variants.
# Applied to both sides of the join so mismatches are minimized.
# ---------------------------------------------------------------------------

_MANUAL_MAP = {
    # OCR artifacts from 2025 draft data
    "brocbowers": "Brock Bowers",
    "breece h": "Breece Hall",
    # Display name variants: nflverse capitalizations differ from Yahoo
    "de'von achane": "De'Von Achane",
    "amon-ra st. brown": "Amon-Ra St. Brown",
    "a.j. brown": "A.J. Brown",
    "d'andre swift": "D'Andre Swift",
    "c.j. stroud": "C.J. Stroud",
    "t.j. hockenson": "T.J. Hockenson",
    "j.k. dobbins": "J.K. Dobbins",
    "j.j. mccarthy": "J.J. McCarthy",
    "sam laporta": "Sam LaPorta",
}


import re as _re

_SUFFIX_PAT = _re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv)\s*$", _re.IGNORECASE)


def _normalize(name: str) -> str:
    """
    Returns a lowercase join key.

    1. Lowercase and strip.
    2. Apply any OCR / capitalization overrides from _MANUAL_MAP.
    3. Strip generational suffixes (Jr/Sr/II/III) — nflverse drops them,
       Yahoo draft data keeps them.

    Always returns lowercase so both sides of the join are comparable.
    """
    raw = name.lower().strip()
    # Apply manual map first (handles OCR artifacts)
    corrected = _MANUAL_MAP.get(raw, raw)
    # Strip suffix from whatever we have now
    key = _SUFFIX_PAT.sub("", corrected.lower()).strip()
    return key


def load_draft(season: int) -> pd.DataFrame:
    """
    Load a single draft CSV and return only skill-position rows.
    Kickers and DEF are not priced by the model, so they're excluded.
    """
    path = os.path.join(DRAFT_DIR, f"draft_{season}.csv")
    df = pd.read_csv(path)
    skill = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    skill["player_norm"] = skill["player"].apply(_normalize)
    skill["draft_season"] = season
    return skill.reset_index(drop=True)


def load_all_drafts() -> pd.DataFrame:
    """Load all available draft CSVs and concatenate."""
    parts = []
    for fname in sorted(os.listdir(DRAFT_DIR)):
        if fname.startswith("draft_") and fname.endswith(".csv"):
            season = int(fname.replace("draft_", "").replace(".csv", ""))
            parts.append(load_draft(season))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_market_board(proj_fn=None) -> pd.DataFrame:
    """
    Join the valuation board to draft history and compute value gaps.

    Returns the valuation board with three new columns:
      market_price   — what the league paid most recently for this player
      value_gap      — model_price - market_price  (positive = underpriced)
      value_ratio    — model_price / market_price  (> 1 = underpriced)

    Players not found in draft history get NaN for market columns; they are
    rookies, recently retired returners, or name-match failures. Both cases
    are reported explicitly.
    """
    board = compute_values(proj_fn=proj_fn)
    drafts = load_all_drafts()

    # Use only the most recent draft appearance per player.
    # If a player appeared in both 2024 and 2025, take 2025 price.
    latest = (
        drafts.sort_values("draft_season")
        .groupby("player_norm", as_index=False)
        .last()[["player_norm", "salary", "draft_season"]]
        .rename(columns={"salary": "market_price", "draft_season": "market_year"})
    )

    # Normalize the board names for joining
    board["player_norm"] = board["player_display_name"].apply(_normalize)

    board = board.merge(latest, on="player_norm", how="left")
    board = board.drop(columns=["player_norm"])

    # hindsight_gap = model_price − market_price.
    # Named "hindsight" because the model uses 2025 actual ppg while the
    # market prices were set before the 2025 season. Player-level gaps
    # reflect who outperformed, not who is structurally underpriced.
    board["hindsight_gap"] = board["price"] - board["market_price"]
    board["hindsight_ratio"] = board["price"] / board["market_price"]

    return board


def positional_market_shares(drafts: pd.DataFrame) -> pd.DataFrame:
    """
    For each draft year, compute each position's share of total dollars spent
    on skill positions (QB+RB+WR+TE only — DEF/K excluded).

    Returns a DataFrame indexed by position with columns for each draft year
    and a 'avg' column averaged across years.
    """
    skill = drafts[drafts["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    years = sorted(skill["draft_season"].unique())
    rows = {}
    for pos in ("QB", "RB", "WR", "TE"):
        row = {}
        for yr in years:
            yr_total = skill[skill["draft_season"] == yr]["salary"].sum()
            pos_total = skill[
                (skill["draft_season"] == yr) & (skill["position"] == pos)
            ]["salary"].sum()
            row[yr] = pos_total / yr_total if yr_total > 0 else 0.0
        row["avg"] = sum(row[yr] for yr in years) / len(years)
        rows[pos] = row
    return pd.DataFrame(rows).T


def _unmatched_expensive(drafts: pd.DataFrame, board: pd.DataFrame,
                         min_salary: int = 10) -> pd.DataFrame:
    """
    Return draft picks at or above min_salary that have no market_price match
    on the model board. These are the gaps where a missing price costs us signal.
    """
    board_norm = set(board["player_display_name"].apply(_normalize))
    skill = drafts[drafts["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    unmatched = skill[
        ~skill["player_norm"].isin(board_norm) &
        (skill["salary"] >= min_salary)
    ]
    # Deduplicate: if a player appears in both years, keep the most expensive
    return (
        unmatched.sort_values("salary", ascending=False)
        .drop_duplicates(subset="player_norm")
        [["player", "position", "salary", "draft_season"]]
        .rename(columns={"salary": "paid", "draft_season": "year"})
        .reset_index(drop=True)
    )


def main() -> None:
    board = build_market_board()
    drafts = load_all_drafts()

    # ── HEADLINE: Positional spending share vs. model ────────────────────────
    shares = positional_market_shares(drafts)
    years = sorted(drafts["draft_season"].unique())

    # Model share: fraction of offensive player dollars (DEF excluded from both sides).
    # Market denominator is skill-only dollars; use the same base for the model.
    drafted_off = board[board["is_drafted"] & (board["position"] != "DEF")]
    model_totals = drafted_off.groupby("position")["price"].sum()
    model_skill_pool = model_totals.sum()   # sum of QB+RB+WR+TE model prices

    print("=" * 72)
    print("POSITIONAL SPENDING: MARKET SHARE vs. MODEL SHARE")
    print("=" * 72)
    print("(Only structurally valid signal here — not contaminated by individual outcomes)")
    print("(Both denominators = skill-position dollars only; DEF excluded)\n")
    header = f"{'Pos':<5}"
    for yr in years:
        header += f"  {yr} Mkt"
    header += "  AvgMkt  ModelSh  Gap(Mkt-Mdl)"
    print(header)
    print("─" * 65)

    for pos in ("QB", "RB", "WR", "TE"):
        line = f"{pos:<5}"
        for yr in years:
            line += f"  {shares.loc[pos, yr] * 100:>5.1f}%"
        avg_mkt = shares.loc[pos, "avg"]
        model_sh = model_totals.get(pos, 0) / model_skill_pool
        gap = avg_mkt - model_sh
        line += f"  {avg_mkt * 100:>5.1f}%  {model_sh * 100:>5.1f}%  {gap * 100:>+6.1f}%"
        print(line)

    print()
    print("  Negative gap = market UNDERSPENDS vs. model (opportunity)")
    print("  Positive gap = market OVERSPENDS vs. model (avoid or fade)")

    # ── Unmatched expensive players (the only player-level analysis worth showing) ──
    print("\n" + "─" * 72)
    print("\nUNMATCHED PLAYERS PAID $10+  (missing market price = lost signal)")
    print("(These are the name-match failures that actually matter)")
    board_copy = board.copy()
    expensive_unmatched = _unmatched_expensive(drafts, board_copy)
    if expensive_unmatched.empty:
        print("  None — all $10+ players matched successfully.")
    else:
        print(expensive_unmatched.to_string(index=False))
        print(
            f"\n  {len(expensive_unmatched)} player(s) missing. Add manual name maps to "
            "_MANUAL_MAP in market.py or fix the draft CSV spelling."
        )

    # ── hindsight_gap reference (labeled clearly as contaminated) ────────────
    print("\n" + "─" * 72)
    print("\nHINDSIGHT GAPS — for reference only, NOT a target list")
    print("(model uses 2025 actual ppg; market prices were set before 2025 season)")
    has_market = board[board["is_drafted"] & board["market_price"].notna() & (board["position"] != "DEF")]
    print(
        has_market.nlargest(10, "hindsight_gap")[
            ["player_display_name", "position", "price", "market_price", "hindsight_gap"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
