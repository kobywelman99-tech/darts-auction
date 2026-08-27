#!/usr/bin/env python3
"""
Mock draft harness — src/mock_draft.py

Simulates a full 10-team, $200-cap DARTS auction.

ME ("King")
  Base ceiling: DraftState three-ceiling framework (feasibility + opp_cost).
  opp_cost uses a per-position $/PAR rate: scarce positions (TE, QB) get a
  higher rate than deep ones (WR, RB) via scarcity_factor = avg_rank / pos_rank.
  Starters: ceiling × 1.20 × scarcity_mult. FLEX: × 1.10 × scarcity_mult.
  Bench bids: cap at 50% of model price. Never bids for a full position.

OPPONENTS (9 teams)
  Calibrated from 2024-2025 DARTS actual draft data. Real league front-loads:
  ~62% of total money in the first 30 nominations; most teams near-broke by
  pick 60; $1-3 clears are common after pick 90.

  Bid formula: model_price × Normal(1.0, σ=0.28) × need_factor.
    need_factor: 1.20 = starter need, 0.72 = flex need, 0.25 = bench need, 0 = full.

  Calibration targets (vs. 2024-2025 actuals):
    nominations 1-20:  avg clearing ~$43-46   (top players, heavy competition)
    nominations 21-40: avg clearing ~$29-31   (good starters, teams still flush)
    nominations 41-60: avg clearing ~$13-15   (depth, wallets thinning)
    nominations 61-80: avg clearing ~$5-7     (bench, many teams near-broke)
    nominations 81+:   avg clearing ~$1-2     (stragglers, near-universal $1)

  The starter factor (1.20) and wider noise (σ=0.28) make opponents overbid on
  top picks, depleting their budgets. max_bid then naturally caps late bids at
  $1-3. This is the key mechanism that separates early from late clearing prices.

AUCTION MECHANISM
  English auction simulation: highest bidder wins at (runner-up + $1).
  If nobody bids, player goes to a random team with roster space at $1.
  Ties broken randomly.

NOMINATION ORDER
  Model price descending (most expensive first). DEF always last.

Usage:
    python src/mock_draft.py                  # single run, seed 2026
    python src/mock_draft.py --seed 42        # single run
    python src/mock_draft.py --runs 50        # distribution across 50 seeds
    python src/mock_draft.py --verbose        # full pick-by-pick log
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from league_config import (
    MIN_BID, BUDGET_PER_TEAM, ROSTER_SIZE,
    STARTERS, FLEX_ELIGIBLE,
)
from valuation import compute_values
from draft_state import DraftState

MY_TEAM = "King"
OPPONENT_NAMES = [
    "Chasing Mason", "MERDY", "The Ra-volution", "Team Taruchus",
    "Field Njigbas", "Justin's Team", "My Nix Hurts",
    "Itty Bitty Pitts", "Who needs a qb?",
]
ALL_TEAMS = [MY_TEAM] + OPPONENT_NAMES


# ---------------------------------------------------------------------------
# Auction primitives
# ---------------------------------------------------------------------------

def _run_auction(
    bids: dict[str, int], rng: np.random.Generator
) -> tuple[str, int]:
    """
    English auction: winner pays runner-up_max + $1.
    Returns (winner_name, clearing_price). Ties broken randomly.
    """
    valid = [(name, amt) for name, amt in bids.items() if amt >= MIN_BID]
    if not valid:
        return ("", MIN_BID)
    # Shuffle before sort so ties are random (stable sort preserves insertion order)
    order = rng.permutation(len(valid))
    valid = [valid[i] for i in order]
    valid.sort(key=lambda x: x[1], reverse=True)
    winner, winner_max = valid[0]
    runner_up = valid[1][1] if len(valid) > 1 else MIN_BID - 1
    price = min(winner_max, max(MIN_BID, runner_up + 1))
    return winner, price


def _opponent_bid(
    ts, player_pos: str, model_price: int, rng: np.random.Generator
) -> int:
    """
    Noisy model-price bid scaled by positional need, capped at feasibility.

    Calibrated to real DARTS draft data: starter factor 1.20 + σ=0.28 causes
    overbidding on early top picks → fast budget depletion → cheap late picks.
    See module docstring for calibration targets.
    """
    need = ts.positional_need(player_pos)
    if need == "none":
        return 0
    raw = model_price * float(rng.normal(1.0, 0.28))
    raw = max(float(MIN_BID), raw)
    factor = {"starter": 1.20, "flex": 0.72, "bench": 0.25}.get(need, 0.0)
    bid = int(round(raw * factor))
    return max(0, min(bid, ts.max_bid))


def _scarcity_mult(state: DraftState, pos: str) -> float:
    """
    Urgency multiplier: rises when demand begins to outstrip supply at a position.

    ratio = teams_needing_starter_or_flex / above-replacement players still available

    ratio ≤ 0.50 → 1.00×  (ample supply, no urgency premium)
    ratio = 1.00 → 1.15×  (roughly one player per team, moderate urgency)
    ratio ≥ 2.00 → 1.50×  (capped — don't chase past 50% above ceiling)

    This is computed fresh each pick from the live board state, so it rises
    naturally as the draft empties out a position without needing a schedule.
    """
    drafted_lower = {p.player.lower() for p in state.all_picks}
    n_avail = state.board[
        (state.board["position"] == pos)
        & (state.board["par"] > 0)
        & state.board["is_drafted"]
        & ~state.board["player_display_name"].str.lower().isin(drafted_lower)
    ].shape[0]

    n_need = sum(
        1 for ts in state.teams.values()
        if ts.starter_need(pos) or (pos in FLEX_ELIGIBLE and ts.flex_open)
    )

    if n_avail <= 0:
        return 1.50
    ratio = n_need / n_avail
    # No premium when ratio < 0.5; rises smoothly from there.
    return min(1.50, 1.0 + 0.30 * max(0.0, ratio - 0.5))


# ---------------------------------------------------------------------------
# Roster slot assignment (for reporting)
# ---------------------------------------------------------------------------

def _optimal_lineup(picks: list) -> dict[str, list]:
    """
    Assign picks to roster slots to maximise starter PAR.

    Skill slot assignment (WR×3 + RB×2 + TE×1 + FLEX×1):
      1. Fill each dedicated slot (WR, RB, TE) with the best eligible player.
      2. FLEX gets the best remaining WR/RB/TE not already in a dedicated slot.
      3. If step 2 would leave FLEX empty (all skill players absorbed by dedicated
         slots), move the lowest-PAR dedicated player to FLEX instead. That player's
         position then covers its dedicated slot via the FLEX, and the dedicated slot
         is recorded as "covered" in scoring rather than a shutout.

    QB and DEF never compete with skill slots.
    Bench = everything not placed above.
    """
    by_par = sorted(picks, key=lambda p: p.par, reverse=True)

    # Separate by category
    qbs   = [p for p in by_par if p.position == "QB"]
    defs  = [p for p in by_par if p.position == "DEF"]
    skill = [p for p in by_par if p.position in FLEX_ELIGIBLE]

    # Fill dedicated skill slots (greedy by PAR within each position)
    wrs = [p for p in skill if p.position == "WR"]
    rbs = [p for p in skill if p.position == "RB"]
    tes = [p for p in skill if p.position == "TE"]
    wr_ded = wrs[:STARTERS["WR"]]
    rb_ded = rbs[:STARTERS["RB"]]
    te_ded = tes[:STARTERS["TE"]]

    dedicated_ids = {id(p) for p in wr_ded + rb_ded + te_ded}
    remaining = [p for p in skill if id(p) not in dedicated_ids]

    if remaining:
        # Normal case: FLEX gets the best leftover skill player
        flex = [remaining[0]]
        bench_skill = remaining[1:]
    else:
        # Edge case: every skill player was absorbed by dedicated slots → FLEX empty.
        # Move the lowest-PAR dedicated player to FLEX so that slot is filled.
        # That player's position "covers" its dedicated slot (scored via effective counts).
        all_ded = sorted(wr_ded + rb_ded + te_ded, key=lambda p: p.par)
        if all_ded:
            victim = all_ded[0]
            if victim.position == "WR":
                wr_ded = [p for p in wr_ded if id(p) != id(victim)]
            elif victim.position == "RB":
                rb_ded = [p for p in rb_ded if id(p) != id(victim)]
            else:
                te_ded = [p for p in te_ded if id(p) != id(victim)]
            flex = [victim]
        else:
            flex = []
        bench_skill = []

    slots: dict[str, list] = {pos: [] for pos in STARTERS}
    slots["BENCH"] = []
    slots["QB"]   = qbs[:STARTERS["QB"]]
    slots["DEF"]  = defs[:STARTERS["DEF"]]
    slots["WR"]   = wr_ded
    slots["RB"]   = rb_ded
    slots["TE"]   = te_ded
    slots["FLEX"] = flex
    slots["BENCH"] = (
        qbs[STARTERS["QB"]:]
        + defs[STARTERS["DEF"]:]
        + bench_skill
    )
    return slots


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate(
    board: pd.DataFrame, seed: int
) -> tuple[DraftState, list[dict]]:
    """
    Run a full 150-pick auction simulation.

    Returns the final DraftState (all teams' rosters) and an auction_log
    (one dict per player showing bids, clearing price, and outcome flags).
    """
    rng = np.random.default_rng(seed)
    state = DraftState(my_team=MY_TEAM, all_teams=ALL_TEAMS, board=board)

    # Nomination order: offensive by model price desc; DEF last.
    off = board[board.is_drafted & (board.position != "DEF")].sort_values(
        ["price", "par"], ascending=[False, False]
    )
    defs = board[board.is_drafted & (board.position == "DEF")]

    # ── DEF: pre-assign one to each team at $1 ──────────────────────────────
    # Each team reserves 1 roster slot for DEF before the offensive auction.
    # This prevents all 15 slots filling with offensive players first.
    def_team_order = ALL_TEAMS[:]
    rng.shuffle(def_team_order)
    for i, (_, dr) in enumerate(defs.iterrows()):
        if i < len(def_team_order):
            state.add_pick(
                str(dr["player_display_name"]), MIN_BID,
                def_team_order[i], position="DEF",
            )

    auction_log: list[dict] = []

    # ── Offensive players ────────────────────────────────────────────────────
    for _, row in off.iterrows():
        player     = str(row["player_display_name"])
        pos        = str(row["position"])
        model_px   = int(row["price"])
        par        = float(row["par"])
        ppg        = float(row["proj_ppg"]) if pd.notna(row["proj_ppg"]) else 0.0

        me      = state.my_state
        my_need = me.positional_need(pos)
        # Capture budget state BEFORE add_pick so the log reflects the decision
        # context, not the post-pick state.
        my_money_pre  = me.money_remaining
        my_maxbid_pre = me.max_bid
        my_slots_pre  = me.slots_remaining

        # My bid. Starters: 1.20× opp_cost ceiling × positional scarcity_mult.
        # opp_cost itself already adjusts for positional scarcity via per-position
        # $/PAR rate in draft_state._my_positional_rate().
        if my_need == "none":
            my_bid     = 0
            my_ceiling = 0
        else:
            c         = state.my_ceilings(player)
            base_ceil = int(c["recommended"])

            if my_need == "starter":
                slot_factor = 1.20
                sc = _scarcity_mult(state, pos)
                my_bid     = min(me.max_bid, max(MIN_BID, int(base_ceil * slot_factor * sc)))
                my_ceiling = my_bid
            elif my_need == "flex":
                slot_factor = 1.10
                sc = _scarcity_mult(state, pos)
                my_bid     = min(me.max_bid, max(MIN_BID, int(base_ceil * slot_factor * sc)))
                my_ceiling = my_bid
            else:   # bench — heavy discount; starter value doesn't apply
                my_bid     = min(base_ceil, max(MIN_BID, model_px // 2))
                my_ceiling = my_bid

        # Opponent bids
        bids: dict[str, int] = {}
        if my_bid >= MIN_BID:
            bids[MY_TEAM] = my_bid
        for name, ts in state.teams.items():
            if name == MY_TEAM:
                continue
            ob = _opponent_bid(ts, pos, model_px, rng)
            if ob >= MIN_BID:
                bids[name] = ob

        # Run auction (or assign at $1 if nobody bid)
        if bids:
            winner, price = _run_auction(bids, rng)
        else:
            needy = [n for n, ts in state.teams.items()
                     if ts.positional_need(pos) != "none"]
            if not needy:
                # Truly nobody needs this player — skip
                auction_log.append({
                    "player": player, "pos": pos, "model_price": model_px,
                    "par": par, "ppg": ppg, "winner": "UNDRAFTED", "price": 0,
                    "my_bid": 0, "my_need": my_need, "my_ceiling": 0,
                    "i_won": False, "passed": False, "could_have_had": False,
                    "my_money_remaining": my_money_pre,
                    "my_max_bid": my_maxbid_pre,
                    "my_slots_remaining": my_slots_pre,
                })
                continue
            winner = str(rng.choice(needy))
            price  = MIN_BID

        state.add_pick(player, price, winner, position=pos)

        i_won      = winner == MY_TEAM
        # "passed" = I wanted the player, I bid, but the market cleared above my max
        passed     = (not i_won) and (my_bid >= MIN_BID) and (price > my_bid)
        # "could_have_had" = I wanted the player, I bid my ceiling, but lost a close race
        could      = (not i_won) and (my_bid >= MIN_BID) and (price <= my_bid)

        auction_log.append({
            "player": player, "pos": pos, "model_price": model_px,
            "par": par, "ppg": ppg, "winner": winner, "price": price,
            "my_bid": my_bid, "my_need": my_need, "my_ceiling": my_ceiling,
            "i_won": i_won, "passed": passed, "could_have_had": could,
            "my_money_remaining": my_money_pre,
            "my_max_bid": my_maxbid_pre,
            "my_slots_remaining": my_slots_pre,
        })

    return state, auction_log


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(
    state: DraftState, auction_log: list[dict], board: pd.DataFrame
) -> None:
    me  = state.my_state
    W   = 74

    # Index ppg by player name for display
    ppg_map: dict[str, float] = {}
    for e in auction_log:
        ppg_map[e["player"]] = e["ppg"]

    print()
    print("=" * W)
    print(f"  MOCK DRAFT FINAL REPORT — {MY_TEAM}")
    print(f"  ${me.money_spent} spent  |  ${me.money_remaining} remaining")
    print("=" * W)

    # ── My full roster ───────────────────────────────────────────────────────
    lineup = _optimal_lineup(me.picks)

    print("\nMY ROSTER\n")
    print(f"  {'Slot':<7} {'Player':<28} {'Pos':>3}  {'$':>3}  {'ppg':>5}  {'PAR':>6}")
    print("  " + "─" * 60)

    starter_par = 0.0
    for slot in ("QB", "RB", "WR", "TE", "FLEX", "DEF", "BENCH"):
        players = lineup.get(slot, [])
        cap     = STARTERS.get(slot, 5) if slot != "BENCH" else 5
        for pick in players:
            ppg_s = f"{ppg_map.get(pick.player, 0):.1f}"
            par_s = f"{pick.par:.0f}"
            print(f"  {slot:<7} {pick.player:<28} {pick.position:>3}  "
                  f"${pick.salary:>2}  {ppg_s:>5}  {par_s:>6}")
            if slot not in ("BENCH", "DEF"):
                starter_par += pick.par
        for _ in range(cap - len(players)):
            print(f"  {slot:<7} {'--- EMPTY ---':<28} {'':>3}  {'':>3}  {'':>5}  {'':>6}")

    total_par = me.total_par
    print("  " + "─" * 60)
    print(f"\n  Starter PAR:  {starter_par:.0f}")
    print(f"  Total PAR:    {total_par:.0f}")
    print(f"  Money left:   ${me.money_remaining}")

    # ── Leaguewide PAR ranking ───────────────────────────────────────────────
    print()
    print("─" * W)
    print("\nLEAGUEWIDE STANDINGS BY STARTER PAR\n")
    print(f"  {'Rk':>2}  {'Team':<30}  {'Spent':>5}  {'Starters':>9}  {'TotalPAR':>9}")
    print("  " + "─" * 62)

    team_rows = []
    for name, ts in state.teams.items():
        lu = _optimal_lineup(ts.picks)
        sp = sum(p.par for slot, picks in lu.items()
                 for p in picks if slot not in ("BENCH",))
        tp = ts.total_par
        team_rows.append((name, ts.money_spent, sp, tp))

    team_rows.sort(key=lambda x: x[2], reverse=True)
    for rk, (name, spent, sp, tp) in enumerate(team_rows, 1):
        marker = "  ← YOU" if name == MY_TEAM else ""
        print(f"  {rk:>2}. {name:<30}  ${spent:>4}  {sp:>9.0f}  {tp:>9.0f}{marker}")

    # ── My bargains ──────────────────────────────────────────────────────────
    my_wins = [e for e in auction_log if e["i_won"]]
    bargains = sorted(
        [e for e in my_wins if e["price"] < e["model_price"] - 3],
        key=lambda x: x["model_price"] - x["price"], reverse=True,
    )
    if bargains:
        print()
        print("─" * W)
        print("\nMY BARGAINS  (won ≥ $4 below model price)\n")
        print(f"  {'Player':<28} {'Pos':>3}  {'Model':>6}  {'Paid':>5}  {'Saved':>6}  {'PAR':>6}")
        print("  " + "─" * 58)
        for e in bargains:
            saved = e["model_price"] - e["price"]
            print(f"  {e['player']:<28} {e['pos']:>3}  "
                  f"${e['model_price']:>5}  ${e['price']:>4}  ${saved:>5}  {e['par']:>6.0f}")

    # ── My overpays ──────────────────────────────────────────────────────────
    overpays = sorted(
        [e for e in my_wins if e["price"] > e["model_price"] + 3],
        key=lambda x: x["price"] - x["model_price"], reverse=True,
    )
    if overpays:
        print()
        print("─" * W)
        print("\nMY OVERPAYS  (paid ≥ $4 above model price)\n")
        print(f"  {'Player':<28} {'Pos':>3}  {'Model':>6}  {'Paid':>5}  {'Over':>5}  {'PAR':>6}")
        print("  " + "─" * 58)
        for e in overpays:
            over = e["price"] - e["model_price"]
            print(f"  {e['player']:<28} {e['pos']:>3}  "
                  f"${e['model_price']:>5}  ${e['price']:>4}  +${over:>4}  {e['par']:>6.0f}")

    # ── Priced out ───────────────────────────────────────────────────────────
    priced_out = sorted(
        [e for e in auction_log if e["passed"] and e["par"] >= 25],
        key=lambda x: x["par"], reverse=True,
    )
    if priced_out:
        print()
        print("─" * W)
        print("\nPRICED OUT  (I bid, market beat me; PAR ≥ 25 — biggest misses)\n")
        print(f"  {'Player':<28} {'Pos':>3}  {'Model':>6}  {'Cleared':>8}  {'MyCeil':>7}  {'PAR':>6}  Winner")
        print("  " + "─" * 72)
        for e in priced_out[:10]:
            gap = e["price"] - e["my_ceiling"]
            short = e["winner"][:18] if len(e["winner"]) > 18 else e["winner"]
            print(f"  {e['player']:<28} {e['pos']:>3}  "
                  f"${e['model_price']:>5}  ${e['price']:>7}  ${e['my_ceiling']:>6}  "
                  f"{e['par']:>6.0f}  {short} (+${gap})")

    # ── Could-have-hads ──────────────────────────────────────────────────────
    could_have = sorted(
        [e for e in auction_log if e["could_have_had"] and e["par"] >= 20],
        key=lambda x: x["par"], reverse=True,
    )
    if could_have:
        print()
        print("─" * W)
        print("\nNEAR MISSES  (my bid ≥ clearing, lost to higher bidder — same ceiling, bad luck)\n")
        print(f"  {'Player':<28} {'Pos':>3}  {'Cleared':>8}  {'MyBid':>6}  {'PAR':>6}  Winner")
        print("  " + "─" * 65)
        for e in could_have[:8]:
            short = e["winner"][:18]
            print(f"  {e['player']:<28} {e['pos']:>3}  "
                  f"${e['price']:>7}  ${e['my_bid']:>5}  {e['par']:>6.0f}  {short}")

    # ── Positional summary and second-guess flags ────────────────────────────
    print()
    print("─" * W)
    print("\nPOSITIONAL ANALYSIS  (where to second-guess)\n")

    pc = me.position_counts
    for pos in ("QB", "RB", "WR", "TE"):
        drafted = pc.get(pos, 0)
        needed  = STARTERS.get(pos, 0)
        my_pos_picks = [p for p in me.picks if p.position == pos]
        avg_ppg = (
            sum(ppg_map.get(p.player, 0) for p in my_pos_picks) / len(my_pos_picks)
            if my_pos_picks else 0.0
        )
        avg_par = (
            sum(p.par for p in my_pos_picks) / len(my_pos_picks)
            if my_pos_picks else 0.0
        )
        # League avg at this position
        all_pos_picks = [
            p for ts in state.teams.values()
            for p in ts.picks if p.position == pos
        ]
        league_avg_par = (
            sum(p.par for p in all_pos_picks) / len(all_pos_picks)
            if all_pos_picks else 0.0
        )
        delta = avg_par - league_avg_par
        status = "✓" if drafted >= needed else f"⚠ ONLY {drafted}/{needed}"
        flag = (
            "  ← WEAK vs league" if delta < -15
            else "  ← STRONG vs league" if delta > 15
            else ""
        )
        print(f"  {pos}: {drafted} drafted  avg {avg_ppg:.1f} ppg  "
              f"avg PAR {avg_par:.0f}  (league avg {league_avg_par:.0f}, "
              f"{'+' if delta >= 0 else ''}{delta:.0f})  {status}{flag}")

    # Money utilisation
    print()
    disc_spent = me.money_spent - len(me.picks) * MIN_BID
    print(f"  Discretionary spent: ${disc_spent}  "
          f"(${me.money_remaining} unspent — {'good' if me.money_remaining <= 5 else 'left on table'})")

    print()


# ---------------------------------------------------------------------------
# Full auction log (verbose mode)
# ---------------------------------------------------------------------------

def _print_auction_log(auction_log: list[dict]) -> None:
    print("\nFULL AUCTION LOG\n")
    print(f"  {'#':>3}  {'Player':<26}  {'Pos':>3}  {'Model':>5}  "
          f"{'Paid':>4}  {'Winner':<22}  {'MyBid':>5}  Notes")
    print("  " + "─" * 80)
    for i, e in enumerate(auction_log, 1):
        if e["winner"] == "UNDRAFTED":
            continue
        note = ""
        if e["i_won"] and e["price"] < e["model_price"] - 3:
            note = f"BARGAIN (saved ${e['model_price'] - e['price']})"
        elif e["i_won"] and e["price"] > e["model_price"] + 3:
            note = f"OVERPAY (+${e['price'] - e['model_price']})"
        elif e["i_won"]:
            note = "won"
        elif e["passed"]:
            note = f"priced out (+${e['price'] - e['my_ceiling']} above ceil)"
        elif e["could_have_had"]:
            note = "near miss"
        elif e["my_need"] == "none":
            note = "—"
        winner_short = e["winner"][:22] if len(e["winner"]) > 22 else e["winner"]
        my_bid_str = f"${e['my_bid']}" if e["my_bid"] else " pass"
        print(f"  {i:>3}. {e['player']:<26}  {e['pos']:>3}  "
              f"${e['model_price']:>4}  ${e['price']:>3}  {winner_short:<22}  "
              f"{my_bid_str:>5}  {note}")


# ---------------------------------------------------------------------------
# Per-run statistics (used both by single-run report and distribution)
# ---------------------------------------------------------------------------

def _run_stats(state: DraftState) -> dict:
    """Compact summary of a completed draft for distribution analysis."""
    me     = state.my_state
    lineup = _optimal_lineup(me.picks)

    starter_par = sum(
        p.par for slot, picks in lineup.items()
        for p in picks if slot not in ("BENCH",)
    )

    # Effective starter counts.
    # FLEX player's position counts toward that position's requirement:
    # a WR in FLEX covers WR3, an RB in FLEX covers RB2, etc.
    flex_picks = lineup.get("FLEX", [])
    flex_pos = flex_picks[0].position if flex_picks else None
    effective = {
        "WR": len(lineup.get("WR", [])) + (1 if flex_pos == "WR" else 0),
        "RB": len(lineup.get("RB", [])) + (1 if flex_pos == "RB" else 0),
        "TE": len(lineup.get("TE", [])) + (1 if flex_pos == "TE" else 0),
    }

    # Unfilled starter slots (using effective counts for skill positions)
    unfilled: list[str] = []
    for slot in ("QB", "DEF"):
        cap    = STARTERS.get(slot, 0)
        filled = len(lineup.get(slot, []))
        unfilled.extend([slot] * (cap - filled))
    for pos in ("WR", "RB", "TE"):
        shortage = max(0, STARTERS[pos] - effective[pos])
        unfilled.extend([pos] * shortage)
    # FLEX shortage: FLEX is only unfilled if we have no skill player for it at all
    if not flex_picks:
        unfilled.append("FLEX")

    # League rank by starter PAR
    team_sp: dict[str, float] = {}
    for name, ts in state.teams.items():
        lu = _optimal_lineup(ts.picks)
        team_sp[name] = sum(
            p.par for sl, pk in lu.items() for p in pk if sl not in ("BENCH",)
        )
    rank = sorted(team_sp.values(), reverse=True).index(team_sp[MY_TEAM]) + 1

    pc = me.position_counts
    return {
        "starter_par":    starter_par,
        "total_par":      me.total_par,
        "rank":           rank,
        "money_left":     me.money_remaining,
        "unfilled":       unfilled,
        "all_filled":     len(unfilled) == 0,
        "pos_counts":     {p: pc.get(p, 0) for p in ("QB", "RB", "WR", "TE")},
        "bench_count":    len(lineup.get("BENCH", [])),
    }


# ---------------------------------------------------------------------------
# 50-seed distribution
# ---------------------------------------------------------------------------

def run_distribution(board: pd.DataFrame, n_runs: int = 50) -> None:
    import statistics as _stats

    results: list[dict] = []
    all_clearing: list[tuple[int, int]] = []  # (pick_idx_1based, price)
    print(f"Running {n_runs} simulated drafts…")
    for seed in range(n_runs):
        state, log = simulate(board, seed)
        s = _run_stats(state)
        s["seed"] = seed
        results.append(s)
        off_log = [e for e in log if e["winner"] != "UNDRAFTED"]
        for i, e in enumerate(off_log, 1):
            all_clearing.append((i, e["price"]))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{n_runs}…")

    # ── Aggregate ────────────────────────────────────────────────────────────
    sp_vals   = [r["starter_par"] for r in results]
    rk_vals   = [r["rank"]        for r in results]
    mon_vals  = [r["money_left"]  for r in results]

    def pct(lst: list, p: float) -> float:
        s = sorted(lst)
        idx = max(0, int(len(s) * p / 100) - 1)
        return s[idx]

    W = 72
    print()
    print("=" * W)
    print(f"  {n_runs}-SEED DISTRIBUTION — starter-slot premium + scarcity bidding")
    print("=" * W)

    # ── Starter PAR ──────────────────────────────────────────────────────────
    print(f"\nSTARTER PAR (points above replacement from starting lineup)\n")
    print(f"  p10  {pct(sp_vals, 10):>7.0f}")
    print(f"  p25  {pct(sp_vals, 25):>7.0f}")
    print(f"  med  {_stats.median(sp_vals):>7.0f}")
    print(f"  p75  {pct(sp_vals, 75):>7.0f}")
    print(f"  p90  {pct(sp_vals, 90):>7.0f}")

    # ── League rank ──────────────────────────────────────────────────────────
    top5  = sum(1 for r in results if r["rank"] <= 5)
    top3  = sum(1 for r in results if r["rank"] <= 3)
    bot3  = sum(1 for r in results if r["rank"] >= 8)
    print(f"\nLEAGUE RANK  (1 = highest starter PAR; 10 = lowest)\n")
    print(f"  Median rank:            {round(_stats.median(rk_vals))}")
    print(f"  Finished top-5 (≥ avg): {top5}/{n_runs}  ({100 * top5 // n_runs}%)")
    print(f"  Finished top-3:         {top3}/{n_runs}  ({100 * top3 // n_runs}%)")
    print(f"  Finished bottom-3:      {bot3}/{n_runs}  ({100 * bot3 // n_runs}%)")
    rank_dist = [0] * 11
    for r in results:
        rank_dist[r["rank"]] += 1
    print(f"\n  Rank distribution:")
    print(f"  {'Rank':<8}", end="")
    for rk in range(1, 11):
        print(f" {rk:>4}", end="")
    print()
    print(f"  {'Count':<8}", end="")
    for rk in range(1, 11):
        print(f" {rank_dist[rk]:>4}", end="")
    print()

    # ── Roster completion ────────────────────────────────────────────────────
    all_filled = sum(1 for r in results if r["all_filled"])
    print(f"\nROSTER COMPLETION\n")
    print(f"  All starting slots filled:  {all_filled}/{n_runs}  "
          f"({100 * all_filled // n_runs}%)")

    # Count shutouts per slot
    slot_short: dict[str, int] = {
        slot: sum(1 for r in results if slot in r["unfilled"])
        for slot in ("QB", "RB", "WR", "TE", "FLEX", "DEF")
    }
    print(f"\n  Slots left empty (any run):")
    for slot, ct in sorted(slot_short.items(), key=lambda x: -x[1]):
        if ct > 0:
            bar = "█" * ct + "░" * (n_runs - ct)
            print(f"    {slot:<5}  {ct:>2}/{n_runs}  ({100 * ct // n_runs:>3}%)  {bar[:30]}")

    # Per-position shortage counts, using effective scoring
    # (WR in FLEX counts as WR3; RB in FLEX counts as RB2; TE in FLEX counts as TE1)
    print(f"\n  QB starters short:  "
          f"{sum(1 for r in results if 'QB' in r['unfilled'])}/{n_runs}")
    print(f"  WR starters short:  "
          f"{sum(1 for r in results if 'WR' in r['unfilled'])}/{n_runs}  "
          f"(WR in FLEX counts as WR3)")
    print(f"  RB starters short:  "
          f"{sum(1 for r in results if 'RB' in r['unfilled'])}/{n_runs}  "
          f"(RB in FLEX counts as RB2)")
    print(f"  TE starters short:  "
          f"{sum(1 for r in results if 'TE' in r['unfilled'])}/{n_runs}  "
          f"(TE in FLEX counts as TE1)")

    # ── Money utilisation ────────────────────────────────────────────────────
    print(f"\nMONEY UTILISATION\n")
    print(f"  Median unspent:   ${round(_stats.median(mon_vals))}")
    print(f"  Mean unspent:     ${sum(mon_vals) / len(mon_vals):.1f}")
    print(f"  Left ≥ $5:        {sum(1 for m in mon_vals if m >= 5)}/{n_runs}  "
          f"({sum(1 for m in mon_vals if m >= 5) * 100 // n_runs}%)")
    print(f"  Left ≥ $20:       {sum(1 for m in mon_vals if m >= 20)}/{n_runs}  "
          f"({sum(1 for m in mon_vals if m >= 20) * 100 // n_runs}%)")

    # ── Market calibration (verify opponent model vs. real draft data) ────────
    print(f"\nMARKET CALIBRATION  (avg clearing price — sim vs. real 2024-2025 drafts)\n")
    print(f"  {'Pick range':<14}  {'Sim avg':>7}  {'Real target':>11}  {'Match?':>6}")
    print(f"  {'─'*50}")
    calibration_targets = [
        (1,   20, 43, 46),
        (21,  40, 29, 31),
        (41,  60, 13, 15),
        (61,  80,  5,  7),
        (81, 999,  1,  2),
    ]
    for lo, hi, real_lo, real_hi in calibration_targets:
        chunk = [p for (i, p) in all_clearing if lo <= i <= hi]
        if not chunk:
            continue
        avg = sum(chunk) / len(chunk)
        pct_dollar_3 = 100 * sum(1 for p in chunk if p <= 3) / len(chunk)
        match = "✓" if real_lo <= avg <= real_hi + 5 else "~" if avg <= real_hi + 12 else "✗ HIGH"
        label = f"{lo}-{min(hi,130)}" if hi < 999 else f"{lo}+"
        extra = f"  ({pct_dollar_3:.0f}% ≤$3)" if lo >= 61 else ""
        print(f"  picks {label:<9}  ${avg:>5.1f}    ${real_lo}-${real_hi}{extra:<20}  {match}")

    # ── Extremes ─────────────────────────────────────────────────────────────
    print(f"\nBEST SEEDS (starter PAR)")
    for r in sorted(results, key=lambda x: -x["starter_par"])[:3]:
        holes = ", ".join(r["unfilled"]) if r["unfilled"] else "none"
        print(f"  seed {r['seed']:>3}: PAR {r['starter_par']:.0f}  "
              f"rank {r['rank']}  ${r['money_left']} left  holes: {holes}")

    print(f"\nWORST SEEDS (starter PAR)")
    for r in sorted(results, key=lambda x: x["starter_par"])[:3]:
        holes = ", ".join(r["unfilled"]) if r["unfilled"] else "none"
        print(f"  seed {r['seed']:>3}: PAR {r['starter_par']:.0f}  "
              f"rank {r['rank']}  ${r['money_left']} left  holes: {holes}")

    print()


# ---------------------------------------------------------------------------
# TE shutout diagnosis
# ---------------------------------------------------------------------------

def diagnose_te_shutouts(board: pd.DataFrame, n_seeds: int = 50) -> None:
    """
    For each seed where King finishes with no TE, determine WHY.

    Bucket A — market priced King out.
        Every TE that cleared while King's TE slot was open cleared above
        King's max_bid at that moment. King couldn't have won regardless.
        Correct pass; handle by nominating early on draft night.

    Bucket B — sequencing failure.
        At least one TE cleared at or below King's max_bid while the slot
        was open, but King didn't win it. King had the money (or a meaningful
        bid capacity) but either bid too low (opp_cost formula underbid) or
        had already spent the budget on bench RBs/WRs and max_bid had
        collapsed to $1.

    Per-seed: which TEs cleared, what King's budget state was, and King's bid.
    Summary: A vs B count, plus the sub-pattern for B (opp_cost underbid vs
    budget exhaustion).
    """
    print(f"Running {n_seeds} simulations to diagnose TE shutouts…")

    shutouts: list[dict] = []
    for seed in range(n_seeds):
        state, log = simulate(board, seed)
        if state.my_state.position_counts.get("TE", 0) > 0:
            continue   # not a shutout

        # TE events where King still had a starter TE slot open
        open_slot = [
            e for e in log
            if e["pos"] == "TE"
            and e["winner"] != "UNDRAFTED"
            and e["my_need"] == "starter"
        ]
        # TE events where slot was gone because roster was full (slots_remaining == 0)
        locked_out = [
            e for e in log
            if e["pos"] == "TE"
            and e["winner"] != "UNDRAFTED"
            and e["my_need"] == "none"
            and e["my_slots_remaining"] == 0
        ]

        if not open_slot and locked_out:
            # Roster filled with non-TE players — never had a free slot for TE
            cheapest = min(locked_out, key=lambda x: x["price"])
            shutouts.append({
                "seed": seed, "bucket": "B", "subreason": "roster-full",
                "summary": f"roster filled with non-TEs; cheapest TE cleared at ${cheapest['price']}",
                "open_slot_events": [],
                "key_event": cheapest,
            })
            continue

        if not open_slot:
            shutouts.append({
                "seed": seed, "bucket": "?", "subreason": "unknown",
                "summary": "no TE events while slot was open",
                "open_slot_events": [], "key_event": {},
            })
            continue

        # Is there any TE that cleared at or below King's max_bid at that moment?
        could_afford = [e for e in open_slot if e["price"] <= e["my_max_bid"]]
        cheapest_overall = min(open_slot, key=lambda x: x["price"])

        if could_afford:
            # King had the money but didn't win — either opp_cost formula bid too low
            # or King spent down to $1 max_bid but could technically still bid $1
            best_opp = min(could_afford, key=lambda x: x["price"])
            if best_opp["my_bid"] >= MIN_BID and best_opp["price"] > best_opp["my_bid"]:
                subreason = "opp-cost-underbid"
            elif best_opp["my_bid"] < MIN_BID:
                subreason = "budget-collapse"   # max_bid technically ≥ price but bid=0
            else:
                subreason = "opp-cost-underbid"
            shutouts.append({
                "seed": seed, "bucket": "B", "subreason": subreason,
                "summary": (
                    f"{best_opp['player']} cleared ${best_opp['price']}  "
                    f"(my max_bid ${best_opp['my_max_bid']}, I bid ${best_opp['my_bid']}, "
                    f"${best_opp['my_money_remaining']} left)"
                ),
                "open_slot_events": open_slot,
                "key_event": best_opp,
            })
        else:
            # All TEs cleared above King's max_bid — genuinely priced out
            shutouts.append({
                "seed": seed, "bucket": "A", "subreason": "priced-out",
                "summary": (
                    f"cheapest TE was {cheapest_overall['player']} at "
                    f"${cheapest_overall['price']}  "
                    f"(my max_bid ${cheapest_overall['my_max_bid']}, "
                    f"${cheapest_overall['my_money_remaining']} left)"
                ),
                "open_slot_events": open_slot,
                "key_event": cheapest_overall,
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    bucket_a = [r for r in shutouts if r["bucket"] == "A"]
    bucket_b = [r for r in shutouts if r["bucket"] == "B"]

    W = 74
    print()
    print("=" * W)
    print(f"  TE SHUTOUT DIAGNOSIS — {len(shutouts)} shutout seeds out of {n_seeds}")
    print("=" * W)
    print(f"\n  Bucket A (market priced me out — correct pass): "
          f"{len(bucket_a)}/{len(shutouts)}")
    print(f"  Bucket B (sequencing failure — had money, lost TE): "
          f"{len(bucket_b)}/{len(shutouts)}")

    if bucket_b:
        sub_counts: dict[str, int] = {}
        for r in bucket_b:
            sub_counts[r["subreason"]] = sub_counts.get(r["subreason"], 0) + 1
        print(f"\n  Bucket B breakdown:")
        label_map = {
            "opp-cost-underbid": "opp_cost formula bid below clearing (had money, ceiling too low)",
            "budget-collapse":   "budget exhausted on other positions before TE came up",
            "roster-full":       "roster filled with non-TEs — no slot left for TE",
        }
        for sub, ct in sorted(sub_counts.items(), key=lambda x: -x[1]):
            print(f"    {ct}x  {label_map.get(sub, sub)}")

    # ── Bucket B detail ───────────────────────────────────────────────────────
    if bucket_b:
        print(f"\n{'─'*W}")
        print(f"\nBUCKET B  — per-seed detail\n")
        print(f"  {'Seed':>4}  {'Subreason':<20}  {'Cheapest TE (cleared)':<38}  My bid")
        print("  " + "─" * 70)
        for r in sorted(bucket_b, key=lambda x: x["seed"]):
            ke = r["key_event"]
            player_s = ke.get("player", "?")[:26]
            print(f"  {r['seed']:>4}  {r['subreason']:<20}  "
                  f"{player_s:<26} ${ke.get('price', 0):>3}  "
                  f"max_bid ${ke.get('my_max_bid', 0):>3}  "
                  f"I bid ${ke.get('my_bid', 0):>3}")
        # For each B seed, also show all TE events while slot was open
        print()
        for r in sorted(bucket_b, key=lambda x: x["seed"]):
            print(f"  Seed {r['seed']:>2} ({r['subreason']}) — ${r['key_event'].get('my_money_remaining',0)} remaining at cheapest TE pick")
            for e in r["open_slot_events"]:
                affordable = e["price"] <= e["my_max_bid"]
                marker = "← could have won" if affordable else ""
                print(f"    {e['player']:<28} cleared ${e['price']:>3}  "
                      f"max_bid ${e['my_max_bid']:>3}  I bid ${e['my_bid']:>3}  "
                      f"${e['my_money_remaining']:>3} left  {marker}")
            print()

    # ── Bucket A detail ───────────────────────────────────────────────────────
    if bucket_a:
        print(f"{'─'*W}")
        print(f"\nBUCKET A  — market was genuinely expensive\n")
        print(f"  {'Seed':>4}  {'Cheapest TE (cleared)':<38}  My max_bid  $ left")
        print("  " + "─" * 60)
        for r in sorted(bucket_a, key=lambda x: x["seed"]):
            ke = r["key_event"]
            player_s = ke.get("player", "?")[:26]
            print(f"  {r['seed']:>4}  {player_s:<26} ${ke.get('price',0):>3}  "
                  f"${ke.get('my_max_bid',0):>3}           "
                  f"${ke.get('my_money_remaining',0):>3}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DARTS mock auction draft")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (default: 2026)")
    parser.add_argument("--runs", type=int, default=0,
                        help="Run N simulations and print distribution (overrides --seed)")
    parser.add_argument("--diagnose-te", action="store_true",
                        help="Diagnose TE shutouts across --runs seeds")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full pick-by-pick auction log")
    args = parser.parse_args()

    print(f"Loading board…")
    board = compute_values()
    print(f"Board loaded: {len(board[board.is_drafted & (board.position != 'DEF')])} "
          f"offensive players + {len(board[board.is_drafted & (board.position == 'DEF')])} DEF")

    n_seeds = args.runs if args.runs > 0 else 50
    if args.diagnose_te:
        diagnose_te_shutouts(board, n_seeds=n_seeds)
    elif args.runs > 0:
        run_distribution(board, n_runs=args.runs)
    else:
        print(f"Running mock draft (seed={args.seed})…\n")
        state, auction_log = simulate(board, seed=args.seed)
        _print_report(state, auction_log, board)
        if args.verbose:
            _print_auction_log(auction_log)


if __name__ == "__main__":
    main()
