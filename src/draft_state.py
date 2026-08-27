"""
Phase 6 — Live draft assistant.

Tracks every pick and answers: "What is the most I should pay for this
player, right now, given everything that's happened?"

CEILINGS
--------
feasibility_max = money_remaining - (slots_remaining - 1)
    Hard constraint. Never exceed this — you'd be unable to fill your roster.

value_max = 1 + (raw_model_price - 1) × inflation_multiplier
    What the player is worth given how much money remains in the room.
    The $1 minimum doesn't inflate; only the discretionary portion does.

recommended = min(feasibility_max, value_max)

LIVE BIDDERS
------------
A team is a competitive bidder for position P if:
  - Their max_bid is meaningful (>= $2)
  - They have starter_need (count_at_P < STARTERS[P]) OR
    flex_need (total skill_drafted < 7 and P is FLEX-eligible)
Bench-only need is tracked but excluded from the competitive count —
it would make every team a bidder for everything.

PACING TRIPWIRE
---------------
Flag when: my $/slot > 1.5x league avg $/slot
           AND scarcity is tightening at any of my open starting positions.
Patience is a strategy; hoarding is a mistake.

Run:  python src/draft_state.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from league_config import (
    BUDGET_PER_TEAM, FLEX_ELIGIBLE, MIN_BID,
    NUM_TEAMS, ROSTER_SIZE, STARTERS,
)
from valuation import compute_values
from inflation import inflation_multiplier

DRAFT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "drafts")

# Total skill starter slots per roster: 2RB + 3WR + 1TE + 1FLEX = 7
SKILL_STARTER_TOTAL = (
    STARTERS["RB"] + STARTERS["WR"] + STARTERS["TE"] + STARTERS["FLEX"]
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Pick:
    player: str
    position: str
    salary: int
    fantasy_team: str
    par: float
    raw_price: float   # pre-rounding model price, for inflation math


@dataclass
class TeamState:
    name: str
    budget: int = BUDGET_PER_TEAM
    picks: list[Pick] = field(default_factory=list)

    # ── Money ────────────────────────────────────────────────────────────────

    @property
    def money_spent(self) -> int:
        return sum(p.salary for p in self.picks)

    @property
    def money_remaining(self) -> int:
        return self.budget - self.money_spent

    # ── Roster slots ─────────────────────────────────────────────────────────

    @property
    def slots_filled(self) -> int:
        return len(self.picks)

    @property
    def slots_remaining(self) -> int:
        return ROSTER_SIZE - self.slots_filled

    @property
    def max_bid(self) -> int:
        """Hard ceiling: can't bid more than money - remaining minimums."""
        if self.slots_remaining <= 0:
            return 0
        return max(
            self.money_remaining - (self.slots_remaining - 1) * MIN_BID,
            MIN_BID,
        )

    @property
    def dollars_per_slot(self) -> float:
        if self.slots_remaining <= 0:
            return float("inf")
        return self.money_remaining / self.slots_remaining

    # ── Positional tracking ───────────────────────────────────────────────────

    @property
    def position_counts(self) -> Counter:
        return Counter(p.position for p in self.picks)

    @property
    def skill_drafted(self) -> int:
        """Total RB + WR + TE picks (the pool that fills starters + FLEX)."""
        pc = self.position_counts
        return sum(pc.get(pos, 0) for pos in FLEX_ELIGIBLE)

    @property
    def flex_open(self) -> bool:
        """FLEX slot is still unfilled when total skill picks < 7."""
        return self.skill_drafted < SKILL_STARTER_TOTAL

    def starter_need(self, position: str) -> bool:
        return self.position_counts.get(position, 0) < STARTERS.get(position, 0)

    def positional_need(self, position: str) -> str:
        """
        'starter' — unfilled starting slot at this position.
        'flex'    — FLEX slot open and position is FLEX-eligible.
        'bench'   — only bench depth left.
        'none'    — fully loaded or roster full.
        """
        if self.slots_remaining <= 0:
            return "none"
        if self.starter_need(position):
            return "starter"
        if position in FLEX_ELIGIBLE and self.flex_open:
            return "flex"
        if self.slots_remaining > 0:
            return "bench"
        return "none"

    @property
    def total_par(self) -> float:
        return sum(p.par for p in self.picks)


# ---------------------------------------------------------------------------
# Draft state
# ---------------------------------------------------------------------------

class DraftState:
    """
    Tracks the full auction pick-by-pick.

    Parameters
    ----------
    my_team : str
        Your fantasy team name — used to split "me vs. the room."
    all_teams : list[str], optional
        All 10 team names. Teams not listed are auto-added when a pick
        comes in, but pre-registering them ensures they appear in bidder
        counts even before their first pick.
    board : DataFrame, optional
        Pre-computed valuation board. Loaded once from compute_values()
        if not supplied.
    """

    def __init__(
        self,
        my_team: str,
        all_teams: Sequence[str] | None = None,
        board: pd.DataFrame | None = None,
    ) -> None:
        self.my_team = my_team
        self._board = board
        self.teams: dict[str, TeamState] = {}
        if all_teams:
            for name in all_teams:
                self.teams[name] = TeamState(name=name)
        if my_team not in self.teams:
            self.teams[my_team] = TeamState(name=my_team)

    # ── Board ─────────────────────────────────────────────────────────────────

    @property
    def board(self) -> pd.DataFrame:
        if self._board is None:
            self._board = compute_values()
        return self._board

    def _player_info(self, player_name: str) -> dict | None:
        lower = player_name.lower().strip()
        rows = self.board[
            self.board["player_display_name"].str.lower().str.strip() == lower
        ]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            "name": row["player_display_name"],
            "position": row["position"],
            "par": float(row["par"]),
            "price": int(row["price"]),
            "raw_price": float(row["raw_price"]),
        }

    # ── Picks ─────────────────────────────────────────────────────────────────

    def add_pick(
        self,
        player: str,
        salary: int,
        fantasy_team: str,
        position: str | None = None,
    ) -> None:
        """
        Record one pick. Auto-adds any team not registered at init.

        position : override for roster-slot accounting when the player is
                   not on the model board (off-board players, recent retirees,
                   etc.). In live use, always pass the actual position so
                   roster state stays accurate even for undrafted-model picks.
        """
        info = self._player_info(player)
        resolved_pos = (
            info["position"]
            if info
            else (position or "UNK")
        )
        pick = Pick(
            player=player,
            position=resolved_pos,
            salary=salary,
            fantasy_team=fantasy_team,
            par=info["par"] if info else 0.0,
            raw_price=info["raw_price"] if info else float(salary),
        )
        if fantasy_team not in self.teams:
            self.teams[fantasy_team] = TeamState(name=fantasy_team)
        self.teams[fantasy_team].picks.append(pick)

    @property
    def all_picks(self) -> list[Pick]:
        return [p for ts in self.teams.values() for p in ts.picks]

    @property
    def my_state(self) -> TeamState:
        return self.teams[self.my_team]

    # ── Inflation ─────────────────────────────────────────────────────────────

    def _drafted_pairs(self) -> list[tuple[str, int]]:
        return [(p.player, p.salary) for p in self.all_picks]

    def inflation_state(self) -> dict:
        return inflation_multiplier(
            self.board,
            self._drafted_pairs(),
            teams_still_bidding=len(self.teams),
            my_remaining_money=self.my_state.money_remaining,
        )

    # ── Personal rate ─────────────────────────────────────────────────────────

    def _my_personal_rate(self) -> float:
        """
        My $/PAR rate based on MY remaining money and MY proportional share
        of the remaining PAR pool — not the league's current rate.

        Formula:
          my_rate = my_discretionary / (remaining_par × my_slots / total_slots)

        When I'm holding cash while others spend, my_slots/total_slots is high
        relative to my share of remaining money, so my_rate exceeds the
        league's current rate and opportunity_cost_max rises above value_max.
        That is the signal to bid up.
        """
        me = self.my_state
        if me.slots_remaining <= 0:
            return 0.0

        my_discretionary = max(
            0, me.money_remaining - (me.slots_remaining - 1) * MIN_BID
        )

        drafted_lower = {p.player.lower() for p in self.all_picks}
        remaining_par = self.board[
            (self.board["par"] > 0)
            & (self.board["is_drafted"])
            & (self.board["position"] != "DEF")
            & ~self.board["player_display_name"].str.lower().isin(drafted_lower)
        ]["par"].sum()

        total_slots = sum(ts.slots_remaining for ts in self.teams.values())
        if total_slots <= 0 or remaining_par <= 0:
            return 0.0

        my_par_share = remaining_par * me.slots_remaining / total_slots
        return my_discretionary / my_par_share

    def _my_positional_rate(self, position: str, need: str = "starter") -> float:
        """
        $/PAR rate scaled up for positionally scarce spots, but ONLY when
        filling a starter slot. Bench/flex bids use the base personal rate —
        no scarcity premium for depth picks.

        TE PAR is scarcer than RB PAR (16 TEs drafted vs 42 RBs).
        A point of TE PAR costs more because there are fewer substitutes.

        scarcity_factor = max(1.0, avg_replacement_rank / pos_replacement_rank)

        Floor at 1.0: deep positions (WR, RB) are not discounted, only
        scarce ones (TE, QB) are boosted.
        """
        from scarcity import replacement_ranks

        base_rate = self._my_personal_rate()
        if base_rate == 0:
            return 0.0

        if need != "starter":
            return base_rate

        deep_ranks = replacement_ranks(include_bench=True)
        pos_rank = deep_ranks.get(position)
        if pos_rank is None:
            return base_rate

        skill_positions = {p: deep_ranks[p] for p in ("QB", "RB", "WR", "TE") if p in deep_ranks}
        avg_rank = sum(skill_positions.values()) / len(skill_positions)
        scarcity_factor = max(1.0, avg_rank / pos_rank)
        return base_rate * scarcity_factor

    # ── Ceilings ──────────────────────────────────────────────────────────────

    def my_ceilings(self, player_name: str) -> dict:
        """
        Three price ceilings for the named player.

        feasibility_max   — hard budget constraint, never exceed
        mkt_value_max     — inflation-adjusted model price at LEAGUE rate
                            (what the market can clear; kept for reference)
        opp_cost_max      — MY personal rate applied to this player's PAR
                            rises above mkt_value_max when I have excess cash

        recommended = min(feasibility_max, opp_cost_max)

        Which ceiling binds tells you WHY:
          feasibility binds → you are budget-constrained on this player
          opp_cost binds    → you have capacity but this player isn't worth more
          opp_cost > mkt    → your excess cash lets you outbid the room
        """
        info = self._player_info(player_name)
        me = self.my_state
        infl = self.inflation_state()
        mult = infl["multiplier"]

        # 1. Hard ceiling
        feasibility_max = me.max_bid

        # 2. Market value ceiling (old value_max — league rate)
        if info is not None and not np.isnan(mult):
            mkt_value_max = max(MIN_BID, round(1 + (info["raw_price"] - 1) * mult))
        elif info is not None:
            mkt_value_max = info["price"]
        else:
            mkt_value_max = feasibility_max

        # 3. Opportunity-cost ceiling (MY rate applied to this player's full PAR)
        #    opp_cost = $1 + par × my_rate
        #    Marginal PAR vs fallback is shown for context, but the formula
        #    uses full PAR because the $1 minimum already covers the "floor."
        pos  = info["position"] if info else ""
        need = self.my_state.positional_need(pos) if info else "bench"
        my_rate = self._my_positional_rate(pos, need=need)

        # Find fallback: best remaining at same position (for display context)
        fallback_name, fallback_par = None, 0.0
        if info is not None:
            drafted_lower = {p.player.lower() for p in self.all_picks}
            target_lower = player_name.lower().strip()
            fb_pool = self.board[
                (self.board["position"] == info["position"])
                & (self.board["par"] > 0)
                & (self.board["is_drafted"])
                & ~self.board["player_display_name"].str.lower().isin(drafted_lower)
                & (self.board["player_display_name"].str.lower() != target_lower)
            ].nlargest(1, "par")
            if not fb_pool.empty:
                fallback_name = fb_pool.iloc[0]["player_display_name"]
                fallback_par = float(fb_pool.iloc[0]["par"])

        if info is not None and my_rate > 0:
            opp_cost_max = max(MIN_BID, round(1 + info["par"] * my_rate))
        elif info is not None:
            opp_cost_max = mkt_value_max
        else:
            opp_cost_max = feasibility_max

        marginal_par = max(0.0, (info["par"] if info else 0.0) - fallback_par)

        recommended = min(feasibility_max, opp_cost_max)

        binding = "feasibility" if feasibility_max <= opp_cost_max else "opp_cost"

        return {
            "player": player_name,
            "position": info["position"] if info else "UNK",
            "model_price": info["price"] if info else None,
            "par": info["par"] if info else None,
            "feasibility_max": feasibility_max,
            "mkt_value_max": mkt_value_max,
            "opp_cost_max": opp_cost_max,
            "recommended": recommended,
            "binding": binding,
            "multiplier": mult,
            "my_rate": my_rate,
            "fallback_player": fallback_name,
            "fallback_par": fallback_par,
            "marginal_par": marginal_par,
        }

    # ── Live bidders ──────────────────────────────────────────────────────────

    def live_bidders(self, position: str, min_bid: int = 2) -> dict:
        """
        Teams (excluding mine) that need position P and can afford to bid.

        competitive_count = starter_need + flex_need teams with max_bid >= min_bid
        max_competing_bid = highest max_bid among competitive bidders
        """
        starter_b, flex_b, bench_b = [], [], []

        for name, ts in self.teams.items():
            if name == self.my_team:
                continue
            need = ts.positional_need(position)
            if ts.max_bid < min_bid:
                continue
            entry = {
                "team": name,
                "need": need,
                "max_bid": ts.max_bid,
                "money_remaining": ts.money_remaining,
                "slots_remaining": ts.slots_remaining,
                "pos_count": ts.position_counts.get(position, 0),
            }
            if need == "starter":
                starter_b.append(entry)
            elif need == "flex":
                flex_b.append(entry)
            elif need == "bench":
                bench_b.append(entry)

        competitive = starter_b + flex_b
        max_competing = max((e["max_bid"] for e in competitive), default=0)

        return {
            "position": position,
            "starter_bidders": sorted(starter_b, key=lambda x: -x["max_bid"]),
            "flex_bidders": sorted(flex_b, key=lambda x: -x["max_bid"]),
            "bench_bidders": bench_b,
            "competitive_count": len(competitive),
            "max_competing_bid": max_competing,
        }

    # ── Pacing ────────────────────────────────────────────────────────────────

    def pacing_check(self) -> dict:
        me = self.my_state
        all_ts = list(self.teams.values())

        league_money = sum(ts.money_remaining for ts in all_ts)
        league_slots = sum(ts.slots_remaining for ts in all_ts)
        league_avg_dps = league_money / league_slots if league_slots > 0 else 0.0

        my_dps = me.dollars_per_slot
        ratio = my_dps / league_avg_dps if league_avg_dps > 0 else float("inf")
        hoarding = ratio > 1.5

        my_open_starters = [
            pos for pos in ("QB", "RB", "WR", "TE")
            if me.starter_need(pos)
        ]
        scarcity_alerts = [
            pos for pos in my_open_starters
            if self._scarcity_tight(pos)
        ]

        # Spend-pacing: avg needed vs avg target price
        avg_needed = (
            me.money_remaining / me.slots_remaining if me.slots_remaining > 0 else 0.0
        )

        # Average inflation price of top-3 undrafted above-replacement players
        # at each open starter position. This is what I actually want to buy.
        drafted_lower = {p.player.lower() for p in self.all_picks}
        infl = self.inflation_state()
        adj_board = infl["adjusted_board"]

        # Target positions: open starters → FLEX-eligible if flex slot open → all skill
        if my_open_starters:
            target_positions = my_open_starters
        elif me.flex_open:
            target_positions = list(FLEX_ELIGIBLE)
        else:
            target_positions = ["QB", "RB", "WR", "TE"]

        n_top = max(3, 3 * len(target_positions))
        target_pool = adj_board[
            adj_board["position"].isin(target_positions)
            & (adj_board["par"] > 0)
            & adj_board["is_drafted"]
            & ~adj_board["player_display_name"].str.lower().isin(drafted_lower)
        ].nlargest(n_top, "par")
        avg_target_price = (
            float(target_pool["inflation_price"].mean())
            if not target_pool.empty else 0.0
        )

        underspending = (avg_needed > avg_target_price) and me.slots_remaining > 0

        return {
            "my_dps": my_dps,
            "league_avg_dps": league_avg_dps,
            "ratio": ratio,
            "hoarding": hoarding,
            "my_open_starters": my_open_starters,
            "scarcity_at_open_positions": scarcity_alerts,
            "tripwire": hoarding and bool(scarcity_alerts),
            "avg_needed": avg_needed,
            "avg_target_price": avg_target_price,
            "underspending": underspending,
        }

    # ── Scarcity ──────────────────────────────────────────────────────────────

    def _scarcity_tight(self, position: str, buffer: float = 1.5) -> bool:
        drafted_lower = {p.player.lower() for p in self.all_picks}
        n_available = (
            self.board[
                (self.board["position"] == position)
                & (self.board["par"] > 0)
                & ~self.board["player_display_name"].str.lower().isin(drafted_lower)
            ]
        ).shape[0]
        teams_needing = sum(
            1 for ts in self.teams.values() if ts.starter_need(position)
        )
        return n_available <= teams_needing * buffer

    def scarcity_status(self, position: str) -> dict:
        drafted_lower = {p.player.lower() for p in self.all_picks}
        available = self.board[
            (self.board["position"] == position)
            & (self.board["par"] > 0)
            & ~self.board["player_display_name"].str.lower().isin(drafted_lower)
        ]
        teams_starter = sum(1 for ts in self.teams.values() if ts.starter_need(position))
        teams_flex = (
            sum(
                1 for ts in self.teams.values()
                if ts.positional_need(position) in ("starter", "flex")
            )
            if position in FLEX_ELIGIBLE else 0
        )
        return {
            "position": position,
            "above_replacement_left": len(available),
            "teams_needing_starter": teams_starter,
            "teams_needing_flex_or_starter": teams_flex,
            "scarcity_tight": self._scarcity_tight(position),
            "top_available": available.head(3)[
                ["player_display_name", "par", "price"]
            ].to_dict("records"),
        }

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def dashboard(self, target_player: str | None = None) -> None:
        """Print the full draft-night dashboard."""
        me = self.my_state
        infl = self.inflation_state()
        pacing = self.pacing_check()
        W = 72

        print("=" * W)
        print(f"DARTS 2026  |  {self.my_team}")
        print(
            f"Pick {len(self.all_picks)}  |  "
            f"${infl['money_spent']:,} spent leaguewide  |  "
            f"${infl['remaining_money']:,} left  |  "
            f"Mult {infl['multiplier']:.3f}x"
        )
        print("=" * W)

        # ── My status ────────────────────────────────────────────────────────
        print("\nMY STATUS")
        pc = me.position_counts
        pos_str = "  ".join(
            f"{p}:{pc.get(p, 0)}" for p in ("QB", "RB", "WR", "TE", "DEF")
        )
        print(f"  ${me.money_remaining} remaining  |  max bid ${me.max_bid}  |  "
              f"{me.slots_remaining} slots left")
        print(f"  {pos_str}")

        trip = pacing["tripwire"]
        pacing_flag = "  ⚠ TRIPWIRE — stop hoarding" if trip else ""
        print(
            f"  ${pacing['my_dps']:.1f}/slot  "
            f"(league avg ${pacing['league_avg_dps']:.1f})  "
            f"= {pacing['ratio']:.2f}x{pacing_flag}"
        )
        if trip:
            print(
                f"    Scarcity tightening at: "
                f"{', '.join(pacing['scarcity_at_open_positions'])}"
            )
        # Spend-pacing line
        pace_flag = "  ← UNDERSPENDING — bid up" if pacing["underspending"] else ""
        print(
            f"  Avg needed/slot ${pacing['avg_needed']:.1f}  "
            f"| targets avg ${pacing['avg_target_price']:.1f}{pace_flag}"
        )

        power = infl["my_purchasing_power"]
        if not np.isnan(power):
            direction = "ABOVE avg — good time to spend" if power > 1.0 else "below avg — conserve"
            print(f"  Purchasing power: {power:.2f}x avg team  ({direction})")

        # ── Scarcity watch ───────────────────────────────────────────────────
        print("\nSCARCITY WATCH")
        for pos in ("QB", "RB", "WR", "TE"):
            sc = self.scarcity_status(pos)
            alert = "  ← ALERT" if sc["scarcity_tight"] else ""
            my_need = "need" if me.starter_need(pos) else "done"
            top_names = ", ".join(
                r["player_display_name"].split()[-1]   # last name only
                for r in sc["top_available"]
            )
            print(
                f"  {pos} [{my_need:>4}]:  "
                f"{sc['above_replacement_left']:>3} above-repl left  |  "
                f"{sc['teams_needing_starter']:>2} teams need starter  |  "
                f"next: {top_names}{alert}"
            )

        # ── Target player ────────────────────────────────────────────────────
        if target_player:
            print("\n" + "─" * W)
            c = self.my_ceilings(target_player)
            b = self.live_bidders(c["position"])
            par_str = f"{c['par']:.1f}" if c["par"] is not None else "n/a"
            print(
                f"\nTARGET: {target_player}  ({c['position']})  "
                f"model ${c['model_price']}  PAR {par_str}"
            )
            print()
            print(f"  Feasibility max:  ${c['feasibility_max']:>3}  "
                  "(hard ceiling — never exceed)")
            print(f"  Mkt value max:    ${c['mkt_value_max']:>3}  "
                  "(league rate, for reference)")
            opp_note = f"  (MY rate ${c['my_rate']:.2f}/PAR)"
            print(f"  Opp-cost max:     ${c['opp_cost_max']:>3}{opp_note}")
            binding_label = (
                "feasibility binds — budget-constrained"
                if c["binding"] == "feasibility"
                else "opp-cost binds"
            )
            print(f"  ▶ Recommended:    ${c['recommended']:>3}  [{binding_label}]")
            if c["opp_cost_max"] > c["mkt_value_max"]:
                print(
                    f"    ↳ Excess cash lifts your ceiling above the market "
                    f"(${c['opp_cost_max']} vs mkt ${c['mkt_value_max']})"
                )
            if c["fallback_player"]:
                print(
                    f"    Fallback: {c['fallback_player']}  "
                    f"PAR {c['fallback_par']:.1f}  "
                    f"(marginal PAR {c['marginal_par']:.1f})"
                )

            print(
                f"\n  Live bidders for {c['position']}:  "
                f"{b['competitive_count']} competitive  "
                f"(starter: {len(b['starter_bidders'])}  "
                f"FLEX: {len(b['flex_bidders'])})"
            )
            print(f"  Highest competing max bid: ${b['max_competing_bid']}")
            if b["max_competing_bid"] > 0 and b["max_competing_bid"] < c["recommended"]:
                print("    ↳ Your ceiling exceeds the room — you set the price.")
            elif b["max_competing_bid"] >= c["recommended"]:
                print("    ↳ Others can match your recommended bid — expect competition.")

            for label, group in (
                ("Starter need", b["starter_bidders"]),
                ("FLEX need", b["flex_bidders"]),
            ):
                if not group:
                    continue
                print(f"\n  {label}:")
                for e in group:
                    print(
                        f"    {e['team']:<32}  max ${e['max_bid']:>3}  "
                        f"${e['money_remaining']} left  {e['slots_remaining']} slots  "
                        f"{c['position']}s:{e['pos_count']}"
                    )

        print()


# ---------------------------------------------------------------------------
# Demo — replay 2025 draft at picks 30, 60, 90, 120 from King's perspective
# ---------------------------------------------------------------------------

def _best_available(state: DraftState, prefer_pos: list[str]) -> str | None:
    """
    Find the highest-PAR undrafted above-replacement player at the first
    position in prefer_pos that has any remaining above-replacement players.
    Falls back to overall best available if prefer_pos is exhausted.
    """
    drafted_lower = {p.player.lower() for p in state.all_picks}
    pool = state.board[
        (state.board["par"] > 0)
        & state.board["is_drafted"]
        & (state.board["position"] != "DEF")
        & ~state.board["player_display_name"].str.lower().isin(drafted_lower)
    ]
    for pos in prefer_pos:
        sub = pool[pool["position"] == pos].nlargest(1, "par")
        if not sub.empty:
            return sub.iloc[0]["player_display_name"]
    overall = pool.nlargest(1, "par")
    return overall.iloc[0]["player_display_name"] if not overall.empty else None


if __name__ == "__main__":
    draft = pd.read_csv(os.path.join(DRAFT_DIR, "draft_2025.csv"))
    skill = draft[draft["position"].isin(["QB", "RB", "WR", "TE"])].sort_values("pick")
    skill_rows = list(skill.iterrows())  # list so we can slice by index

    all_teams = sorted(draft["fantasy_team"].dropna().unique().tolist())
    MY_TEAM = "King"

    # Load board once; pass to every DraftState instance
    board = compute_values()

    MILESTONES = [30, 60, 90, 120]

    for milestone in MILESTONES:
        state = DraftState(my_team=MY_TEAM, all_teams=all_teams, board=board)

        for _, row in skill_rows[:milestone]:
            state.add_pick(
                row["player"], int(row["salary"]), row["fantasy_team"],
                position=row["position"],
            )

        me = state.my_state

        # Target priority: open starter positions ordered by scarcity value,
        # then flex, then bench. TE first because the positional edge is largest there.
        open_starters = [p for p in ("TE", "WR", "RB", "QB") if me.starter_need(p)]
        if not open_starters and me.flex_open:
            # Flex still open — prefer whatever has best PAR left
            open_starters = ["WR", "RB", "TE"]
        target = _best_available(state, open_starters or ["WR", "RB", "QB", "TE"])

        print(f"\n{'#'*72}")
        print(f"# SKILL PICK {milestone}  —  King's picks so far:")
        pc = me.position_counts
        king_picks_str = "  ".join(
            f"{p.player} ({p.position}, ${p.salary})"
            for p in sorted(me.picks, key=lambda x: x.salary, reverse=True)
        )
        print(f"#  {king_picks_str}")
        print(f"#  Spent ${me.money_spent} | ${me.money_remaining} left | "
              f"QB:{pc.get('QB',0)} RB:{pc.get('RB',0)} "
              f"WR:{pc.get('WR',0)} TE:{pc.get('TE',0)}")
        print(f"{'#'*72}")

        state.dashboard(target_player=target)
