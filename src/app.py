"""
Phase 5 — Streamlit draft-night interface for DARTS 2026.

Clean single-screen layout optimized for live auction speed:
  - Type-ahead player search (fuzzy substring)
  - Big MAX BID number dominates the screen
  - 3-4 line player context card
  - Always-visible roster / budget / pacing / inflation
  - Auto-save state after every pick; undo supported

Uses draft_state.py for all pricing logic — this is a UI layer only.

Run:  streamlit run src/app.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from league_config import (  # noqa: E402
    STARTERS, FLEX_ELIGIBLE, ROSTER_SIZE, BUDGET_PER_TEAM, MIN_BID,
)
from valuation import compute_values  # noqa: E402
from draft_state import DraftState  # noqa: E402

# ---------------------------------------------------------------------------
# League config — ground-truth 2025 team names from Yahoo
# ---------------------------------------------------------------------------
MY_TEAM = "King"
OPPONENT_NAMES = [
    "Chasing Mason",
    "MERDY",
    "The Ra-volution",
    "Team Taruchus",
    "Field Njigbas",
    "Justin's (not Jamie's) Team",
    "My Nix Hurts",
    "Itty Bitty Pitts Committee",
    "Who needs a qb?",
]
ALL_TEAMS = [MY_TEAM] + OPPONENT_NAMES

DATA_DIR = os.path.join(APP_DIR, "..", "data")
STATE_FILE = os.path.join(DATA_DIR, "drafts", "live_draft_2026.json")
LAST_YEAR_FILE = os.path.join(DATA_DIR, "drafts", "draft_2025.csv")
OPP_DIR = os.path.join(DATA_DIR, "opportunity")
HISTORY_FILE = os.path.join(DATA_DIR, "processed", "player_seasons.parquet")
SCOUTING_FILE = os.path.join(DATA_DIR, "drafts", "scouting_report.md")

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DARTS 2026 Draft",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Remove top padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

    /* The big number */
    .max-bid-num {
        font-size: 140px;
        font-weight: 800;
        line-height: 1;
        text-align: center;
        letter-spacing: -6px;
        color: #1a5490;
        margin: 0.1em 0 0 0;
    }
    .max-bid-num.over  { color: #c73030; }
    .max-bid-num.done  { color: #888; opacity: 0.35; }
    .max-bid-num.small { font-size: 100px; }

    .max-bid-label {
        text-align: center;
        font-size: 15px;
        letter-spacing: 2px;
        color: #888;
        text-transform: uppercase;
        margin-bottom: 0.3em;
    }

    .player-headline {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin: 0.3em 0 0 0;
        line-height: 1.15;
    }
    .player-sub {
        text-align: center;
        font-size: 17px;
        color: #666;
        margin: 0 0 0.6em 0;
    }
    .support-line {
        text-align: center;
        color: #888;
        font-size: 15px;
        margin: 0.4em 0;
    }
    .support-line b { color: #333; }

    .context-card {
        background: #fafafa;
        border-left: 3px solid #ccc;
        padding: 0.8em 1.2em;
        margin: 1em auto 0 auto;
        max-width: 640px;
        border-radius: 4px;
    }
    .context-line {
        font-size: 15px;
        margin: 0.3em 0;
        color: #333;
        line-height: 1.45;
    }
    .context-line.risk { color: #a04040; }
    .context-line.market { color: #666; font-style: italic; }

    .roster-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35em 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 15px;
    }
    .roster-need { color: #c73030; font-weight: 600; }
    .roster-done { color: #999; }

    .status-line {
        color: #666;
        font-size: 13px;
        padding: 0.1em 0;
    }
    .alert-red   { background: #fff2f2; padding: 0.15em 0.5em; border-radius: 3px; color: #a03030; }
    .alert-amber { background: #fff8e6; padding: 0.15em 0.5em; border-radius: 3px; color: #8a6a1e; }
    .alert-green { color: #2a7a2a; }

    /* Make headers quieter */
    h2 { font-size: 20px !important; color: #555 !important; font-weight: 600 !important;
         margin-top: 0.6em !important; }
    h3 { font-size: 16px !important; color: #777 !important; font-weight: 600 !important; }

    /* Tighter input styling */
    div[data-testid="stNumberInput"] input { font-size: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_board():
    return compute_values()

@st.cache_data
def load_last_year_prices():
    df = pd.read_csv(LAST_YEAR_FILE)
    return {
        str(row["player"]).strip().lower(): {
            "salary": int(row["salary"]),
            "winner": str(row["fantasy_team"]),
        }
        for _, row in df.iterrows()
    }

@st.cache_data
def load_opportunity_notes():
    """situation_note keyed by name — role_tier is already on the board."""
    notes: dict[str, str] = {}
    for pos in ("rb", "wr", "te"):
        fpath = os.path.join(OPP_DIR, f"{pos}_2026.csv")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            note = row.get("situation_note")
            if pd.notna(note) and str(note).strip():
                notes[str(row["player_name"]).strip().lower()] = str(note).strip()
    return notes

@st.cache_data
def load_positional_ranks():
    """Last-season (2025) positional rank + ppg + games."""
    df = pd.read_parquet(HISTORY_FILE)
    df = df[df["season"] == 2025].copy()
    ranks: dict[str, dict] = {}
    for pos in df["position"].unique():
        sub = df[df["position"] == pos].sort_values("ppg", ascending=False).reset_index(drop=True)
        for i, row in sub.iterrows():
            key = str(row["player_display_name"]).strip().lower()
            ranks[key] = {
                "position": pos,
                "rank": i + 1,
                "ppg": float(row["ppg"]) if pd.notna(row["ppg"]) else 0.0,
                "games": int(row["games"]) if pd.notna(row["games"]) else 0,
            }
    return ranks

@st.cache_data
def load_scouting_md():
    if not os.path.exists(SCOUTING_FILE):
        return "*(scouting report not found)*"
    with open(SCOUTING_FILE) as f:
        return f.read()

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def _save_picks_to_disk():
    picks_out = []
    for team in st.session_state.draft.teams.values():
        for p in team.picks:
            picks_out.append({
                "player": p.player, "salary": p.salary,
                "fantasy_team": p.fantasy_team, "position": p.position,
            })
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({
            "my_team": MY_TEAM,
            "picks": picks_out,
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2)

def _rebuild_from_picks(picks: list[dict]) -> DraftState:
    board = load_board()
    ds = DraftState(my_team=MY_TEAM, all_teams=ALL_TEAMS, board=board)
    for p in picks:
        ds.add_pick(
            p["player"], int(p["salary"]),
            p["fantasy_team"], position=p.get("position"),
        )
    return ds

def _load_state():
    board = load_board()
    if not os.path.exists(STATE_FILE):
        st.session_state.picks_history = []
        return DraftState(my_team=MY_TEAM, all_teams=ALL_TEAMS, board=board)
    with open(STATE_FILE) as f:
        data = json.load(f)
    st.session_state.picks_history = data.get("picks", [])
    return _rebuild_from_picks(st.session_state.picks_history)

def add_pick(player, salary, team, position):
    st.session_state.draft.add_pick(player, int(salary), team, position=position)
    st.session_state.picks_history.append({
        "player": player, "salary": int(salary),
        "fantasy_team": team, "position": position,
    })
    _save_picks_to_disk()

def undo_last():
    if not st.session_state.picks_history:
        return False
    st.session_state.picks_history.pop()
    st.session_state.draft = _rebuild_from_picks(st.session_state.picks_history)
    _save_picks_to_disk()
    return True

# ---------------------------------------------------------------------------
# Init session state
# ---------------------------------------------------------------------------
if "draft" not in st.session_state:
    st.session_state.draft = _load_state()
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "current_bid" not in st.session_state:
    st.session_state.current_bid = 0

board = load_board()
last_year = load_last_year_prices()
opp_notes = load_opportunity_notes()
pos_ranks = load_positional_ranks()

draft: DraftState = st.session_state.draft
me = draft.my_state

# ---------------------------------------------------------------------------
# Player context card
# ---------------------------------------------------------------------------
def build_context(row) -> list[tuple[str, str]]:
    """Returns [(class, text), ...] — 3–4 lines about the player."""
    lines: list[tuple[str, str]] = []
    key = str(row["player_display_name"]).strip().lower()

    # Line 1: last season rank + ppg + gp
    if key in pos_ranks:
        r = pos_ranks[key]
        lines.append(("", f"{r['position']}{r['rank']} in 2025 · {r['ppg']:.1f} ppg · {r['games']} games"))
    else:
        lines.append(("", "No 2025 stat line (rookie or DNP)"))

    # Line 2: role + projection
    role = row.get("role_tier")
    proj = row.get("proj_ppg")
    parts = []
    if pd.notna(role) and str(role).strip():
        parts.append(str(role).replace("_", " ").strip())
    if pd.notna(proj) and proj:
        parts.append(f"{float(proj):.1f} projected ppg")
    if parts:
        lines.append(("", " · ".join(parts)))

    # Line 3: situation note (risk / context)
    if key in opp_notes:
        note = opp_notes[key]
        if len(note) > 160:
            note = note[:157] + "…"
        lines.append(("risk", note))

    # Line 4: last-year market price
    if key in last_year:
        ly = last_year[key]
        lines.append(("market", f"2025 auction: ${ly['salary']} → {ly['winner']}"))

    return lines

# ---------------------------------------------------------------------------
# TOP BAR — always visible
# ---------------------------------------------------------------------------
pace = draft.pacing_check()
infl = draft.inflation_state()
budget_left = me.money_remaining
slots_left  = me.slots_remaining
max_bid     = me.max_bid
dps         = me.dollars_per_slot if slots_left > 0 else 0
lg_avg_dps  = pace["league_avg_dps"]

top_a, top_b, top_c, top_d, top_e = st.columns([1.2, 1.2, 1.2, 1.2, 1.4])
with top_a:
    st.markdown(f"<div class='status-line'>BUDGET</div><h3 style='margin-top:0'>${budget_left}</h3>", unsafe_allow_html=True)
with top_b:
    st.markdown(f"<div class='status-line'>SLOTS LEFT</div><h3 style='margin-top:0'>{slots_left}/{ROSTER_SIZE}</h3>", unsafe_allow_html=True)
with top_c:
    st.markdown(f"<div class='status-line'>MAX BID</div><h3 style='margin-top:0'>${max_bid}</h3>", unsafe_allow_html=True)
with top_d:
    ratio = pace["ratio"] if pace["ratio"] != float("inf") else 0
    hoard_flag = " ⚠" if pace["hoarding"] else ""
    st.markdown(
        f"<div class='status-line'>$/SLOT</div>"
        f"<h3 style='margin-top:0'>${dps:.1f} <span class='subtle'>vs ${lg_avg_dps:.1f}{hoard_flag}</span></h3>",
        unsafe_allow_html=True,
    )
with top_e:
    mult = infl.get("multiplier", 1.0)
    power = infl.get("my_purchasing_power", 1.0)
    if pd.notna(mult):
        st.markdown(
            f"<div class='status-line'>INFLATION</div>"
            f"<h3 style='margin-top:0'>{mult:.2f}× <span class='subtle'>purchasing {power:.2f}×</span></h3>",
            unsafe_allow_html=True,
        )

st.markdown("<hr style='margin:0.4em 0 1.2em 0; border:none; border-top:1px solid #eee'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_live, tab_board, tab_scout = st.tabs(["● Live draft", "Full board", "Scouting"])

# ============================================================================
# LIVE TAB
# ============================================================================
with tab_live:
    left, right = st.columns([2.6, 1])

    with left:
        # Available players (undrafted, on board)
        drafted_lower = {p.player.lower().strip() for p in draft.all_picks}
        undrafted = board[
            ~board["player_display_name"].str.lower().str.strip().isin(drafted_lower)
        ].copy()
        undrafted = undrafted.sort_values("price", ascending=False)
        player_options = undrafted["player_display_name"].tolist()

        # Type-ahead search
        selected = st.selectbox(
            "🔍 Player up for bid",
            options=[""] + player_options,
            index=0,
            key="player_search",
            placeholder="Start typing…",
            label_visibility="collapsed",
        )

        if selected:
            row = undrafted[undrafted["player_display_name"] == selected].iloc[0]
            c = draft.my_ceilings(selected)
            bidders = draft.live_bidders(c["position"])
            my_need = me.positional_need(c["position"])

            # Player headline
            team = row.get("team", "")
            st.markdown(
                f"<div class='player-headline'>{selected}</div>"
                f"<div class='player-sub'>{c['position']} · {team}</div>",
                unsafe_allow_html=True,
            )

            # THE MAX BID — big number
            rec = int(c["recommended"])
            current_bid = st.session_state.current_bid
            css_class = "max-bid-num"
            label = "MY MAX BID"
            if my_need == "none":
                css_class += " done"
                label = "POSITION FULL"
            elif current_bid and current_bid > rec:
                css_class += " over"
                label = "BID EXCEEDS MY MAX"

            st.markdown(f"<div class='max-bid-label'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='{css_class}'>${rec}</div>", unsafe_allow_html=True)

            # Supporting line
            model_px = c.get("model_price") or 0
            binding = "opp-cost binds" if c["binding"] == "opp_cost" else "feasibility binds"
            support = f"Model <b>${model_px}</b> · {binding} · MY rate ${c['my_rate']:.2f}/PAR"
            st.markdown(f"<div class='support-line'>{support}</div>", unsafe_allow_html=True)

            # Live bidder line
            bidder_line = (
                f"<b>{bidders['competitive_count']}</b> teams competitive · "
                f"highest max <b>${bidders['max_competing_bid']}</b>"
            )
            st.markdown(f"<div class='support-line'>{bidder_line}</div>", unsafe_allow_html=True)

            # Live current-bid input (small, unobtrusive)
            with st.container():
                bid_col1, bid_col2 = st.columns([1, 3])
                with bid_col1:
                    st.session_state.current_bid = st.number_input(
                        "Room is at $",
                        min_value=0, max_value=BUDGET_PER_TEAM,
                        value=int(st.session_state.current_bid or 0),
                        step=1, label_visibility="visible",
                    )

            # Player context card
            ctx = build_context(row)
            if ctx:
                ctx_html = "<div class='context-card'>"
                for cls, txt in ctx:
                    ctx_html += f"<div class='context-line {cls}'>{txt}</div>"
                ctx_html += "</div>"
                st.markdown(ctx_html, unsafe_allow_html=True)

        else:
            st.markdown(
                "<div style='text-align:center; padding: 4em 0; color:#aaa;'>"
                "Start typing a player name above."
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Pick entry (bottom of left column) ───────────────────────────────
        st.markdown("<hr style='margin:1.5em 0 0.8em 0'>", unsafe_allow_html=True)
        st.markdown("### Record pick")

        with st.form("pick_form", clear_on_submit=True):
            f1, f2, f3, f4 = st.columns([3, 1, 2.4, 1.2])
            with f1:
                pick_player = st.selectbox(
                    "Player",
                    options=[""] + player_options,
                    index=(1 + player_options.index(selected)) if selected in player_options else 0,
                    label_visibility="collapsed",
                    placeholder="Player…",
                )
            with f2:
                pick_price = st.number_input(
                    "Price", min_value=1, max_value=BUDGET_PER_TEAM,
                    value=1, step=1, label_visibility="collapsed",
                )
            with f3:
                pick_team = st.selectbox(
                    "Team", options=ALL_TEAMS,
                    index=ALL_TEAMS.index(MY_TEAM),
                    label_visibility="collapsed",
                )
            with f4:
                submitted = st.form_submit_button("SAVE", width="stretch")

            if submitted and pick_player:
                r = undrafted[undrafted["player_display_name"] == pick_player]
                pos = r.iloc[0]["position"] if not r.empty else None
                add_pick(pick_player, int(pick_price), pick_team, pos)
                st.session_state.current_bid = 0
                st.rerun()

        # Undo + last pick display
        under_a, under_b = st.columns([1, 4])
        with under_a:
            if st.button("↶ UNDO", width="stretch"):
                if undo_last():
                    st.session_state.current_bid = 0
                    st.rerun()
        with under_b:
            if st.session_state.picks_history:
                last = st.session_state.picks_history[-1]
                st.markdown(
                    f"<div class='status-line' style='padding-top:0.6em'>"
                    f"Last: <b>{last['player']}</b> → {last['fantasy_team']} at ${last['salary']}"
                    f"</div>", unsafe_allow_html=True,
                )

    # ── RIGHT PANEL: roster + scarcity + pacing ─────────────────────────────
    with right:
        st.markdown("### My roster")
        pc = me.position_counts
        my_starter_slots = ("QB", "RB", "WR", "TE", "FLEX", "DEF")

        # Compute FLEX filled status
        skill_ct  = me.skill_drafted
        flex_max  = STARTERS["FLEX"]
        flex_filled = max(0, min(flex_max, skill_ct - (STARTERS["RB"] + STARTERS["WR"] + STARTERS["TE"])))

        roster_html = ""
        for slot in my_starter_slots:
            cap = STARTERS.get(slot, 0)
            if slot == "FLEX":
                cur = flex_filled
            else:
                cur = min(pc.get(slot, 0), cap)
            need = cur < cap
            cls = "roster-need" if need else "roster-done"
            roster_html += (
                f"<div class='roster-row'><span class='{cls}'>{slot}</span>"
                f"<span class='{cls}'>{cur}/{cap}</span></div>"
            )
        # Bench summary
        drafted = len(me.picks)
        starter_slots_total = sum(STARTERS.values())
        bench_used = max(0, drafted - starter_slots_total)
        bench_cap = ROSTER_SIZE - starter_slots_total
        roster_html += (
            f"<div class='roster-row'><span class='roster-done'>BENCH</span>"
            f"<span class='roster-done'>{bench_used}/{bench_cap}</span></div>"
        )
        st.markdown(roster_html, unsafe_allow_html=True)

        # Pacing single line
        st.markdown("<div style='height:0.6em'></div>", unsafe_allow_html=True)
        avg_needed = pace["avg_needed"]
        avg_target = pace["avg_target_price"]
        if pace["tripwire"]:
            pace_msg = f"<span class='alert-red'>⚠ HOARDING — targets tightening</span>"
        elif pace["underspending"]:
            pace_msg = f"<span class='alert-amber'>← UNDERSPENDING — bid up</span>"
        else:
            pace_msg = f"<span class='alert-green'>on pace</span>"
        st.markdown(
            f"<div class='status-line'>Avg $/slot needed: <b>${avg_needed:.1f}</b> · "
            f"target avg <b>${avg_target:.1f}</b><br>{pace_msg}</div>",
            unsafe_allow_html=True,
        )

        # Scarcity alerts — only for positions I still need
        st.markdown("<div style='height:0.8em'></div>", unsafe_allow_html=True)
        st.markdown("### Scarcity")
        open_positions = [p for p in ("QB", "RB", "WR", "TE") if me.starter_need(p)]
        if open_positions:
            for pos in open_positions:
                sc = draft.scarcity_status(pos)
                left_ct = sc["above_replacement_left"]
                teams_need = sc["teams_needing_starter"]
                tight = sc["scarcity_tight"]
                if tight:
                    line = (
                        f"<div class='alert-red'>"
                        f"<b>{pos}</b>: {left_ct} left, {teams_need} teams need starter"
                        f"</div>"
                    )
                else:
                    line = f"<div class='status-line'><b>{pos}</b>: {left_ct} left, {teams_need} teams need starter</div>"
                st.markdown(line, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='status-line' style='color:#999'>All starter slots addressed</div>",
                unsafe_allow_html=True,
            )

# ============================================================================
# BOARD TAB — full player table
# ============================================================================
with tab_board:
    drafted_lower = {p.player.lower().strip() for p in draft.all_picks}
    b = board.copy()
    b["drafted"] = b["player_display_name"].str.lower().str.strip().isin(drafted_lower)

    f1, f2, f3 = st.columns([1, 1, 4])
    with f1:
        pos_filter = st.selectbox(
            "Position", ["ALL", "QB", "RB", "WR", "TE", "DEF"], index=0,
        )
    with f2:
        show_drafted = st.checkbox("Show drafted", value=False)

    disp = b.copy()
    if pos_filter != "ALL":
        disp = disp[disp["position"] == pos_filter]
    if not show_drafted:
        disp = disp[~disp["drafted"]]
    disp = disp.sort_values(["par"], ascending=False)

    table = disp[[
        "player_display_name", "position", "team", "proj_ppg",
        "par", "price", "role_tier", "drafted",
    ]].rename(columns={
        "player_display_name": "Player", "position": "Pos", "team": "Team",
        "proj_ppg": "ppg", "par": "PAR", "price": "$", "role_tier": "Role",
        "drafted": "✓",
    })
    st.dataframe(table, width="stretch", hide_index=True, height=650)

# ============================================================================
# SCOUTING TAB
# ============================================================================
with tab_scout:
    st.markdown(load_scouting_md())
