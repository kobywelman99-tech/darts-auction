"""
DARTS league configuration — the single source of truth.

Every downstream calculation reads from here. If a league setting changes,
change it in this file only and rerun the pipeline.
"""

LEAGUE_NAME = "DARTS"
LEAGUE_ID = 207758
SEASON = 2026

# ---------------------------------------------------------------------------
# Auction economics
# ---------------------------------------------------------------------------
NUM_TEAMS = 10
BUDGET_PER_TEAM = 200
MIN_BID = 1
FAAB_BUDGET = 100

# ---------------------------------------------------------------------------
# Roster construction
# Starters: QB QB WR WR WR RB RB TE W/R/T DEF  = 10
# Bench: 5   |   IR: 2 (does not consume draft capital)
# ---------------------------------------------------------------------------
STARTERS = {
    "QB": 2,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "FLEX": 1,   # W/R/T
    "DEF": 1,
}
FLEX_ELIGIBLE = ("RB", "WR", "TE")
BENCH_SPOTS = 5
IR_SPOTS = 2

# NOTE: no kicker in this league.

ROSTER_SIZE = sum(STARTERS.values()) + BENCH_SPOTS          # 15
TOTAL_PLAYERS_DRAFTED = ROSTER_SIZE * NUM_TEAMS            # 150
TOTAL_MONEY = BUDGET_PER_TEAM * NUM_TEAMS                  # 2000
MONEY_LOCKED_IN_MINIMUMS = TOTAL_PLAYERS_DRAFTED * MIN_BID  # 150
DISCRETIONARY_MONEY = TOTAL_MONEY - MONEY_LOCKED_IN_MINIMUMS  # 1850
MAX_OPENING_BID = BUDGET_PER_TEAM - (ROSTER_SIZE - 1) * MIN_BID  # 186

REGULAR_SEASON_WEEKS = 14
PLAYOFF_WEEKS = (15, 16, 17)

# ---------------------------------------------------------------------------
# Offensive scoring
# ---------------------------------------------------------------------------
SCORING = {
    "passing_yards_per_point": 25.0,
    "passing_td": 4.0,
    "interception": -2.0,          # league custom (Yahoo default is -1)
    "rushing_yards_per_point": 10.0,
    "rushing_td": 6.0,
    "reception": 0.5,              # half PPR
    "receiving_yards_per_point": 10.0,
    "receiving_td": 6.0,
    "return_td": 6.0,
    "two_point_conversion": 2.0,
    "fumble_lost": -2.0,
    "offensive_fumble_return_td": 6.0,
}

# ---------------------------------------------------------------------------
# Defense / Special Teams scoring (used in a later phase)
# ---------------------------------------------------------------------------
DST_SCORING = {
    "sack": 1.0,
    "interception": 2.0,
    "fumble_recovery": 2.0,
    "touchdown": 6.0,
    "safety": 2.0,
    "block_kick": 2.0,
    "return_td": 6.0,
    "extra_point_returned": 2.0,
}

DST_POINTS_ALLOWED = [
    (0, 0, 10.0),
    (1, 6, 7.0),
    (7, 13, 4.0),
    (14, 20, 1.0),
    (21, 27, 0.0),
    (28, 34, -1.0),
    (35, 999, -4.0),
]

# ---------------------------------------------------------------------------
# Derived scarcity notes
# ---------------------------------------------------------------------------
# Two starting QBs across 10 teams means 20 QBs start every week, so
# replacement level at QB sits far deeper than in a standard 1QB league.
# This is computed properly in the valuation phase, not hardcoded.

HISTORY_START_SEASON = 2018
HISTORY_END_SEASON = 2025
