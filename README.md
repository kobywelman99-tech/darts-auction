# DARTS Auction Valuation Model

Auction pricing tool for a 10-team, $200 salary cap, **2QB / no-kicker**, half-PPR redraft league.

## Why this exists

Every published auction value you can find is built for a 1QB league. This league starts two quarterbacks, which moves replacement level at QB from roughly QB13 to roughly QB22. That single difference makes off-the-shelf values structurally wrong for two of your ten starting slots. The model also accounts for −2 INT (not the Yahoo default of −1) and the half-PPR / no-kicker roster construction, which shifts money away from pass-catchers and toward QBs relative to any generic tool.

## Current state (as of 2026-08-26)

The pipeline is complete end-to-end:

- **Historical database** — 4,768 player-seasons (2018–2025) scored under exact DARTS rules. Validated: McCaffrey 366, Josh Allen 365, Drake Maye 352, Jonathan Taylor 339, Puka Nacua 311, Trey McBride 253.
- **Replacement level + dollar values** — positional scarcity baked in. QB replacement sits at 12.2 ppg vs the standard-league assumption of 16.1 ppg — a 66-point-per-season premium per QB slot that published values miss entirely.
- **2026 opportunity layer** — hand-maintained depth charts in `data/opportunity/`. Covers every player with realistic auction value at RB, WR, TE. Wired into the valuation pipeline.
- **Mock draft simulator** — 50-seed validation against a calibrated opponent model (matched to real DARTS 2024–2025 spending patterns). Results: median rank **2nd of 10**, TE shutout **0%**, all starting slots filled **84%** of seeds.
- **Opponent scouting report** — `data/drafts/scouting_report.md` profiles all 10 managers from 2024–2025 actual draft data: spending shape by nomination window, confirmed positional leans, and a TE competition map.

## Setup

```bash
pip install pandas pyarrow numpy
python src/build_history.py   # downloads ~8 seasons from nflverse, caches to data/raw/
```

Later runs read from cache. The opportunity layer CSVs in `data/opportunity/` are hand-maintained — edit them directly before each draft.

## Project layout

```
darts-auction/
├── src/
│   ├── league_config.py   # single source of truth for all league settings
│   ├── scoring.py         # DARTS scoring engine
│   ├── ingest.py          # nflverse pulls + disk cache
│   ├── build_history.py   # Phase 1 pipeline
│   ├── scarcity.py        # replacement level + PAR calculation
│   ├── valuation.py       # dollar values from the $1,850 pool
│   ├── opportunity.py     # 2026 depth chart layer
│   ├── draft_state.py     # live draft assistant (ceilings, inflation, scarcity)
│   ├── inflation.py       # mid-draft inflation multiplier
│   └── mock_draft.py      # auction simulator + distribution analysis
└── data/
    ├── raw/               # cached nflverse parquet files
    ├── processed/         # player_seasons.parquet
    ├── opportunity/       # rb_2026.csv, wr_2026.csv, te_2026.csv
    └── drafts/            # draft_2024.csv, draft_2025.csv, scouting_report.md
```

## Auction constants (derived in `league_config.py`)

| Quantity | Value |
|---|---|
| Roster size (draftable) | 15 |
| Players drafted leaguewide | 150 |
| Total money in room | $2,000 |
| Locked in $1 minimums | $150 |
| **Discretionary money** | **$1,850** |
| Max opening bid | $186 |

The $1,850 figure is the denominator for the whole valuation model. Dollar values sum to $1,850, not $2,000 — the most common error in auction value calculations.

## Key design decisions

**We compute fantasy points ourselves.** nflverse ships a `fantasy_points_ppr` column. It assumes full PPR and −1 interceptions. This league is half PPR with −2 interceptions, so using that column would misprice every pass-catcher and every turnover-prone QB.

**Opportunity and efficiency are modeled separately.** Efficiency (yards per target, YPC, catch rate) is player-owned and stable year over year. Opportunity (targets, carries, snap share) is team-owned and resets each offseason — it lives in the hand-maintained CSVs, not in historical data.

**TD points are isolated.** `pts_from_td` is a separate field so touchdown rate (the noisiest input in fantasy) can be regressed toward positional mean rather than taken at face value.

**Regular season only.** Playoff games inflate totals for players on good teams.

**Per-game rates alongside totals.** Season totals punish injured players twice — once for missing games, once when the model reads the low total as decline.

## Known caveats

1. **Games played is approximated** by counting weeks with a recorded stat line. A healthy player who was active but got zero touches won't be counted. Snap count data fixes this and is a later addition.
2. **No DST scoring.** Rules are in `league_config.py` but team defense scoring needs a separate data source. DEF slots are pre-assigned at $1 in the simulator.
3. **Offensive fumble return TDs** are in the league rules but not in the data. They're vanishingly rare and are ignored.
4. **Opportunity layer knowledge cutoff ~August 2026.** Re-verify volatile rows (injury reports, camp depth chart battles) in the week before the draft.

## Roadmap

- [x] Phase 1 — historical database under exact league scoring
- [x] Phase 2 — age curves and trajectory detection by position
- [x] Phase 3 — 2026 opportunity layer (depth charts, coordinator changes, target competition)
- [x] Phase 4 — replacement level and dollar values from the $1,850 discretionary pool
- [x] Phase 4.5 — mock draft simulator + opponent model calibrated to real DARTS data
- [x] Phase 4.6 — opponent scouting report from 2024–2025 actual drafts
- [ ] Phase 5 — Streamlit front end with player swapping and live max-bid calculation
- [ ] Phase 6 — live draft tracking with inflation adjustment
