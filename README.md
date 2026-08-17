# DARTS Auction Valuation Model

Auction pricing tool for a 10-team, $200 salary cap, **2QB / no-kicker**, half-PPR redraft league.

## Why this exists

Every published auction value you can find is built for a 1QB league. This league starts two quarterbacks, which moves replacement level at QB from roughly QB13 to roughly QB22. That single difference makes off-the-shelf values structurally wrong for two of your ten starting slots.

## Setup

```bash
pip install pandas pyarrow numpy
python src/build_history.py
```

First run downloads ~8 seasons from nflverse and caches them in `data/raw/`. Later runs read from cache. Pass `refresh=True` to the loaders to re-pull.

## Project layout

```
darts-auction/
├── src/
│   ├── league_config.py   # single source of truth for all league settings
│   ├── scoring.py         # DARTS scoring engine
│   ├── ingest.py          # nflverse pulls + disk cache
│   └── build_history.py   # Phase 1 pipeline
└── data/
    ├── raw/               # cached nflverse parquet files
    └── processed/         # player_seasons.parquet / .csv
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

The $1,850 figure is the denominator for the whole valuation model.

## Output: `player_seasons.parquet`

One row per player-season, 2018–2025. Key fields:

- `fantasy_points`, `ppg` — under DARTS scoring specifically
- `pts_from_td`, `pts_non_td` — TD points isolated, because TD rate is the noisiest input in fantasy and the main driver of false breakouts
- `age` — computed at September 1 of the season, so age cohorts are comparable
- `experience`, `draft_round`, `draft_pick` — draft capital predicts how long a team keeps feeding a struggling player
- `targets_pg`, `carries_pg`, `target_share`, `air_yards_share`, `wopr` — opportunity
- `yards_per_target`, `yards_per_carry`, `catch_rate`, `td_rate` — efficiency

The opportunity/efficiency split is deliberate. Efficiency is player-owned and stable year to year; opportunity is team-owned and resets each offseason. They get modeled separately.

## Design decisions worth knowing

**We compute fantasy points ourselves.** nflverse ships a `fantasy_points_ppr` column. It assumes full PPR and −1 interceptions. This league is half PPR with −2 interceptions, so using that column would misprice every pass-catcher and every turnover-prone QB.

**Regular season only.** Playoff games would inflate totals for players on good teams.

**Per-game rates alongside totals.** Season totals punish injured players twice — once for missing games, once when the model reads the low total as decline.

## Known caveats

1. **Games played is approximated** by counting weeks with a recorded stat line. A healthy player who was active but got zero touches won't be counted. Snap count data fixes this and is a later addition.
2. **No DST scoring yet.** Rules are in `league_config.py` but team defense scoring needs a separate data source.
3. **No 2026 projections yet.** This phase is descriptive history only.
4. **Offensive fumble return TDs** are in the league rules but not in the data. They're vanishingly rare and are ignored.

## Roadmap

- [x] Phase 1 — historical database under exact league scoring
- [ ] Phase 2 — age curves and trajectory detection by position
- [ ] Phase 3 — 2026 opportunity layer (depth charts, coordinator changes, target competition)
- [ ] Phase 4 — replacement level and dollar values from the $1,850 discretionary pool
- [ ] Phase 5 — Streamlit front end with player swapping and live max-bid calculation
- [ ] Phase 6 — live draft tracking with inflation adjustment
