# CLAUDE.md

Context for Claude Code working in this repository.

## What this project is

An auction valuation model for a specific fantasy football league. The owner is building it partly as a working tool and partly to learn — so **explain reasoning as you go, and prefer clarity over cleverness in the code.** When there's a modeling choice with real tradeoffs, surface it rather than silently picking one.

Draft date: approximately **September 1–2, 2026**. The tool needs to be usable by then.

## The league (DARTS, Yahoo ID 207758)

10 teams, $200 salary cap auction, **redraft — no keepers**, half PPR, head-to-head.

**Roster (15 draftable spots):** QB, QB, WR, WR, WR, RB, RB, TE, W/R/T, DEF + 5 bench + 2 IR.

Two things make this league unusual and they drive most of the modeling:
1. **Two starting QBs** — 20 QBs start weekly out of ~32 viable. Replacement level sits near QB22, not QB13.
2. **No kicker.**

**Scoring:** passing 25 yd/pt, 4 pt pass TD, **−2 INT** (Yahoo default is −1), rushing/receiving 10 yd/pt, 6 pt rush/rec TD, 0.5 PPR, −2 fumble lost. Rushing yards are worth 2.5× passing yards per yard, so mobile QBs carry a large premium here.

All settings live in `src/league_config.py`. **Never hardcode league values elsewhere** — import from there.

## Auction math

| Quantity | Value |
|---|---|
| Players drafted leaguewide | 150 |
| Total money | $2,000 |
| Locked in $1 minimums | $150 |
| **Discretionary money** | **$1,850** |
| Max opening bid | $186 |

Dollar values must ultimately sum to the $1,850 discretionary pool, not $2,000. This is the most common error in auction value calculations.

## Modeling philosophy

Fantasy production = **opportunity × efficiency**, and the two come from different places:

- **Efficiency** (yards per target, YPC, catch rate) is player-owned and reasonably stable year over year. Historical data estimates it well.
- **Opportunity** (targets, carries, snap share, red zone touches) is team-owned and resets each offseason. Historical data knows nothing about it — it requires current depth chart and coaching information.

Keep these layers separate in code. When a projection calls wrong, we need to know which layer caused it.

**Touchdown rate is the noisiest input in fantasy football** and the main cause of false breakouts. `pts_from_td` is isolated in the dataset specifically so it can be regressed toward positional mean.

## Current state

**Phase 1 is complete.** `data/processed/player_seasons.parquet` holds 4,768 player-seasons (2018–2025) scored under exact league rules, with age, experience, draft capital, usage rates, and efficiency metrics.

Validated against 2025: McCaffrey 366, Josh Allen 365, Drake Maye 352, Jonathan Taylor 339, Puka Nacua 311, Trey McBride 253.

Key finding already: QB replacement level is 12.2 ppg in this league vs 16.1 ppg standard. That 3.9 ppg gap is roughly 66 points per season per QB slot that published values fail to credit — across two slots.

## Roadmap

- [x] **Phase 1** — historical database under exact league scoring
- [ ] **Phase 2** — age curves and trajectory detection by position
- [ ] **Phase 3** — 2026 opportunity layer (depth charts, coordinator changes, target competition)
- [~] **Phase 4** (scarcity baseline done in `src/scarcity.py`) — replacement level + dollar values from the $1,850 pool
- [ ] **Phase 5** — Streamlit front end, player swapping, live max-bid calculation
- [ ] **Phase 6** — live draft tracking with inflation adjustment

## Conventions

- Data pulls go through `src/ingest.py` and cache to `data/raw/`. Don't re-download inside loops.
- Scoring goes through `src/scoring.py`. Never use nflverse's built-in `fantasy_points_ppr` — it assumes full PPR and −1 INT, which misprices this league.
- Regular season only (`season_type == "REG"`).
- Compare players on per-game rates, not season totals. Totals punish injured players twice.
- Age is computed at September 1 of the season so cohorts are comparable.

## Known caveats

1. **Games played is approximated** by counting weeks with a recorded stat line. An active player with zero touches isn't counted. Snap count data would fix this.
2. **No DST scoring implemented.** Rules are in `league_config.py`; team defense needs a separate data source.
3. **Model knowledge of the 2026 offseason runs through roughly May.** Training camp (June–August) is not included. Pull current news before finalizing any values — depth chart battles that look open in the data may have resolved in July.
4. **Offensive fumble return TDs** are in the rules but not in the data. Rare; ignored.

## Data source

nflverse GitHub releases. Weekly stats use the `stats_player` release (`stats_player_week_{season}.parquet`), which covers 2018–2025 with a consistent 150-column schema. The older `player_stats` release has a different naming convention and does not include 2025 — don't use it.

## Phase 3 — opportunity layer (in progress)

The opportunity layer is a **hand-maintained CSV**, not code: `data/opportunity/{pos}_2026.csv`.
Depth charts change daily until kickoff; a CSV can be edited in seconds, code cannot.

```bash
python src/opportunity.py --skeleton RB   # generate starter file (won't overwrite)
python src/opportunity.py --audit RB      # what still needs filling
```

Assign a `role_tier` from current news and the numeric shares fill in from defaults.
Override any numeric column directly when you know better than the default.

**2026 context that makes this phase unusually important:** 10 teams changed head coaches
and 21 of 32 hired a new offensive coordinator. Play-callers decide who gets screens,
checkdowns and goal-line work, so 2025 usage is a weaker prior than in a normal year.

**Known roster moves already seeded (as of 2026-08-18):** Kenneth Walker to KC,
Rico Dowdle to PIT, Rachaad White to WAS, Etienne gone from JAX, Dowdle gone from CAR.
Team columns in the skeleton are 2025 teams and must be verified individually.

**`job_security` is not a tiebreaker, it's a risk discount.** A 50%-chance-of-a-big-role
player is worth less than a certain-smaller-role player with identical expected touches,
because an auction bust costs the roster spot as well as the money.

### Phase 3 caveats
- Rookies have no history, so they are absent from generated skeletons and must be added by hand.
- Backtest MAE (2.22 ppg) is measured on players who qualified in BOTH seasons. Players who
  lost their job entirely are excluded, so the real error is worse than measured and the
  age curve should carry a bust-risk flag, not just a point estimate.

## Phase 4.5 — market prices (added to roadmap)

Model value alone cannot drive bidding. The edge is the GAP between projected value and
market price. Pull Yahoo average auction values for this league format and compute
value-minus-price per player. A $40 player bought at $38 earns nothing; a $22 player
bought at $9 wins the draft.

### Phase 3 RB status (updated 2026-08-18)
50 of 163 RB rows filled — covers every back with realistic auction value.
The remaining ~113 are deep bench/practice-squad names that will go for $1 or undrafted.

**Late-breaking camp items that changed earlier entries:**
- Chuba Hubbard: hamstring, weeks-long absence. Jonathon Brooks rising fast in CAR.
- Rico Dowdle now reported as PIT's primary runner slightly AHEAD of Jaylen Warren
  (McCarthy favorite from their Dallas years). Reverses the earlier even-split entry.
- Josh Jacobs: May arrest, investigation ongoing. Role is safe if he plays — availability is the risk.
- Achane lost both Tua and Mike McDaniel; Miami's offense is materially different.
- Montgomery traded DET->HOU and is reported as Houston's featured back.
- Pacheco is now a Gibbs handcuff in Detroit, not a starter.

These are the volatile rows. Re-check in the final week before the draft.

### Phase 3 WR/TE status (2026-08-18)
TE: 24/127 filled. WR: 49/260 filled. Covers everyone with realistic auction value.

**STRUCTURAL SHIFT THAT SUPPORTS THE WR-OVERSPEND / TE-UNDERSPEND THESIS**
2025 league-wide data: TE target share hit 23.8% — the highest rate ever recorded — and
TEs caught 231 TDs, the most since 2013. Meanwhile WR combined target share fell to 57.9%,
the LOWEST since 2017 and 2+ points below 2024. Only 19 WRs averaged 7+ targets/game in
2025, down from 29 in both 2024 and 2023. Teams are shifting to 12/13 personnel.
Targets are structurally moving from WR to TE. Our league still prices the old distribution.

**COUNTER-CAVEAT — DO NOT IGNORE**
15 TEs averaged 8+ half-PPR ppg in 2025, up from 10 in 2024. TE depth is INCREASING, which
means our TE16 replacement level of 6.3 ppg (computed on 2022-2025) is probably TOO LOW for
2026. A rising TE replacement level shrinks the TE edge. Before finalizing values, recompute
TE replacement on 2025 alone and compare. If TE16 is now materially above 6.3, elite TEs are
worth less than our current model says, and the correct play shifts from "buy an elite TE"
toward "wait and take TE8-12 cheap."

**Key TE injury/situation flags:** Kittle torn Achilles (age 33, may miss start).
Kraft knee but ahead of schedule with Doubs AND Wicks gone from GB. LaPorta has a new OC
(Drew Petzing) who turned McBride into the overall TE1. Pitts has a new TE-friendly OC.

**Key WR moves:** A.J. Brown to NE, DJ Moore to BUF, Michael Pittman to PIT,
Wan'Dale Robinson to TEN (rookie Carnell Tate drafted 4th overall there),
Jaylen Waddle to DEN. Nabers returning from ACL but healthy in camp.
