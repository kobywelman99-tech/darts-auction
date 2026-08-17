# AGENTS.md

Project context lives in **[CLAUDE.md](./CLAUDE.md)** — read it first.

Quick orientation:
- Fantasy football auction valuation model for a 10-team, $200 cap, **2QB / no-kicker**, half-PPR redraft league.
- All league settings live in `src/league_config.py`. Never hardcode league values elsewhere.
- Never use nflverse's `fantasy_points_ppr` column — it assumes full PPR and -1 INT; this league is half PPR with -2 INT.
- Dollar values must sum to the **$1,850 discretionary pool**, not $2,000.
- The owner is learning as he builds. Explain your reasoning and surface modeling tradeoffs rather than silently picking one.

Phase 1 (historical database) is complete. Phase 2 is age curves and trajectory detection.
