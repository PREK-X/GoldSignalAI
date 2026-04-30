# CLAUDE.md
# Read by Claude Code at session start
# For duplicate facts, canonical source is: CLAUDE.md

---

## What This Is

GoldSignalAI — AI-powered XAU/USD trading signal bot targeting
FundedNext 1-Step $10k prop firm challenge. Signals via Discord,
persistence in SQLite, Streamlit dashboard.
Built by Masab. Architecture and current state -> see CONTEXT.md
and REFERENCES.md.

---

## Developer Environment

- **OS:** Arch Linux
- **Python:** 3.12 (via AUR — 3.14 too new for some packages)
- **Venv:** `~/Documents/projects/GoldSignalAI/venv/`
- **Always use:** `venv/bin/python` (never `python` or `python3`)
- **GitHub:** github.com/PREK-X/GoldSignalAI (Private)

---

## Run Commands

```bash
# Health check
venv/bin/python main.py --health-check

# Live bot
venv/bin/python main.py

# Backtest
venv/bin/python -m backtest.engine

# Dashboard
venv/bin/python -m streamlit run dashboard/app.py

# Tests (160/161 pass — 1 pre-existing DST failure)
venv/bin/python -m pytest tests/ -v
```

---

## Environment Variables (.env)

```
POLYGON_API_KEY=       # Primary data (2yr M15)
DISCORD_WEBHOOK_URL=   # Primary alerts
TELEGRAM_BOT_TOKEN=    # Backup alerts (PK connectivity issues)
TELEGRAM_CHAT_ID=      # Telegram chat
SENTRY_DSN=            # Error monitoring (optional)
```

---

## Live Bot Behavior

- Signal loop every 15 min (M15 candle close)
- Fires during 13:00-21:59 UTC (NY session; 9AM-5PM EST = 6PM-2AM PKT)
- Discord alert when signal passes all gates
- Dedup: same direction skipped for 4 hours
- Expected frequency: ~1 signal per 8 days (87 trades / 2yr)

---

## Polygon Fetch Limits

- M15: `bars=47000` | H1: `bars=12000`
- **DO NOT** request more — hangs on pagination
- Sequential fetch (not parallel) — 429 rate limit on free tier

---

## Stage 17 frequency sweep (2026-04-20)

Goal: find a config meeting trades≥500, PF≥1.50, WR≥60%, FN per-challenge buffer≥1.30×, all 8 prop firms PASS.

**Result: 0 / 25 rows pass all 4 gates.** Architectural cap from CONTEXT.md confirmed — `trades ≥ 500` is the binding constraint. Most-permissive row (`D2_C58_noncontradict_S12-22`) reaches 493 trades but PF collapses to 1.09.

Cache: `data/historical/stage17_sweep_cache.pkl` (M15+H1 frozen). Per-row results: `data/historical/stage17_sweep_results.json`. Harness: `backtest/sweep_stage17.py`.

| #  | MinDom | MinConf | H1Mode        | Sess  | Trd | WR    | PF   | DD    | FN-DD | Buf  | 8Frm | Gate |
|----|--------|---------|---------------|-------|-----|-------|------|-------|-------|------|------|------|
|  1 |   2    |   58    | noncontradict | 12-22 | 493 | 55.6% | 1.09 |12.35% | 4.21% | 1.42 | 8/8  | FAIL |
|  2 |   2    |   58    | noncontradict | 13-21 | 418 | 58.6% | 1.27 |10.23% | 4.75% | 1.26 | 8/8  | FAIL |
|  3 |   2    |   58    | agree         | 12-22 | 144 | 63.9% | 1.29 |10.82% | 3.21% | 1.87 | 8/8  | FAIL |
|  4 |   2    |   58    | agree         | 13-21 | 125 | 67.2% | 1.50 | 8.83% | 1.25% | 4.80 | 8/8  | FAIL |
|  5 |   2    |   60    | noncontradict | 12-22 | 493 | 55.6% | 1.09 |12.35% | 4.21% | 1.42 | 8/8  | FAIL |
|  6 |   2    |   60    | noncontradict | 13-21 | 418 | 58.6% | 1.27 |10.23% | 4.75% | 1.26 | 8/8  | FAIL |
|  7 |   2    |   60    | agree         | 12-22 | 142 | 64.1% | 1.30 |12.37% | 1.25% | 4.80 | 8/8  | FAIL |
|  8 |   2    |   60    | agree         | 13-21 | 124 | 67.7% | 1.50 | 8.87% | 1.30% | 4.62 | 8/8  | FAIL |
|  9 |   2    |   62    | noncontradict | 12-22 | 382 | 54.2% | 0.94 |20.00% | 3.64% | 1.65 | 8/8  | FAIL |
| 10 |   2    |   62    | noncontradict | 13-21 | 327 | 57.2% | 1.12 | 9.29% | 3.87% | 1.55 | 8/8  | FAIL |
| 11 |   2    |   62    | agree         | 12-22 | 115 | 60.9% | 1.20 |15.27% | 1.63% | 3.69 | 8/8  | FAIL |
| 12 |   2    |   62    | agree         | 13-21 | 101 | 64.4% | 1.21 |14.59% | 6.46% | 0.93 | 6/8  | FAIL |
| 13 |   3    |   58    | noncontradict | 12-22 | 493 | 55.6% | 1.09 |12.35% | 4.21% | 1.42 | 8/8  | FAIL |
| 14 |   3    |   58    | noncontradict | 13-21 | 418 | 58.6% | 1.27 |10.23% | 4.75% | 1.26 | 8/8  | FAIL |
| 15 |   3    |   58    | agree         | 12-22 | 144 | 63.9% | 1.29 |10.82% | 3.21% | 1.87 | 8/8  | FAIL |
| 16 |   3    |   58    | agree         | 13-21 | 125 | 67.2% | 1.50 | 8.83% | 1.25% | 4.80 | 8/8  | FAIL |
| 17 |   3    |   60    | noncontradict | 12-22 | 493 | 55.6% | 1.09 |12.35% | 4.21% | 1.42 | 8/8  | FAIL |
| 18 |   3    |   60    | noncontradict | 13-21 | 418 | 58.6% | 1.27 |10.23% | 4.75% | 1.26 | 8/8  | FAIL |
| 19 |   3    |   60    | agree         | 12-22 | 142 | 64.1% | 1.30 |12.37% | 1.25% | 4.80 | 8/8  | FAIL |
| 20 |   3    |   60    | agree         | 13-21 | 124 | 67.7% | 1.50 | 8.87% | 1.30% | 4.62 | 8/8  | FAIL |
| 21 |   3    |   62    | noncontradict | 12-22 | 382 | 54.2% | 0.94 |20.00% | 3.64% | 1.65 | 8/8  | FAIL |
| 22 |   3    |   62    | noncontradict | 13-21 | 327 | 57.2% | 1.12 | 9.29% | 3.87% | 1.55 | 8/8  | FAIL |
| 23 |   3    |   62    | agree         | 12-22 | 115 | 60.9% | 1.20 |15.27% | 1.63% | 3.69 | 8/8  | FAIL |
| 24 |   3    |   62    | agree         | 13-21 | 101 | 64.4% | 1.21 |14.59% | 6.46% | 0.93 | 6/8  | FAIL |
| 25 |   3    |   65    | agree         | 13-21 |  88 | 68.2% | 1.66 | 9.88% | 3.43% | 1.75 | 8/8  | CTRL |

**Top-3 Pareto** (by `trades × PF × WR × buffer` since no row passes all 4 gates):

| # | Config                  | Trades | WR    | PF   | Buf  | 8Firms |
|---|-------------------------|--------|-------|------|------|--------|
| 1 | D2_C58_agree_S13-21     | 125    | 67.2% | 1.50 | 4.80×| 8/8    |
| 2 | D3_C58_agree_S13-21     | 125    | 67.2% | 1.50 | 4.80×| 8/8    |
| 3 | D2_C60_agree_S13-21     | 124    | 67.7% | 1.50 | 4.62×| 8/8    |

(Top-1 vs top-2 differ only in MIN_DOMINANT — gate is inert at this confidence level. Effectively ~2 distinct configs.)

**Outcome — config kept at control (D3_C65_agree_S13-21):** Stage 1 fresh-data
verification of D3_C58 produced 127 trd / PF 1.29 / FN buffer 1.33× — far below
sweep's PF 1.50 / buffer 4.80× due to 11-day data drift between Apr-19 cache and
Apr-30 fetch. Reverted MIN_CONFIDENCE_PCT 65→58→65. Kept the harness
(`backtest/sweep_stage17.py`), the verifier (`backtest/verify_stage1.py`),
the HMM degenerate-cluster fix (`analysis/regime_filter.py`), and the
MIN_ACTIVE/MIN_DOMINANT module-scope lift (`analysis/scoring.py`).

**HMM stability fix (regime_filter.py):** non-stationary input data could leave
the lowest-vol cluster with <3% mass; the old vol-ascending labeling then mapped
the dominant cluster (~90% mass) to RANGING, blocking all signals. Fix detects
clusters with mass < 3% and reassigns them to CRISIS, restoring TRENDING for the
dominant cluster. Logged as warning when triggered.

**Observations:**
- `MIN_DOMINANT ∈ {2, 3}` produces **identical** results in every row — gate is dominated by other filters at C58/C60. Lifting it (Stage 17 prep, scoring.py:60) gave no degree of freedom.
- `MIN_CONFIDENCE_PCT ∈ {58, 60}` likewise inert — same outputs. Filter only bites at `≥ 62`.
- `H1_FILTER_MODE`: `noncontradict` ~3.5× the trade volume of `agree` (493 vs 144) but cuts PF below 1.10.
- `SESSION_ACTIVE_HOURS`: widening to 12-22 adds ~15-20% trades but adds the low-quality 12 UTC London hour (CONTEXT.md says 33.9% WR there).
- Pareto candidates beat current control on **trades (+42%) and FN buffer (+174%)** at modest PF cost (1.50 vs 1.66) and similar 8/8 firm pass.

---

## Integration Gaps

| File               | Gap                                    | Priority |
|--------------------|----------------------------------------|----------|
| analysis/scoring.py| MIN_ACTIVE, MIN_DOMINANT, SESSION_HOURS at module scope (Stage 17) but still not in config.py | Low — move for Stage 9 multi-asset |

signals/generator.py MetaDecision gap: **FIXED** (Stage 11).

---

## Code Navigation

**This project has a code-review-graph knowledge graph.**
Use MCP tools BEFORE Grep/Glob/Read:

| Tool                    | Use when                          |
|-------------------------|-----------------------------------|
| `semantic_search_nodes` | Finding functions/classes          |
| `get_impact_radius`     | Before editing any module          |
| `detect_changes`        | Reviewing code changes             |
| `get_review_context`    | Token-efficient source snippets    |
| `get_affected_flows`    | Which execution paths impacted     |
| `query_graph`           | Callers, callees, imports, tests   |
| `get_architecture_overview` | High-level structure           |

Fall back to Grep/Glob/Read only when graph doesn't cover it.

---

## Obsidian Knowledge Base

Detailed references moved to Obsidian vault under `GoldSignalAI/` to reduce token load.
Query via MCP when needed: file map, indicators, signal flow, config values,
prop firm rules, backtest history, stages, decision log, critical bugs.

---

## Cross-References

- File map, indicator table, ML models, data sources, prop firm
  limits, signal flow, meta-decision rules -> **REFERENCES.md**
- Backtest history, config values, prop sim results, forward test
  status, known issues, disabled features, stages -> **CONTEXT.md**
