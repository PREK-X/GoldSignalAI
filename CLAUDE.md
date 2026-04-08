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

# Tests (159/161 pass — 2 pre-existing failures)
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
- Only fires during NY session (13:00-22:00 UTC = 6PM-1AM PKT)
- Discord alert when signal passes all gates
- Dedup: same direction skipped for 4 hours
- Expected frequency: ~1 signal per 6 days

---

## Polygon Fetch Limits

- M15: `bars=47000` | H1: `bars=12000`
- **DO NOT** request more — hangs on pagination
- Sequential fetch (not parallel) — 429 rate limit on free tier

---

## Key Decisions and Why

- **Active-ratio scoring:** old `/10` made 70% unreachable.
  New: `dominant / (bull + bear)` ignores neutrals
- **NY session only:** diagnostic 277 signals: NY 63.3% WR,
  London 33.9%. Session filter is the single biggest edge
- **ML disabled:** XGBoost 47% (trained on indicator outputs =
  redundant). Need independent features -> macro pipeline built
- **SL = ATR x 1.5 (~130 pips):** gold M15 median candle = 125
  pips. Old 30-pip SL = noise stop-out every trade
- **9 indicators FROZEN:** adding 4 more in Stage 2 dropped
  PF 1.23 -> 0.90. Do not add without per-indicator backtest
- **RANGING blocked (not reduced):** RANGING trades avg $+17.87
  vs $+81.64 TRENDING, with disproportionate DD
- **FN daily ceiling 2.8%:** pre-emptive block below 3.0% hard
  limit. Dropped max daily loss from 3.00% to 2.13%
- **38% base WR is fine:** with 3.3:1 R:R, break-even is 23%
- **PrecomputedIndicators:** computing 12 indicators per bar on
  48k bars takes hours without the shim in indicators.py

---

## Critical Bugs (Historical)

| Bug                          | Impact            | Fix                    |
|------------------------------|-------------------|------------------------|
| 70% confidence unreachable   | 0 signals ever    | Active-ratio scoring   |
| SL capped at 30 pips        | Every trade hit SL| 50-200 pips ATR-based  |
| BBands in scoring            | 42.3% accuracy    | Removed from voting    |
| London session trading       | 33.9% WR          | NY-only session filter |
| yfinance 60-day limit        | Invalid backtest  | Polygon.io added       |
| H1 resampled from M15       | Wrong H1 values   | Separate H1 fetch      |
| ML blocking good signals     | PF degraded       | USE_ML_FILTER=False    |
| MIN_ACTIVE=3                 | PF->1.08, DD 15%  | Reverted to 4          |
| Backtest hang on 48k bars    | Never completes   | PrecomputedIndicators  |
| Polygon 5yr fetch            | Timeout/hang      | bars=47000 cap         |
| Stage 2 indicators           | PF->0.90          | Reverted to commit 88c1496 |
| Parallel Polygon 429         | Both fetches fail  | Sequential + 300s timeout |
| Telegram blocked in PK       | No alerts          | Discord webhook        |
| LGBM macro merge bug         | 0 samples          | Set index.name in features.py |

---

## Integration Gaps

| File               | Gap                                    | Priority |
|--------------------|----------------------------------------|----------|
| analysis/scoring.py| MIN_ACTIVE, MIN_DOMINANT, SESSION_HOURS hardcoded (not in config.py) | Low — move for Stage 9 multi-asset |

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

## Cross-References

- File map, indicator table, ML models, data sources, prop firm
  limits, signal flow, meta-decision rules -> **REFERENCES.md**
- Backtest history, config values, prop sim results, forward test
  status, known issues, disabled features, stages -> **CONTEXT.md**
