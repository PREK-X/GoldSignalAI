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
- Fires during 12:00-22:59 UTC (8AM-7PM EST = 5PM-4AM PKT)
- Discord alert when signal passes all gates
- Dedup: same direction skipped for 4 hours
- Expected frequency: ~1 signal per 7 days (103 trades / 2yr)

---

## Polygon Fetch Limits

- M15: `bars=47000` | H1: `bars=12000`
- **DO NOT** request more — hangs on pagination
- Sequential fetch (not parallel) — 429 rate limit on free tier

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
