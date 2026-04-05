# GoldSignalAI — XAU/USD Algorithmic Trading Signal Bot

15-minute signal bot for Gold (XAU/USD), trading the NY session only, combining 9 voted indicators + GaussianHMM regime detection + LightGBM macro filter + news/ATR filter into a 5-rule meta-decision cascade. Signals fire via Discord. Built to pass the FundedNext 1-Step $10k prop firm challenge.

---

## Performance (Backtested Apr 2024 – Mar 2026)

| Metric | Value |
|--------|-------|
| Profit Factor | 2.15 |
| Max Drawdown | 4.99% |
| Win Rate | 72.0% |
| Sharpe Ratio | 5.31 |
| Trades | 75 |
| Prop firm simulations | All 8 PASS (FTMO, FN 1-Step, FN 2-Step, The5ers, E8, MFF, Apex, Custom) |

---

## Setup

```bash
git clone https://github.com/PREK-X/GoldSignalAI
cd GoldSignalAI

bash deploy/setup_arch.sh        # Arch Linux
# OR
bash deploy/setup_vps.sh         # Ubuntu/Debian VPS (also installs systemd service)

cp deploy/.env.template .env
# Fill in POLYGON_API_KEY and DISCORD_WEBHOOK_URL
```

---

## How to Run

```bash
venv/bin/python main.py                              # live bot
venv/bin/python -m backtest.engine                   # backtest
venv/bin/python main.py --health-check               # health check
venv/bin/python -m streamlit run dashboard/app.py    # dashboard (localhost:8501)
venv/bin/python -m pytest tests/ -v                  # tests (159/161 passing)
```

> **Always use `venv/bin/python`** — never bare `python` or `python3`.

---

## Architecture

- **Data:** Polygon.io 2yr M15/H1, yfinance macro (DXY/VIX/US10Y → SQLite cache)
- **Indicators:** 9 voted — EMA (20/50/200), ADX-14, Ichimoku Cloud, RSI-14, MACD (12,26,9), Stochastic (14,3,3), CCI-20, ATR-14 (SL sizing only), Volume surge
- **Regime:** GaussianHMM 3-state on H1 (TRENDING / RANGING / CRISIS) — hard gate
- **ML:** LightGBM soft vote on 24 macro/statistical features (52% CV, informational)
- **Meta-decision:** 5-rule cascade — HMM gate → LGBM vote → confidence ±5% → session loss circuit → news/ATR filter
- **Risk:** ATR-based SL (1.5×, 50–200 pips), 1:2/1:3 R/R, Half-Kelly sizing
- **Compliance:** 2.8% daily ceiling (FundedNext), 6% trailing DD halt, challenge state persisted
- **Alerts:** Discord webhook (primary), Telegram (backup)
- **Dashboard:** Streamlit (localhost:8501) — equity curve, ML status, regime timeline, challenge gauges

Signal flow per M15 candle close:
```
Polygon M15 + H1 → 9-indicator voting → session gate (13:00–21:59 UTC)
→ 5-rule meta-decision → ATR risk sizing → dedup check → Discord + SQLite
```

---

## Current Phase

**Forward testing** — 20 trades on IC Markets demo (account 52791555, ICMarketsGlobal-Demo).
Signals sent via Discord. Trades placed manually on MT5 mobile. Outcomes tracked in Google Sheet.
Bot auto-stops at trade 20 and sends a Discord completion alert (`FORWARD_TEST_MODE=True`).

FundedNext $10k challenge begins after 20 demo trades are logged.

---

## Prop Firm Target

**FundedNext Stellar 1-Step — $10,000**

| Rule | Limit | Bot action |
|------|-------|-----------|
| Profit target | 10% ($1,000) | — |
| Max daily loss | 3% ($300) | Pre-emptive block at 2.8%; hard halt at 3% |
| Max total drawdown | 6% ($600) trailing | Auto-pause at 5%; hard halt at 6% |
| Min trading days | None (1-Step) | — |

Other presets available: FTMO, FundedNext 2-Step, The5%ers, E8 Funding, MyForexFunds, Apex, Custom. Change `ACTIVE_PROP_FIRM` in `config.py`.

---

## Environment

- **OS:** Arch Linux (dev) / Ubuntu 22.04 (VPS)
- **Python:** 3.12 — `venv/bin/python` always
- **Deployment:** `deploy/goldsignalai.service` (systemd, `Restart=always`)
- **VPS detection:** set `VPS_API_KEY=<any value>` in `.env` → `IS_VPS=True`
