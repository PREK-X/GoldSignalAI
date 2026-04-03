# GoldSignalAI

AI-powered XAU/USD algorithmic trading signal bot targeting the FundedNext 1-Step $10,000 prop firm challenge.

Built on Python 3.12 with a multi-layer ML pipeline: 9-indicator voting engine → HMM regime filter → LightGBM soft vote → 5-rule meta-decision cascade → ATR-based risk management → Discord alerts. Backtested on 2 years of Polygon.io M15 data (Apr 2024–Mar 2026): **PF 2.45 | DD 3.60% | Win Rate 72.9% | Sharpe 6.00**.

---

## Features

**Signal Engine**
- 9-indicator active-ratio voting (EMA, ADX, Ichimoku, RSI, MACD, Stochastic, CCI, ATR, Volume)
- Active-ratio scoring: `confidence = dominant / (bull + bear)` — ignores neutral indicators
- NY session filter: 13:00–21:59 UTC only (63% win rate vs 34% London — hardcoded gate)
- Dual-timeframe confirmation (M15 primary + H1 agreement)
- S/R levels, Fibonacci retracements, candlestick pattern detection

**ML Layer**
- HMM regime detector — 3-state GaussianHMM on H1 (TRENDING / RANGING / CRISIS): active hard gate
- LightGBM direction classifier — 52.0% CV, trained, soft vote only, filter disabled pending 150+ live trades
- CNN-BiLSTM deep model — 52.1% test accuracy, trained, filter disabled (UP bias noted)
- 24 independent features for LGBM (macro, statistical, temporal — no indicator outputs)
- 15 independent features for CNN-BiLSTM, 60-bar sliding window

**Meta-Decision (5-rule cascade)**
1. HMM hard gate — CRISIS blocks all signals; RANGING halves position size
2. LGBM soft vote — blocks if strong disagreement with technical direction (P<0.40 / P>0.60)
3. Confidence adjustment — ±5% boost/penalty based on regime + LGBM agreement
4. Session loss circuit — skip rest of session after 2 consecutive losses
5. News/volatility filter — ATR spike block (2×) / reduce (1.5×) + ForexFactory calendar + spread check

**Risk Management**
- ATR-based SL (1.5× ATR, capped 50–200 pips) and TP at 1:2 / 1:3 R/R
- Half-Kelly position sizing (1% risk per trade)
- Circuit breaker on consecutive losses and daily drawdown
- Trailing stop to breakeven at 1R, 48-bar time exit, Friday 20:00 UTC forced close

**Challenge Compliance (Stage 12)**
- Real-time FundedNext 1-Step tracking: daily loss + trailing drawdown
- Auto-pause at 2.5% daily / 5.0% total DD warning thresholds
- Hard halt at 3% daily / 6% total DD hard limits
- Daily Discord challenge report at 21:00 UTC; immediate breach alerts
- State persisted to `state/challenge_state.json` (survives restarts)

**ML Auto-Retraining (Stage 13)**
- Weekly LightGBM retrain Sunday 02:00 UTC on latest 2yr Polygon data
- Deploy gate: new CV ≥ 50% and no accuracy regression > 1% vs previous model
- Model backup before every retrain; auto-restore on gate failure or exception
- CNN-BiLSTM retrain triggers automatically after 150+ live trade outcomes
- Discord reports: accuracy before/after, deployed status, backup filename

**Execution Bridge (Stage 11)**
- `execution/mt5_bridge.py`: simulation mode on Linux, real MT5 order execution on Windows/VPS
- `execution/position_monitor.py`: trailing stop, 48-bar timeout, Friday forced close
- `MT5_EXECUTION_ENABLED = False` by default — safe until explicitly enabled

**Alerts & Observability**
- Discord webhook (primary — reliable from Pakistan)
- Telegram bot (backup — connectivity issues in PK)
- SQLite trade and signal logging (`database/goldsignalai.db`)
- Streamlit dashboard at `localhost:8501`
- Sentry error monitoring (optional)

---

## Performance (Backtest — Apr 2024 to Mar 2026)

| Run | Config | Trades | Win Rate | PF | Max DD | PnL |
|-----|--------|--------|----------|----|--------|-----|
| Original (yfinance 60d) | conf=30% | 46 | 30.4% | 0.89 | 9.17% | -$332 |
| After SL fix (50–200 pips) | conf=65% | 30 | 36.7% | 1.04 | 6.34% | +$85 |
| Session filter added | conf=65% | 13 | 38.5% | 1.36 | 3.50% | +$298 |
| **Polygon 2yr baseline** | **conf=65%** | **112** | **38.4%** | **1.23** | **10.04%** | **+$1,773** |
| Stage 5: +LGBM filter (informational) | conf=65% | 78 | 69.2% | 2.38 | 3.89% | +$4,321 |
| Stage 6: Risk mgmt (circuit breaker + Half-Kelly) | conf=65% | 214 | ~40% | 1.62 | 10.50% | — |
| **Stage 10: News & volatility filter** | **conf=65%** | **107** | **72.9%** | **2.45** | **3.60%** | **+$6,748** |
| Stage 11: MT5 bridge + MetaDecision live wiring | N/A | N/A | N/A | N/A | N/A | structural |
| Stage 12: FundedNext challenge compliance | N/A | N/A | N/A | N/A | N/A | protection |
| Stage 13: ML auto-retraining pipeline | N/A | N/A | N/A | N/A | N/A | infrastructure |

> **Current best validated config (Stage 10):** PF 2.45 | DD 3.60% | Win Rate 72.9% | Sharpe 6.00 | 107 trades | +$6,748
>
> The Apr–Oct 2024 period has a 6-month losing streak that depresses the full 2yr PF. Sep 2024–Mar 2026 alone: +$2,884. Stage 5 LGBM result is informational only — CV 52.0% is below the 53% deployment gate.

---

## Quick Start

```bash
# 1. Set up virtual environment (Python 3.12 required)
python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# 2. Create .env and fill in keys (see Environment Setup below)

# 3. Health check
venv/bin/python main.py --health-check

# 4. Run live bot
venv/bin/python main.py

# 5. Run backtest (2yr Polygon data)
venv/bin/python -m backtest.engine

# 6. Run dashboard
venv/bin/python -m streamlit run dashboard/app.py

# 7. Run tests
venv/bin/python -m pytest tests/ -v
```

> **Always use `venv/bin/python`** — never bare `python` or `python3`.

---

## Project Structure

```
GoldSignalAI/
├── main.py                    # Entry point, signal loop, health check
├── config.py                  # Single source of truth for all settings
├── requirements.txt
├── .env                       # API keys (gitignored)
│
├── data/                      # Fetching (Polygon/yfinance), processing, macro (DXY/VIX/US10Y)
├── analysis/                  # Indicators, scoring, S/R, Fibonacci, HMM regime
├── signals/                   # Signal generation, risk sizing, meta-decision cascade
├── ml/                        # XGBoost, LightGBM, CNN-BiLSTM models + retrainer
├── backtest/                  # Backtest engine (PrecomputedIndicators) + reports
├── alerts/                    # Discord webhook, Telegram, chart generator
├── dashboard/                 # Streamlit app (localhost:8501)
├── database/                  # SQLite (signals + trades)
├── infrastructure/            # Loguru logger, Sentry monitoring
├── scheduler/                 # 15-min signal cycle, weekly retrain jobs
├── propfirm/                  # ChallengeTracker, compliance reports, firm profiles
├── execution/                 # MT5Bridge, PositionMonitor
├── state/                     # JSON state files (session loss, challenge, retrain)
├── tests/                     # 154 tests (152 passing)
├── models/                    # Trained model files (gitignored)
│   └── backups/               # Pre-retrain model backups (gitignored)
├── data/historical/           # Cached OHLCV data (gitignored)
├── logs/                      # Daily rotating logs (gitignored)
└── reports/                   # Backtest CSV + PDF reports
```

---

## Environment Setup

**Python 3.12 required.** Python 3.14 is too new — several packages (hmmlearn, tensorflow) break.

```bash
# Arch Linux / Ubuntu
python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

**`.env` file:**

```env
POLYGON_API_KEY=          # Primary data source — 2yr M15/H1 history
DISCORD_WEBHOOK_URL=      # Primary alerts channel webhook

# Optional
TELEGRAM_BOT_TOKEN=       # Backup alerts
TELEGRAM_CHAT_ID=
SENTRY_DSN=               # Error monitoring
MT5_LOGIN=                # MT5 account number (for execution on Windows)
MT5_PASSWORD=
MT5_SERVER=               # e.g. ICMarketsGlobal-Demo
```

Without `POLYGON_API_KEY` the bot falls back to yfinance (60-day limit — backtests will be statistically invalid). Without `DISCORD_WEBHOOK_URL` all alerts are silently skipped.

---

## Configuration

All settings live in `config.py`. Key values:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MIN_CONFIDENCE_PCT` | 65 | Minimum signal confidence to fire |
| `MAX_CONFIDENCE_PCT` | 75 | Above this = lagging over-consensus |
| `ATR_SL_MULTIPLIER` | 1.5 | SL = entry ± (ATR × 1.5) ≈ 130 pips |
| `MIN_SL_PIPS` / `MAX_SL_PIPS` | 50 / 200 | Gold M15 candle range ≈ 125 pips |
| `RISK_PER_TRADE_PCT` | 1.0 | % of account per trade |
| `USE_LGBM_FILTER` | False | CV 52.0% — below 53% gate |
| `USE_DEEP_FILTER` | False | Test accuracy 52.1% — below 54% gate |
| `MT5_EXECUTION_ENABLED` | False | Enable only on Windows/VPS with live MT5 |
| `CHALLENGE_MODE_ENABLED` | True | FundedNext compliance tracking |
| `ACTIVE_PROP_FIRM` | `"FundedNext_1Step"` | Change to switch firm preset |
| `RETRAIN_LGBM_INTERVAL_DAYS` | 7 | Weekly LGBM retrain |
| `RETRAIN_LGBM_MIN_ACCURACY` | 0.50 | Deploy gate for retrained LGBM |
| `RETRAIN_DEEP_MIN_TRADES` | 150 | Live outcomes required for CNN-BiLSTM retrain |

**Note:** `MIN_ACTIVE` (minimum 4 active indicators) and `MIN_DOMINANT` (minimum 3 in dominant direction) are hardcoded in `analysis/scoring.py`, not `config.py`. Do not lower them — see Important Rules.

---

## Prop Firm Target

**FundedNext Stellar 1-Step — $10,000**

| Rule | Limit | Bot action |
|------|-------|-----------|
| Profit target | 10% ($1,000) | — |
| Max daily loss | 3% ($300) | Hard halt at 3%; auto-pause at 2.5% |
| Max total drawdown | 6% ($600) trailing | Hard halt at 6%; auto-pause at 5% |
| Min trading days | None (1-Step) | — |
| Challenge fee | $99 | — |

> **Current backtest DD: 3.60%** (Stage 10, news filter active).
> The trailing DD limit was missed by $19 in simulation — one extreme day slipped through the news filter. **Do not attempt the live challenge until Stage 15 DD fix is confirmed.**

Other supported firm presets: FTMO, FundedNext 2-Step, The5%ers, E8 Funding, MyForexFunds, Apex, Custom. Change `ACTIVE_PROP_FIRM` in `config.py`.

---

## Indicator Set

| # | Indicator | Status | Role |
|---|-----------|--------|------|
| 1 | EMA (20/50/200 stack) | ✅ Active | Price vs EMA alignment |
| 2 | ADX-14 | ✅ Active | Trend strength + direction |
| 3 | Ichimoku Cloud | ✅ Active | Very effective on gold |
| 4 | RSI-14 + divergence | ✅ Active | Momentum + reversal |
| 5 | MACD (12,26,9) | ✅ Active | Trend momentum |
| 6 | Stochastic (14,3,3) | ✅ Active | %K/%D crossover |
| 7 | CCI-20 | ✅ Active | Commodity momentum |
| 8 | ATR-14 | ✅ Active | Volatility — SL sizing only (neutral vote) |
| 9 | Volume | ✅ Active | Surge confirmation |
| — | Bollinger Bands | ⚠️ ML feature only | 42.3% voting accuracy — removed from voting |
| — | Williams %R | ❌ Not validated | Added Stage 2, caused PF regression 1.23→0.90 |
| — | Supertrend | ❌ Not validated | Caused regression |
| — | Connors RSI | ❌ Not validated | Caused regression |
| — | Keltner Channels | ❌ Not validated | Caused regression |

---

## Architecture

Signal flow on each 15-minute candle close:

```
Data fetch     Polygon.io M15 + H1 → processor → validator
               DXY / VIX / US10Y via yfinance → SQLite macro cache

Indicators     9-indicator voting → active-ratio confidence score

Session gate   Reject outside 13:00–21:59 UTC
Confidence     Reject if score < 65% or > 75%

Meta-decision  5-rule cascade:
               1. HMM regime gate  (CRISIS=block, RANGING=half-size)
               2. LGBM soft vote   (blocks strong disagreement)
               3. Confidence ±5%   (regime + LGBM alignment)
               4. Session circuit  (≥2 consecutive losses → skip session)
               5. News/ATR filter  (spike block/reduce + calendar + spread)

Risk sizing    ATR SL/TP + Half-Kelly lot size
Dedup check    Skip if same direction fired in last 4 hours

Output         Discord alert + SQLite log + MT5 execution (if enabled)
```

The backtest engine mirrors this exact flow. `PrecomputedIndicators` runs indicator calculations once per bar across all 48k bars, making the full 2yr backtest complete in seconds rather than hours.

---

## Development Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup (Arch Linux, Python 3.12) | ✅ Complete |
| 2 | Additional indicators | ❌ Reverted — PF 1.23→0.90 |
| 3 | Macro features (DXY, VIX, US10Y → ML) | ✅ Complete |
| 4 | HMM regime detection (3-state GaussianHMM) | ✅ Complete |
| 5 | LightGBM classifier (CV 52.0%, gate not met) | ✅ Complete |
| 6 | Risk management (circuit breaker, Half-Kelly) | ✅ Complete |
| 7 | CNN-BiLSTM deep model (52.1%, gate not met) | ✅ Complete |
| 8 | Meta-decision layer (4→5 rule cascade) | ✅ Complete |
| 9 | Multi-asset support (EURUSD, US30, NAS100…) | 📋 Planned |
| 10 | News & volatility filter (PF 2.45, DD 3.60%) | ✅ Complete |
| 11 | MT5 auto-execution + live MetaDecision wiring | ✅ Complete |
| 12 | FundedNext challenge compliance mode | ✅ Complete |
| 13 | ML auto-retraining pipeline | ✅ Complete |
| 14 | Dashboard upgrade (equity curve, ML status…) | 📋 Planned |
| 15 | Final testing + trailing DD breach fix | 📋 Planned |
| 16 | VPS deployment + forward testing + challenge | 📋 Planned |

---

## Important Rules

1. **Always use `venv/bin/python`** — never bare `python` or `python3`. Python 3.14 breaks several packages.

2. **NY session only (13:00–21:59 UTC).** London session has a 34% win rate vs NY's 63%. The session gate is hardcoded in `analysis/scoring.py` — do not remove it.

3. **Do not lower `MIN_ACTIVE` below 4 or `MIN_DOMINANT` below 3.** Both constants live in `analysis/scoring.py`. Lowering `MIN_ACTIVE` to 3 previously raised DD to 14.94% and dropped PF to 1.08.

4. **Do not add unvalidated indicators.** Connors RSI, Keltner, Supertrend, and Williams %R were added in Stage 2 and dropped PF from 1.23 → 0.90. Any new indicator must be validated with a per-indicator 2yr backtest before adding to the voting set.

5. **`MT5_EXECUTION_ENABLED = False`.** Leave False until running on Windows or a VPS with a live MT5 terminal. On Linux this runs in simulation mode only.

6. **Do not attempt the FundedNext challenge until Stage 15 is confirmed.** The current simulation breaches the $600 trailing DD limit by $19 on one extreme day not caught by the news filter.

7. **ML filters remain disabled.** `USE_LGBM_FILTER = False`, `USE_DEEP_FILTER = False`. The LGBM model participates in the meta-decision soft vote regardless of this flag. Re-evaluate both models after 150+ live trade outcomes are collected via the Stage 13 auto-retraining pipeline.
