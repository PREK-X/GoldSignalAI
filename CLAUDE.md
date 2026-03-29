# GoldSignalAI — Complete Project Context

## What This Is
GoldSignalAI is an AI-powered algorithmic trading signal bot for XAU/USD (Gold), built by Masab on Ubuntu/Arch Linux using Python. The primary goal is to pass prop firm challenges (FundedNext 1-Step $10k) and eventually generate consistent funded account returns. The bot generates trading signals, sends alerts via Discord, logs everything to SQLite, and has a Streamlit dashboard.

---

## Developer Environment
- **OS:** Arch Linux (previously Ubuntu)
- **Python:** 3.12 (via AUR — Python 3.14 is too new for some packages)
- **Venv:** `~/Documents/projects/GoldSignalAI/venv/`
- **Always run with:** `venv/bin/python` (never `python` or `python3`)
- **IDE:** VS Code
- **Claude Code:** Primary implementation tool
- **GitHub:** github.com/PREK-X/GoldSignalAI (Private)

---

## Project Structure
```
GoldSignalAI/
├── main.py                    # Entry point, signal loop, health check
├── config.py                  # Single source of truth for all settings
├── .env                       # API keys (gitignored)
├── requirements.txt           # Dependencies (MetaTrader5 removed — Linux)
├── CLAUDE.md                  # This file
│
├── data/
│   ├── fetcher.py             # Fallback hierarchy: Polygon → MT5 → yfinance
│   ├── polygon_fetcher.py     # Primary data source (2yr M15/H1)
│   ├── processor.py           # UTC conversion, dedup, normalization
│   ├── validator.py           # Strict OHLCV validation
│   ├── news_fetcher.py        # ForexFactory RSS (high-impact events)
│   └── macro_fetcher.py       # PLANNED: DXY, VIX, US10Y feeds
│
├── analysis/
│   ├── indicators.py          # 9 voted indicators + PrecomputedIndicators shim (backtest perf)
│   ├── scoring.py             # Active-ratio scoring, session filter, gates
│   ├── sr_levels.py           # S/R detection (H4 + daily pivots)
│   ├── fibonacci.py           # Fibonacci retracement levels
│   ├── candlestick.py         # Pattern detection
│   ├── multi_timeframe.py     # M15 + H1 agreement logic
│   └── regime_filter.py       # PLANNED: HMM regime detection
│
├── signals/
│   ├── generator.py           # Signal generation with dedup check
│   ├── formatter.py           # Signal formatting for alerts
│   └── risk_manager.py        # Position sizing, SL/TP calculation
│
├── ml/
│   ├── features.py            # 62 features (indicator + statistical + temporal)
│   ├── model.py               # XGBoost + Random Forest models
│   ├── trainer.py             # Training with walk-forward CV
│   ├── validator.py           # Model validation
│   └── predictor.py           # Prediction + batch prediction
│
├── backtest/
│   ├── engine.py              # Full simulation engine with PrecomputedIndicators
│   └── report_generator.py    # Results formatting
│
├── alerts/
│   ├── discord_notifier.py    # Primary alerts (webhook, works in Pakistan)
│   ├── telegram_bot.py        # Backup (has connectivity issues in Pakistan)
│   └── chart_generator.py     # Signal charts
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard (localhost:8501)
│
├── database/
│   └── db.py                  # SQLite persistence (signals + trades)
│
├── infrastructure/
│   ├── logger.py              # Loguru daily rotation
│   └── monitoring.py          # Sentry integration (optional)
│
├── scheduler/
│   └── tasks.py               # 15-min signal cycle, weekly retrain
│
├── propfirm/
│   ├── tracker.py             # Real-time prop firm compliance
│   ├── profiles.py            # FundedNext, FTMO, The5ers etc.
│   └── compliance_report.py   # Challenge progress reports
│
├── state/
│   └── state_manager.py       # Runtime state (open_trade, daily_loss etc.)
│
├── tests/
│   ├── test_data_pipeline.py  # 17 tests (all passing)
│   ├── test_scoring.py        # Scoring tests
│   ├── test_risk_manager.py   # Risk limit tests
│   └── test_data_validation.py# Validation tests
│
├── logs/                      # Daily rotating logs
├── models/                    # ML model files (gitignored)
├── data/historical/           # Cached data (gitignored)
├── reports/                   # Backtest reports + trade CSV
└── database/                  # SQLite DB file
```

---

## Environment Variables (.env)
```
POLYGON_API_KEY=           # Primary data source (2yr M15 data)
DISCORD_WEBHOOK_URL=       # Primary alerts (works in Pakistan)
TELEGRAM_BOT_TOKEN=        # Backup alerts (connectivity issues in PK)
TELEGRAM_CHAT_ID=          # Telegram chat
SENTRY_DSN=                # Error monitoring (optional, Student Pack)
```

---

## Key Configuration (config.py)

### Validated Backtest Config (9-indicator system, restored 2026-03-28)
```python
MIN_CONFIDENCE_PCT   = 65    # Active-ratio scoring threshold
MAX_CONFIDENCE_PCT   = 75    # Cap — above = lagging signal
ATR_SL_MULTIPLIER    = 1.5   # SL = ATR × 1.5 (~130 pips)
MIN_SL_PIPS          = 50    # Minimum stop loss
MAX_SL_PIPS          = 200   # Maximum stop loss
TOTAL_INDICATORS     = 9     # 9 voted indicators (BBands removed — negative accuracy)
USE_ML_FILTER        = False # ML disabled (47% accuracy = worse than coin flip)
ACTIVE_PROFILE       = "FundedNext_1Step"
RISK_PER_TRADE_PCT   = 1.0
```

### Session Filter (Critical)
```python
SESSION_ACTIVE_HOURS = frozenset(range(13, 22))  # 13:00–21:59 UTC (NY + Overlap)
# NY session: 13:00-22:00 UTC = 6:00 PM - 1:00 AM PKT
# London session deliberately excluded (33.9% win rate vs 63.3% NY)
```

### Scoring Gates
```python
MIN_ACTIVE    = 4  # Minimum active (bull+bear) indicators
MIN_DOMINANT  = 3  # Minimum in dominant direction (9-indicator system)
# DO NOT raise MIN_DOMINANT to 4 — with 9 indicators it filters too aggressively
# DO NOT add more unvalidated indicators (Stage 2 additions dropped PF 1.23→0.90)
```

### Indicator Set (9 Voted — FROZEN until Stage 3 macro features added)
```
1. EMA (20/50/200 stack)    6. Stochastic (%K/%D crossover)
2. ADX-14                   7. CCI-20
3. Ichimoku Cloud           8. Bollinger Bands (touch + squeeze)
4. RSI-14 + divergence      9. ATR-14 (volatility only, neutral vote)
5. MACD (12,26,9)          10. Volume surge confirmation
BBands kept as ML feature only — removed from voting (42.3% accuracy)
```

---

## Indicator Set (9 Voted — Validated Config, DO NOT ADD MORE)

| # | Indicator | Status | Notes |
|---|-----------|--------|-------|
| 1 | EMA (20/50/200 stack) | ✅ Active | Price vs EMA alignment |
| 2 | ADX-14 | ✅ Active | Trend strength + direction |
| 3 | Ichimoku Cloud | ✅ Active | Very effective on gold |
| 4 | RSI-14 | ✅ Active | + divergence detection |
| 5 | MACD (12,26,9) | ✅ Active | Trend momentum |
| 6 | Stochastic (14,3,3) | ✅ Active | %K/%D crossover |
| 7 | CCI-20 | ✅ Active | Commodity momentum |
| 8 | ATR-14 | ✅ Active | Volatility / SL sizing only |
| 9 | Volume | ✅ Active | Surge confirmation |
| — | Bollinger Bands | ⚠️ ML only | 42.3% voting accuracy = removed from scoring |
| — | Williams %R | ❌ Not validated | Added in Stage 2, caused PF regression |
| — | Supertrend | ❌ Not validated | Added in Stage 2, caused PF regression |
| — | Connors RSI | ❌ Not validated | Added in Stage 2, caused PF regression |
| — | Keltner Channels | ❌ Not validated | Added in Stage 2, caused PF regression |

**WARNING:** Adding indicators 10-13 (Connors RSI, Keltner, Supertrend, Williams %R)
dropped PF from 1.23 → 0.90. Do not re-add without per-indicator backtested validation.

---

## Data Sources

### Primary: Polygon.io (Free Tier)
- Symbol: `C:XAUUSD`
- Coverage: ~2 years of M15 data (~47,000 bars)
- M15 bars: ~48,000 | H1 bars: ~12,000
- Fetch limit: Use `bars=47000` for M15, `bars=12000` for H1
- **DO NOT** request more — hangs on pagination

### Fallback: yfinance
- Symbol: `GC=F` (Gold futures)
- M15 limit: Last 60 days only (hard limit)
- Used only when Polygon key missing

### Planned: macro_fetcher.py
- DXY via yfinance (`DX-Y.NYB`)
- VIX via yfinance (`^VIX`)
- US10Y via yfinance (`^TNX`) or FRED API

---

## Backtest Results History

| Run | Config | Trades | Win Rate | PF | Max DD | PnL |
|-----|--------|--------|----------|----|--------|-----|
| Original (yfinance 60d) | conf=30% | 46 | 30.4% | 0.89 | 9.17% | -$332 |
| After SL fix (50-200 pips) | conf=65% | 30 | 36.7% | 1.04 | 6.34% | +$85 |
| Session filter added | conf=65% | 13 | 38.5% | 1.36 | 3.50% | +$298 |
| **Polygon 2yr data** | **conf=65%** | **112** | **38.4%** | **1.23** | **10.04%** | **+$1,773** |
| After MIN_ACTIVE=3 (reverted) | conf=65% | 180 | 35.6% | 1.08 | 14.94% | +$1,003 |
| Stage 2 indicators (broken) | conf=65% | 111 | 31.5% | 0.90 | 15.72% | -$696 |
| **9-indicator revert (current)** | **conf=65%** | **180** | **36.1%** | **1.09** | **12.27%** | **+$1,030** |

**Best validated config:** 9 indicators, PF 1.09–1.23, confirmed profitable on 2yr Polygon data.
Note: PF varies by data window. Original PF 1.23 was on an earlier dataset; current 2yr window
(Apr 2024–Mar 2026) hits a 6-month losing streak at the start (Apr–Oct 2024) which depresses PF.
Sep 2024–Mar 2026 alone is strongly profitable (+$2,884).

---

## Critical Bugs Found & Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| 70% confidence mathematically unreachable | 0 signals ever | Changed to active-ratio scoring |
| SL capped at 30 pips | Every trade hit SL (noise) | Changed to 50-200 pips ATR-based |
| Bollinger Bands in scoring | 42.3% accuracy | Removed from voting |
| London session trading | 33.9% win rate | Session filter (NY only) |
| yfinance 60-day limit | Invalid backtest | Added Polygon.io |
| Backtest resampling H1 from M15 | Wrong H1 indicators | Separate H1 fetch |
| ML filter blocking good signals | PF degraded | Disabled (USE_ML_FILTER=False) |
| MIN_ACTIVE=3 too low | PF 1.23→1.08, DD 14.94% | Reverted to MIN_ACTIVE=4 |
| Backtest hanging on 48k bars | Never completes | PrecomputedIndicators shim in indicators.py |
| Polygon fetching 5yr instead of 2yr | Timeout/hang | bars=47000 for M15 |
| Stage 2 indicators (Connors/Keltner/Supertrend) | PF 1.23→0.90 | Reverted indicators.py+scoring.py to commit 88c1496 |
| Parallel Polygon fetch rate-limited (429) | Both fetches time out | Sequential fetch with 300s timeout |
| Telegram blocked in Pakistan | No alerts | Replaced with Discord webhook |

---

## Why These Decisions Were Made

### Active-Ratio Scoring (not /10)
Old: `confidence = bullish_count / 10` → max was 55% → 70% threshold unreachable
New: `confidence = bullish_count / (bullish + bearish)` → ignores neutral indicators
Example: 4 bull, 1 bear, 7 neutral → 4/5 = 80% confidence

### NY Session Only
Diagnostic on 277 signals showed:
- NY session: 63.3% win rate ✅
- Overlap: 53.6% ✅
- London only: 33.9% ❌
- Outside sessions: 46.6% ❌

### ML Disabled
- XGBoost CV accuracy: 47% (worse than 50% coin flip)
- Was blocking 78% of signals for no benefit
- Root cause: trained on same indicator outputs as scoring engine
- Fix: need independent features (DXY, VIX, macro data) — planned Stage 5

### SL = ATR × 1.5 (~130 pips)
- Gold M15 median candle range: 125 pips
- SL must be wider than one candle to avoid noise stop-outs
- Old MAX_SL_PIPS=30 was 3 price units — less than one candle

---

## Prop Firm Target

**FundedNext 1-Step $10,000**
- Profit Target: 10% ($1,000)
- Max Daily Loss: 4% ($400)
- Max Total Drawdown: 8% ($800)
- Challenge Fee: $99
- Configured in: `ACTIVE_PROFILE = "FundedNext_1Step"`

**Current backtest DD: 10.04%** — just over the 10% absolute limit
- Prop firm simulation uses 8% limit → currently failing
- Fix planned: HMM regime filter (Stage 4) should bring DD to 6-8%

---

## Complete Build Roadmap

### ✅ COMPLETED
- **Phase 1** — Data Infrastructure (Polygon.io, 2yr M15/H1)
- **Phase 2** — Backtesting (validated config PF 1.09–1.23, profitable on 2yr data)
- **Phase 3** — Stability (SQLite, Discord, health check, dedup)
- **Stage 1** — Environment setup (Arch Linux, Python 3.12)
- **Stage 2** — REVERTED (Connors RSI/Keltner/Supertrend added noise, PF→0.90, fixed by revert)

### 📋 REMAINING STAGES

#### Stage 3 — Macro Features Pipeline
```
Create data/macro_fetcher.py
Fetch: DXY, VIX, US10Y via yfinance (free)
Add to ml/features.py as independent features
DXY is most important — ~-0.80 correlation with gold
```

#### Stage 4 — HMM Regime Detection ⭐ HIGHEST PRIORITY
```
Library: hmmlearn
Model: GaussianHMM, 3 states on H1 data
State 0: Trending low-vol → full trading
State 1: Ranging medium → half position size
State 2: Crisis high-vol → no trading
Expected: DD drops 2-3%, filters ~30% losing trades
```

#### Stage 5 — LightGBM Classifier
```
Independent features (NOT indicator values):
- Multi-lookback returns (5,15,30,60,120 bars)
- ATR ratio (ATR7/ATR28)
- DXY trend flag, VIX level/change
- Session encoding, calendar flags
- Rolling Hurst exponent
Target: 54-57% accuracy (better than 47% current)
```

#### Stage 6 — Risk Management Overhaul
```
Multi-level circuit breaker:
- 2% daily loss → 50% position size
- 3% daily loss → high confidence only
- 4% daily loss → stop for day
- 8% total DD → 25% size

Half-Kelly position sizing with ATR adjustment
Friday 20:00 UTC close (weekend gap protection)
48-bar time exit (stale trade killer)
Trailing stop at 1R profit
```

#### Stage 7 — CNN-BiLSTM Deep Learning
```
Conv1D → BatchNorm → BiLSTM → Attention → Dense
60-bar lookback window
Train on Google Colab (free GPU)
Expected: 55-58% accuracy
```

#### Stage 8 — Meta-Decision Layer
```
Scoring Engine + LightGBM + CNN-BiLSTM + HMM
→ single decision function
→ HMM regime gates all signals
→ LightGBM must agree with scoring direction
→ CNN boosts/reduces confidence
```

#### Stage 9 — Multi-Asset Support
```
XAGUSD → EURUSD → US30 → NAS100 → USOIL
Per-asset ML models and risk parameters
Portfolio correlation monitoring (max 0.7)
```

#### Stage 10 — Smarter News Filter
```
Add Finnhub economic calendar API
Volatility spike detection (ATR doubles → pause)
Spread monitoring (>5 pips → skip signal)
```

#### Stage 11 — MT5 Auto-Execution
```
Windows/VPS required (MT5 Python API is Windows-only)
Automatic order placement with SL/TP
Position monitoring every 15 minutes
```

#### Stage 12 — FundedNext Challenge Mode
```
Real-time profit/loss tracking toward targets
Auto-pause when approaching limits
Daily Discord progress notifications
```

#### Stage 13 — ML Auto-Retraining Pipeline
```
Weekly retrain on latest 2yr data
Walk-forward validation before deploying new model
Discord notification with accuracy results
```

#### Stage 14 — Dashboard Upgrade
```
Add pages: Trade History, ML Status, Regime Detection,
Multi-Asset Overview, Challenge Progress, Risk Monitor
Add charts: Equity curve, signal heatmap, correlation matrix
```

#### Stage 15 — Final Testing
```
Full 2-year backtest with all stages combined
Target: PF 1.6-1.9, DD 5-7%, Win Rate 46-50%
```

#### Stage 16 — Deployment
```
VPS hosting (DigitalOcean $6/month — Student Pack)
24/7 operation, auto-restart
Forward test: 20 real trades on IC Markets demo
Small live account: $200-500, 0.1% risk
FundedNext challenge: only after live profitable
```

---

## Pending Features (Remind at Phase 8+)
```
1. Multiple assets (EURUSD, US30, NAS100, USOIL, XAGUSD)
2. Smarter news filter (Finnhub, volatility spikes, spread monitor)
3. MT5 auto-execution (Windows/VPS)
4. ML retraining on real forward test data (100+ trades)
5. FundedNext challenge mode (auto-pause near limits)
```

---

## How to Run

```bash
# Health check
venv/bin/python main.py --health-check

# Run live bot
venv/bin/python main.py

# Run backtest
venv/bin/python -m backtest.engine

# Run dashboard
venv/bin/python -m streamlit run dashboard/app.py

# Run tests
venv/bin/python -m pytest tests/ -v
```

---

## Live Bot Behavior

- Runs signal loop every 15 minutes (M15 candle close)
- Only fires BUY/SELL during NY session (13:00-22:00 UTC = 6PM-1AM PKT)
- Sends Discord alert when signal passes all gates
- Logs everything to SQLite database
- Dedup: skips signal if same direction fired in last 4 hours
- Expected frequency: ~1 signal per 6 days (quality over quantity)

---

## IC Markets Demo Account
- Account: 52791555
- Server: ICMarketsGlobal-Demo
- Platform: MT5 (mobile app)
- Used for: Manual forward testing (Phase 4)
- Phase 4 goal: Log 20 real trades in Google Sheet

---

## Expected Performance (Validated Backtest)
```
Win Rate:       38.4% (need only 23% to break even with 3.3:1 R:R)
Avg Win:        +364 pips
Avg Loss:       -110 pips
Profit Factor:  1.23
2yr Return:     +17.7% on $10k
Max Drawdown:   10.04%
Sharpe Ratio:   1.47
```

## Expected Performance (After All Stages)
```
Win Rate:       46-50%
Profit Factor:  1.6-1.9
Max Drawdown:   5-7%
Sharpe Ratio:   2.0-2.5
2yr Return:     +35-50%
```

---

## ML Architecture (Planned)

### Current (Disabled)
- XGBoost + Random Forest
- Features: same indicator outputs as scoring (redundant → 47% accuracy)
- Status: USE_ML_FILTER = False

### Planned Three-Model Architecture
```
Model A: LightGBM — direction classifier
         Independent features (macro, statistical, temporal)
         Target accuracy: 54-57%

Model B: HMM — regime detector (3 states)
         Features: log returns + realized vol on H1
         NOT a predictor — a FILTER

Model C: CNN-BiLSTM — deep direction model
         60-bar lookback, Conv1D + BiLSTM + Attention
         Target accuracy: 55-58%
         Requires GPU for training (Google Colab)
```

---

## Key Learnings

1. **Scoring engine is not broken** — it identifies real setups. Problem is it trades indiscriminately regardless of regime.
2. **Every upgrade = adding selectivity** — filtering losers, not finding more winners.
3. **NY session is the edge** — 63% win rate vs 33% London. This alone is the most valuable finding.
4. **ML needs independent features** — training on indicator outputs = redundant information = no edge.
5. **SL must match volatility** — gold's average M15 candle is 125 pips. SL < 100 pips = noise stop-out.
6. **38% win rate is fine** — with 3.3:1 R:R, break-even is 23%. Quality not quantity.
7. **Backtest on 13 trades = meaningless** — need 100+ trades for statistical validity (required Polygon).
8. **PrecomputedIndicators is critical** — computing 12 indicators on every bar of 48k bars takes hours without it.

---

## Tools & Resources
- **Data:** Polygon.io (primary), yfinance (fallback)
- **Alerts:** Discord webhook (primary), Telegram (backup)
- **ML:** XGBoost, LightGBM (planned), hmmlearn (planned), PyTorch (planned)
- **Dashboard:** Streamlit
- **Database:** SQLite
- **Hosting:** Local PC (future: DigitalOcean VPS via Student Pack)
- **Prop Firm:** FundedNext ($99 challenge fee)
- **Demo Broker:** IC Markets (Raw Spread, MT5)

---

## Current Status (2026-03-28)
**9-indicator system restored. PF 1.09 on current 2yr Polygon window. Ready for Stage 3.**

Stage 2 regression was caused by adding 4 unvalidated indicators (Connors RSI, Keltner Channels,
Supertrend, Williams %R) and raising MIN_DOMINANT from 3→4. This degraded signal quality.
Fix: reverted `analysis/indicators.py` and `analysis/scoring.py` to commit `88c1496`.

Note on PF 1.09 vs original 1.23: different market windows. Apr–Oct 2024 was a bad regime
for this strategy (5 consecutive losing months). Sep 2024–Mar 2026 is strongly profitable.
The strategy is sound. Next priority: Stage 3 macro features (DXY/VIX) to filter bad regimes.
