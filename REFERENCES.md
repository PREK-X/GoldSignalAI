# REFERENCES.md
# Read by Claude Code at session start
# For duplicate facts, canonical source is: REFERENCES.md

---

## File Map

```
GoldSignalAI/
├── main.py                     Entry point, signal loop, health check
├── config.py                   Single source of truth for all settings
├── .env                        API keys (gitignored)
├── requirements.txt            Dependencies (no MetaTrader5 on Linux)
├── CLAUDE.md                   How to work on this project
├── CONTEXT.md                  Current state snapshot
├── REFERENCES.md               This file — static architecture
│
├── data/
│   ├── fetcher.py              Fallback: Polygon -> MT5 -> yfinance
│   ├── polygon_fetcher.py      Primary data (2yr M15/H1)
│   ├── processor.py            UTC conversion, dedup, normalization
│   ├── validator.py            Strict OHLCV validation
│   ├── news_fetcher.py         ForexFactory RSS (high-impact events)
│   └── macro_fetcher.py        DXY/VIX/US10Y via yfinance -> SQLite
│
├── analysis/
│   ├── indicators.py           9 voted indicators + PrecomputedIndicators
│   ├── scoring.py              Active-ratio scoring, session filter, gates
│   ├── sr_levels.py            S/R detection (H4 + daily pivots)
│   ├── fibonacci.py            Fibonacci retracement levels
│   ├── candlestick.py          Pattern detection
│   ├── multi_timeframe.py      M15 + H1 agreement logic
│   └── regime_filter.py        GaussianHMM 3-state regime (active)
│
├── signals/
│   ├── generator.py            Signal generation + dedup + MetaDecision
│   ├── formatter.py            Signal formatting for alerts
│   ├── risk_manager.py         Position sizing, SL/TP calculation
│   ├── meta_decision.py        5-rule cascade (HMM+LGBM+conf+session+news)
│   └── news_filter.py          ATR spike + calendar + spread gate
│
├── ml/
│   ├── features.py             62 features (indicator+statistical+temporal)
│   ├── model.py                XGBoost + Random Forest
│   ├── trainer.py              Walk-forward CV (+ LGBM train_lgbm)
│   ├── validator.py            Model validation
│   ├── predictor.py            Prediction + batch (+ LGBM predict_lgbm)
│   ├── retrainer.py            Auto-retrain pipeline (Stage 13)
│   ├── deep_features.py        15 independent features, 60-bar windows
│   ├── deep_model.py           CNN-BiLSTM architecture
│   ├── deep_predictor.py       CNN-BiLSTM inference + batch
│   └── deep_trainer.py         CNN-BiLSTM training pipeline
│
├── backtest/
│   ├── engine.py               Simulation with PrecomputedIndicators
│   └── report_generator.py     Results formatting
│
├── alerts/
│   ├── discord_notifier.py     Primary alerts (webhook)
│   ├── telegram_bot.py         Backup (connectivity issues in PK)
│   └── chart_generator.py      Signal charts
│
├── dashboard/
│   └── app.py                  Streamlit dashboard (6 tabs, Bloomberg theme)
│
├── database/
│   └── db.py                   SQLite (signals + trades + forward_test)
│
├── infrastructure/
│   ├── logger.py               Loguru daily rotation
│   ├── monitoring.py           Sentry integration (optional)
│   └── environment.py          Runtime env detection (VPS/local/Win/Linux)
│
├── deploy/
│   ├── setup_arch.sh           Arch Linux local dev setup
│   ├── setup_vps.sh            Ubuntu/Debian VPS + systemd install
│   ├── setup_windows.bat       Windows local setup
│   ├── goldsignalai.service    systemd unit (Restart=always)
│   └── .env.template           Env var template (no real keys)
│
├── scheduler/
│   └── tasks.py                15-min signal cycle, weekly retrain
│
├── propfirm/
│   ├── tracker.py              Real-time compliance + daily ceiling
│   ├── profiles.py             FundedNext, FTMO, The5ers etc.
│   └── compliance_report.py    Challenge progress reports
│
├── execution/
│   ├── __init__.py
│   ├── mt5_bridge.py           MT5 execution (sim on Linux, real on Win)
│   └── position_monitor.py     Trailing stop, time exit, Friday close
│
├── state/
│   ├── __init__.py
│   ├── state_manager.py        Session loss tracking, JSON persistence
│   └── state.json              Runtime state (gitignored)
│
├── tests/                      160/161 passing (1 pre-existing DST failure)
├── logs/                       Daily rotating logs
├── models/                     ML model files (gitignored)
├── data/historical/            Cached data (gitignored)
├── reports/                    Backtest reports + trade CSV
└── database/                   SQLite DB file
```

---

## Signal Flow (ASCII)

```
Polygon M15+H1 data
       |
  data/fetcher.py --> data/processor.py --> data/validator.py
       |
  analysis/indicators.py  (9 voted indicators)
       |
  analysis/scoring.py  (active-ratio -> BUY/SELL/WAIT)
       |                   + session gate (13-22 UTC)
  analysis/multi_timeframe.py  (M15+H1 must agree)
       |
  signals/meta_decision.py  (5-rule cascade)
       |  R1: HMM gate (CRISIS+RANGING block)
       |  R2: LGBM soft vote
       |  R3: Confidence boost/penalty
       |  R4: Session loss circuit (>=2 losses)
       |  R5: News/volatility filter
       |
  signals/generator.py  (dedup 4hr, fire signal)
       |
  +--> alerts/discord_notifier.py  (webhook)
  +--> database/db.py  (SQLite persist)
  +--> propfirm/tracker.py  (compliance check)
  +--> execution/mt5_bridge.py  (if enabled)
```

---

## Indicator Table

| # | Indicator          | Status    | Notes                              |
|---|--------------------|-----------|------------------------------------|
| 1 | EMA (20/50/200)    | Active    | Price vs EMA stack alignment       |
| 2 | ADX-14             | Active    | Trend strength + direction         |
| 3 | Ichimoku Cloud     | Active    | Very effective on gold             |
| 4 | RSI-14             | Active    | + divergence detection             |
| 5 | MACD (12,26,9)     | Active    | Trend momentum                     |
| 6 | Stochastic (14,3,3)| Active    | %K/%D crossover                    |
| 7 | CCI-20             | Active    | Commodity momentum                 |
| 8 | ATR-14             | Active    | Volatility / SL sizing only        |
| 9 | Volume             | Active    | Surge confirmation                 |
| — | Bollinger Bands    | ML only   | 42.3% voting accuracy -> removed   |
| — | Williams %R        | Rejected  | Stage 2 regression (PF 1.23->0.90) |
| — | Supertrend         | Rejected  | Stage 2 regression                 |
| — | Connors RSI        | Rejected  | Stage 2 regression                 |
| — | Keltner Channels   | Rejected  | Stage 2 regression                 |

**FROZEN** — do not add indicators without per-indicator backtest.

---

## ML Model Summary

| Model     | Architecture              | Status   | Gate   | Actual | Why disabled              |
|-----------|---------------------------|----------|--------|--------|---------------------------|
| XGBoost   | XGB + RF ensemble         | Disabled | 70%    | 47% CV | Trained on indicator outs |
| LightGBM  | LGBM classifier           | Soft use | 53% CV | 52% CV | Used in meta soft vote    |
| HMM       | GaussianHMM 3-state on H1 | Active   | N/A    | N/A    | Hard gate, not predictor  |
| CNN-BiLSTM| Conv1D+BiLSTM+Attention   | Disabled | 54%    | 52.1%  | UP bias; retrain at 150+  |
| Meta      | 5-rule cascade            | Active   | N/A    | N/A    | Wired in backtest + live  |

- LGBM top features: dxy_1d_return, us10y_level, dxy_5d_return, vix_level
- LGBM 24 independent features: returns, ATR ratio, DXY/VIX/US10Y, session, Hurst
- CNN-BiLSTM: 15 features, 60-bar sliding window
- All ML retrained via ml/retrainer.py (Stage 13): LGBM weekly Sun 02:00 UTC,
  CNN-BiLSTM after 150+ live trade outcomes

---

## Meta-Decision Cascade (5 Rules)

| Rule | Name               | Action                                      |
|------|--------------------|---------------------------------------------|
| R1   | HMM Hard Gate      | CRISIS -> block all; RANGING -> block all   |
| R2   | LGBM Soft Vote     | P(UP)<0.40 blocks BUY; P(UP)>0.60 blocks SELL |
| R3   | Confidence Adj     | +5% when TRENDING+LGBM agrees               |
| R4   | Session Loss       | >=2 consecutive losses -> skip session       |
| R5   | News/Volatility    | ATR>2x block; ATR>1.5x reduce 50%; calendar |

---

## Data Sources

| Source    | Symbol     | Coverage         | Limits                          |
|-----------|------------|------------------|---------------------------------|
| Polygon   | C:XAUUSD   | ~2yr M15 (~47k)  | bars=47000 M15, 12000 H1 max   |
| yfinance  | GC=F       | 60 days M15      | Fallback only (hard limit)      |
| yfinance  | DX-Y.NYB   | Macro: DXY       | SQLite cached                   |
| yfinance  | ^VIX       | Macro: VIX       | SQLite cached                   |
| yfinance  | ^TNX       | Macro: US10Y     | SQLite cached                   |
| FF RSS    | calendar   | ~2 weeks forward  | Empty for historical bars       |

**DO NOT** request >47k M15 bars from Polygon — hangs on pagination.
Sequential fetch only — parallel triggers 429 on free tier.

---

## Prop Firm Limits

| Firm           | Daily Loss | Total DD | Profit | Min Days |
|----------------|-----------|----------|--------|----------|
| FundedNext 1S  | 3.0%      | 6.0%     | 10.0%  | 0        |
| FundedNext 2S  | 5.0%      | 10.0%    | 8.0%   | 5        |
| FTMO           | 5.0%      | 10.0%    | 10.0%  | 4        |
| The5ers        | 4.0%      | 6.0%     | 6.0%   | 0        |
| E8 Funding     | 5.0%      | 8.0%     | 8.0%   | 0        |
| MyForexFunds   | 5.0%      | 12.0%    | 8.0%   | 0        |
| Apex           | 3.0%      | 6.0%     | 9.0%   | 0        |

---

## Scoring Bonuses & Penalties

| Modifier                  | Value  | Condition                    |
|---------------------------|--------|------------------------------|
| ADX very strong trend     | +3%    | ADX > 40                     |
| Volume surge              | +2%    | Volume >= 2x average         |
| At strong S/R zone        | +3%    | S/R confirms direction       |
| Fib 61.8% (golden ratio)  | +3%    | Price at golden level        |
| Candlestick pattern       | +2%/ea | Confirming pattern (cap +6%) |
| Doji indecision           | -5%    | Doji detected                |

---

## Dashboard Tabs (Stage 14)

| Tab                | Content                                   |
|--------------------|-------------------------------------------|
| Trade History      | Equity curve (Plotly) + filterable table   |
| ML Status          | 3 model cards: LGBM/HMM/CNN-BiLSTM       |
| Regime Detection   | HMM badge + timeline + distribution pie   |
| Challenge Progress | 4 gauges: PnL/daily loss/total DD/days    |
| Risk Monitor       | CB level, session losses, news events     |
| Signal Heatmap     | WR by hour (NY shading) + weekday         |

---

## Architecture Decisions

| Decision                      | Reasoning                                                                 |
|-------------------------------|---------------------------------------------------------------------------|
| Active-ratio scoring          | Old `/10` made 70% confidence unreachable. New: `dominant/(bull+bear)` ignores neutrals |
| NY session only               | Diagnostic on 277 signals: NY 63.3% WR vs London 33.9%. Session filter is the single biggest edge |
| ML disabled                   | XGBoost 47% CV (trained on indicator outputs = redundant). Need independent features → macro pipeline built |
| SL = ATR × 1.5 (~130 pips)   | Gold M15 median candle = 125 pips. Old 30-pip SL = noise stop-out every trade |
| 9 indicators FROZEN           | Adding 4 more in Stage 2 dropped PF 1.23 → 0.90. Do not add without per-indicator backtest |
| RANGING blocked (not reduced) | RANGING trades avg $+17.87 vs $+81.64 TRENDING, with disproportionate DD |
| FN daily ceiling 2.8%         | Pre-emptive block below 3.0% hard limit. Dropped max daily loss from 3.00% to 2.13% |
| 38% base WR is fine           | With 3.3:1 R:R, break-even is 23%. High WR is a bonus, not a requirement |
| PrecomputedIndicators         | Computing 12 indicators per bar on 48k bars takes hours without the shim in indicators.py |
| MIN_ACTIVE = 4 (not 3)        | MIN_ACTIVE=3 caused PF→1.08 and DD→15%. Reverted. |
| MIN_DOMINANT = 3 (not 4)      | MIN_DOMINANT=4 filters too aggressively with only 9 indicators |
| BBands removed from voting    | 42.3% voting accuracy — worse than random |
| Polygon over yfinance         | yfinance has 60-day hard limit; need 2yr history for valid backtest |
| Separate H1 fetch             | H1 resampled from M15 gives wrong H1 values |
| Sequential Polygon fetch      | Parallel triggers 429 rate limit on free tier |
