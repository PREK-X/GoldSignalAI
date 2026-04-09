# File Map

Source: REFERENCES.md

```
GoldSignalAI/
├── main.py                     Entry point, signal loop, health check
├── config.py                   Single source of truth for all settings
├── .env                        API keys (gitignored)
├── requirements.txt            Dependencies (no MetaTrader5 on Linux)
├── CLAUDE.md                   How to work on this project
├── CONTEXT.md                  Current state snapshot
├── REFERENCES.md               Static architecture reference
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
