# Current Config Values

Source: CONTEXT.md — Last updated 2026-04-09

## config.py

```
MIN_CONFIDENCE_PCT           = 65
MAX_CONFIDENCE_PCT           = 75
ATR_SL_MULTIPLIER            = 1.5   (SL ~130 pips)
MIN_SL_PIPS / MAX_SL_PIPS   = 50 / 200
TOTAL_INDICATORS             = 9
USE_ML_FILTER                = False  (47% CV)
USE_LGBM_FILTER              = False  (52% CV < 53% gate)
USE_DEEP_FILTER              = False  (52.1% < 54% gate)
RISK_PER_TRADE_PCT           = 1.0
ACTIVE_PROP_FIRM             = "FundedNext_1Step"
CHALLENGE_MODE_ENABLED       = True
FUNDEDNEXT_DAILY_CEILING_PCT = 2.8
FORWARD_TEST_MODE            = True   (never ship False)
FORWARD_TEST_MAX_TRADES      = 20
MT5_EXECUTION_ENABLED        = False  (manual on Linux)
NEWS_FILTER_ENABLED          = True
META_LGBM_BLOCK_LOW/HIGH     = 0.40 / 0.60
META_CONFIDENCE_BOOST        = 5.0
META_MAX_SESSION_LOSS        = 2
```

## analysis/scoring.py (NOT in config.py)

```
MIN_ACTIVE           = 4    # min active (bull+bear) indicators
MIN_DOMINANT         = 3    # min in dominant direction
SESSION_ACTIVE_HOURS = frozenset(range(13, 22))  # 13:00-21:59 UTC
```

## What Is Disabled and Why

| Feature       | Flag                    | Why                               |
|---------------|-------------------------|-----------------------------------|
| XGBoost       | USE_ML_FILTER=False     | 47% CV, worse than coin flip      |
| LGBM filter   | USE_LGBM_FILTER=False   | 52% CV < 53% gate; soft vote ok   |
| CNN-BiLSTM    | USE_DEEP_FILTER=False   | 52.1% < 54% gate; UP bias         |
| MT5 execution | MT5_EXECUTION_ENABLED=False | Linux has no MT5 terminal     |
| RANGING trades| meta_decision.py R1     | Blocked: $+17.87 avg, high DD     |

Re-enable ML models after 150+ real trade outcomes for retraining.

## Data Source Limits

| Source  | Symbol    | Coverage          | Limit                         |
|---------|-----------|-------------------|-------------------------------|
| Polygon | C:XAUUSD  | ~2yr M15 (~47k)   | bars=47000 M15, 12000 H1 max |
| yfinance| GC=F      | 60 days M15       | Fallback only (hard limit)    |
| yfinance| DX-Y.NYB  | Macro: DXY        | SQLite cached                 |
| yfinance| ^VIX      | Macro: VIX        | SQLite cached                 |
| yfinance| ^TNX      | Macro: US10Y      | SQLite cached                 |
| FF RSS  | calendar  | ~2 weeks forward  | Empty for historical bars     |

> **DO NOT** request >47k M15 bars from Polygon — hangs on pagination.
> Sequential fetch only — parallel triggers 429 on free tier.
