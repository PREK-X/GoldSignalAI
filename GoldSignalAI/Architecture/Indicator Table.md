# Indicator Table

Source: REFERENCES.md

> **FROZEN** — do not add indicators without per-indicator backtest.
> Stage 2 added 4 indicators → PF dropped 1.23 → 0.90. Reverted to commit 88c1496.

## Active Indicators (9 Voted)

| # | Indicator           | Status  | Notes                              |
|---|---------------------|---------|------------------------------------|
| 1 | EMA (20/50/200)     | Active  | Price vs EMA stack alignment       |
| 2 | ADX-14              | Active  | Trend strength + direction         |
| 3 | Ichimoku Cloud      | Active  | Very effective on gold             |
| 4 | RSI-14              | Active  | + divergence detection             |
| 5 | MACD (12,26,9)      | Active  | Trend momentum                     |
| 6 | Stochastic (14,3,3) | Active  | %K/%D crossover                    |
| 7 | CCI-20              | Active  | Commodity momentum                 |
| 8 | ATR-14              | Active  | Volatility / SL sizing only        |
| 9 | Volume              | Active  | Surge confirmation                 |

## Rejected Indicators

| Indicator       | Status  | Reason                                      |
|-----------------|---------|---------------------------------------------|
| Bollinger Bands | ML only | 42.3% voting accuracy → removed from voting |
| Williams %R     | Rejected| Stage 2 regression (PF 1.23→0.90)          |
| Supertrend      | Rejected| Stage 2 regression                          |
| Connors RSI     | Rejected| Stage 2 regression                          |
| Keltner Channels| Rejected| Stage 2 regression                          |

## ML Model Summary

| Model      | Architecture              | Status   | Gate  | Actual | Why Disabled              |
|------------|---------------------------|----------|-------|--------|---------------------------|
| XGBoost    | XGB + RF ensemble         | Disabled | 70%   | 47% CV | Trained on indicator outs |
| LightGBM   | LGBM classifier           | Soft use | 53%CV | 52% CV | Used in meta soft vote    |
| HMM        | GaussianHMM 3-state on H1 | Active   | N/A   | N/A    | Hard gate, not predictor  |
| CNN-BiLSTM | Conv1D+BiLSTM+Attention   | Disabled | 54%   | 52.1%  | UP bias; retrain at 150+  |
| Meta       | 5-rule cascade            | Active   | N/A   | N/A    | Wired in backtest + live  |

- LGBM top features: dxy_1d_return, us10y_level, dxy_5d_return, vix_level
- LGBM: 24 independent features — returns, ATR ratio, DXY/VIX/US10Y, session, Hurst
- CNN-BiLSTM: 15 features, 60-bar sliding window
- Auto-retrain: LGBM weekly Sun 02:00 UTC; CNN-BiLSTM after 150+ live trade outcomes
