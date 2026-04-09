# Stage Progress

Source: CONTEXT.md — Last updated 2026-04-09

## Current Status

**All 16 stages complete. Forward testing phase.**
Awaiting 20 demo trades on IC Markets before FundedNext challenge.

## Completed Stages

| Stage | Name                    | Date       | Key Outcome                     |
|-------|-------------------------|------------|---------------------------------|
| Ph1-3 | Data+Backtest+Stability | pre-2026   | Polygon 2yr, SQLite, Discord    |
| 1     | Environment             | pre-2026   | Arch Linux, Python 3.12         |
| 2     | Indicators (REVERTED)   | pre-2026   | 4 indicators caused PF→0.90     |
| 3     | Macro Features          | pre-2026   | DXY/VIX/US10Y pipeline          |
| 4     | HMM Regime              | pre-2026   | 3-state detector, active        |
| 5     | LightGBM                | pre-2026   | 52% CV, soft vote in meta       |
| 6     | Risk Management         | pre-2026   | CB + Half-Kelly, PF 1.62        |
| 7     | CNN-BiLSTM              | pre-2026   | 52.1% acc, disabled             |
| 8     | Meta-Decision           | pre-2026   | 5-rule cascade                  |
| 10    | News Filter             | pre-2026   | PF 2.45, WR 72.9%               |
| 11    | MT5 Execution           | 2026-04-02 | Bridge + monitor + generator    |
| 12    | Challenge Mode          | 2026-04-02 | Compliance tracking + Discord   |
| 13    | ML Auto-Retrain         | 2026-04-03 | Weekly LGBM, CNN-BiLSTM @150    |
| 14    | Dashboard               | 2026-04-03 | Bloomberg theme, 6 tabs         |
| 15    | Final Testing           | 2026-04-05 | RANGING block + FN ceiling      |
| 16    | Deployment              | 2026-04-05 | Env detect, forward test mode   |

## Remaining: Stage 9 — Multi-Asset

```
XAGUSD -> EURUSD -> US30 -> NAS100 -> USOIL
Per-asset ML models and risk parameters
Portfolio correlation monitoring (max 0.7)
```

Deferred until after FundedNext challenge funded.

## Dashboard Tabs (Stage 14)

| Tab                | Content                                    |
|--------------------|--------------------------------------------|
| Trade History      | Equity curve (Plotly) + filterable table   |
| ML Status          | 3 model cards: LGBM/HMM/CNN-BiLSTM        |
| Regime Detection   | HMM badge + timeline + distribution pie    |
| Challenge Progress | 4 gauges: PnL/daily loss/total DD/days     |
| Risk Monitor       | CB level, session losses, news events      |
| Signal Heatmap     | WR by hour (NY shading) + weekday          |
