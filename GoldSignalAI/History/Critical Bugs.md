# Critical Bugs

Source: CLAUDE.md + audit session 2026-04-09

## Historical Bugs (Resolved)

| Bug                        | Impact               | Fix                         |
|----------------------------|----------------------|-----------------------------|
| 70% confidence unreachable | 0 signals ever       | Active-ratio scoring        |
| SL capped at 30 pips       | Every trade hit SL   | 50-200 pips ATR-based       |
| BBands in scoring          | 42.3% accuracy       | Removed from voting         |
| London session trading     | 33.9% WR             | NY-only session filter      |
| yfinance 60-day limit      | Invalid backtest     | Polygon.io added            |
| H1 resampled from M15      | Wrong H1 values      | Separate H1 fetch           |
| ML blocking good signals   | PF degraded          | USE_ML_FILTER=False         |
| MIN_ACTIVE=3               | PF→1.08, DD 15%      | Reverted to 4               |
| Backtest hang on 48k bars  | Never completes      | PrecomputedIndicators       |
| Polygon 5yr fetch          | Timeout/hang         | bars=47000 cap              |
| Stage 2 indicators         | PF→0.90              | Reverted to commit 88c1496  |
| Parallel Polygon 429       | Both fetches fail    | Sequential + 300s timeout   |
| Telegram blocked in PK     | No alerts            | Discord webhook             |
| LGBM macro merge bug       | 0 samples            | Set index.name in features.py|
| Stale Polygon data         | Bot used yesterday's bars | +1 day end_date — FIXED 2026-04-08 |
| Scoring WAIT at 88% conf   | Valid signals blocked | MAX_CONFIDENCE ceiling gate removed — FIXED 2026-04-08 |

## Audit Fixes (2026-04-09)

| Bug                           | Severity | Fix                                                |
|-------------------------------|----------|----------------------------------------------------|
| Session gate disabled         | Critical | bar_time threaded through multi_timeframe.py       |
| ATR calculation divergence    | High     | Live path now uses Wilder's EWM (matches backtest) |
| Session loss rollover         | High     | get_session_losses() auto-resets on day boundary  |
| Same-bar TP1+TP2 handling     | High     | Backtest handles both TPs hitting same candle      |
| Sharpe ratio annualization    | High     | Uses actual trade frequency not hardcoded 252      |
| R/R uses post-breakeven SL    | High     | Switched to initial_sl_pips                        |
| Scheduler timezone-naive      | Medium   | All .at() calls now specify UTC explicitly         |
| Confidence logging unclear    | Medium   | Shows before→after values                          |
| Discord no retry logic        | Medium   | 3 retries with backoff + 429 handling              |
| Indicator count hardcoded     | Medium   | Dynamic len() instead of "/10"                     |
| Dashboard PnL gauge abs()     | Medium   | max(0, ...) — negative PnL no longer shows positive|
| Sim uses midpoint entry       | Medium   | Uses actual signal entry_price                     |
| DB update no row check        | Medium   | Warns if update_trade_result matches 0 rows        |
| Feature inf with dropna=False | Medium   | inf replaced before dropna conditional             |
| Chart temp files never cleaned| Low      | os.remove() after Discord send                     |
| Confidence threshold hardcoded| Low      | Uses Config.MIN_CONFIDENCE_PCT                     |
| FN ceiling missing in sim     | Low      | prop firm simulation now enforces 2.8% ceiling     |
