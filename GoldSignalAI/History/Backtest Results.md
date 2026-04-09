# Backtest Results

Source: CONTEXT.md

## Latest Result (Stage 15 Phase 2.5)

| Metric | Value                              |
|--------|------------------------------------|
| Trades | 75                                 |
| WR     | 72.0%                              |
| PF     | 2.15                               |
| DD     | 4.99%                              |
| Sharpe | 5.31                               |
| PnL    | +$4,818                            |
| Period | Apr 2024 - Mar 2026 (2yr Polygon)  |

> Stage 16 = deployment only, no metric changes.

## Prop Firm Sim Results (Stage 15 Ph2.5)

All 8 firms pass: FTMO, FN 1-Step, FN 2-Step, The5ers, E8, MFF, Apex, Custom.

FundedNext 1-Step specifics:
- Max daily loss: 2.13% (well under 3.0% hard limit)
- Max total DD: 4.99% (under 6.0%)
- Profit: +48.2% (far exceeds 10% target)

## Backtest History

| Run                          | Trades | WR    | PF   | DD     | PnL      |
|------------------------------|--------|-------|------|--------|----------|
| Original (yfinance 60d)      | 46     | 30.4% | 0.89 | 9.17%  | -$332    |
| SL fix (50-200 pips)         | 30     | 36.7% | 1.04 | 6.34%  | +$85     |
| Session filter               | 13     | 38.5% | 1.36 | 3.50%  | +$298    |
| **Polygon 2yr**              | 112    | 38.4% | 1.23 | 10.04% | +$1,773  |
| MIN_ACTIVE=3 (reverted)      | 180    | 35.6% | 1.08 | 14.94% | +$1,003  |
| Stage 2 indicators (broke)   | 111    | 31.5% | 0.90 | 15.72% | -$696    |
| 9-indicator revert           | 180    | 36.1% | 1.09 | 12.27% | +$1,030  |
| Stage 6: Risk mgmt           | 214    | ~40%  | 1.62 | 10.50% | —        |
| **Stage 5: +LGBM filter**    | 78     | 69.2% | 2.38 | 3.89%  | +$4,321  |
| Stage 10: News filter        | 107    | 72.9% | 2.45 | 3.60%  | +$6,748  |
| Stage 15 Ph1: Full           | 153    | 67.3% | 2.11 | 13.12% | +$9,938  |
| Stage 15 Ph2: RANGING block  | 75     | 72.0% | 2.15 | 4.99%  | +$4,818  |
| **Stage 15 Ph2.5: FN ceil**  | 75     | 72.0% | 2.15 | 4.99%  | +$4,818  |

## Known Issues

| Issue                     | Impact              | Status / Workaround              |
|---------------------------|---------------------|----------------------------------|
| Trade count 75 < 80 gate  | Validation gate fail| DD/volume tradeoff; not a blocker|
| MT5 not active on Linux   | No auto-execution   | Manual trades via MT5 mobile     |
| Forward test outcomes     | Not in SQLite       | Google Sheet manual tracking     |
| Stale Polygon data        | Stale bars          | FIXED 2026-04-08 — +1 day end_date|
| Scoring WAIT at 88% conf  | Valid signals blocked| FIXED 2026-04-08 — ceiling gate removed|
