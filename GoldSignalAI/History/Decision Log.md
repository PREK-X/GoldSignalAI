# Decision Log

Source: CLAUDE.md — Key Decisions and Why

| Decision                  | Reasoning                                                                 |
|---------------------------|---------------------------------------------------------------------------|
| Active-ratio scoring      | Old `/10` made 70% confidence unreachable. New: `dominant/(bull+bear)` ignores neutrals |
| NY session only           | Diagnostic on 277 signals: NY 63.3% WR vs London 33.9%. Session filter is the single biggest edge |
| ML disabled               | XGBoost 47% CV (trained on indicator outputs = redundant). Need independent features → macro pipeline built |
| SL = ATR × 1.5 (~130 pips)| Gold M15 median candle = 125 pips. Old 30-pip SL = noise stop-out every trade |
| 9 indicators FROZEN       | Adding 4 more in Stage 2 dropped PF 1.23 → 0.90. Do not add without per-indicator backtest |
| RANGING blocked (not reduced)| RANGING trades avg $+17.87 vs $+81.64 TRENDING, with disproportionate DD |
| FN daily ceiling 2.8%    | Pre-emptive block below 3.0% hard limit. Dropped max daily loss from 3.00% to 2.13% |
| 38% base WR is fine       | With 3.3:1 R:R, break-even is 23%. High WR is a bonus, not a requirement |
| PrecomputedIndicators     | Computing 12 indicators per bar on 48k bars takes hours without the shim in indicators.py |
| MIN_ACTIVE = 4 (not 3)   | MIN_ACTIVE=3 caused PF→1.08 and DD→15%. Reverted. |
| MIN_DOMINANT = 3 (not 4) | MIN_DOMINANT=4 filters too aggressively with only 9 indicators |
| BBands removed from voting| 42.3% voting accuracy — worse than random |
| Polygon over yfinance     | yfinance has 60-day hard limit; need 2yr history for valid backtest |
| Separate H1 fetch         | H1 resampled from M15 gives wrong H1 values |
| Sequential Polygon fetch  | Parallel triggers 429 rate limit on free tier |
