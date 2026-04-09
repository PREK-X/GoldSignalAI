# Signal Flow

Source: REFERENCES.md

## Pipeline (ASCII)

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

## Meta-Decision Cascade (5 Rules)

| Rule | Name             | Action                                         |
|------|------------------|------------------------------------------------|
| R1   | HMM Hard Gate    | CRISIS -> block all; RANGING -> block all      |
| R2   | LGBM Soft Vote   | P(UP)<0.40 blocks BUY; P(UP)>0.60 blocks SELL |
| R3   | Confidence Adj   | +5% when TRENDING+LGBM agrees                  |
| R4   | Session Loss     | >=2 consecutive losses -> skip session          |
| R5   | News/Volatility  | ATR>2x block; ATR>1.5x reduce 50%; calendar    |

## Scoring Bonuses & Penalties

| Modifier                | Value | Condition                    |
|-------------------------|-------|------------------------------|
| ADX very strong trend   | +3%   | ADX > 40                     |
| Volume surge            | +2%   | Volume >= 2x average         |
| At strong S/R zone      | +3%   | S/R confirms direction       |
| Fib 61.8% (golden ratio)| +3%   | Price at golden level        |
| Candlestick pattern     | +2%ea | Confirming pattern (cap +6%) |
| Doji indecision         | -5%   | Doji detected                |

## Scoring Gate Values (in scoring.py — NOT config.py)

```
MIN_ACTIVE           = 4    # min active (bull+bear) indicators
MIN_DOMINANT         = 3    # min in dominant direction
SESSION_ACTIVE_HOURS = frozenset(range(13, 22))  # 13:00-21:59 UTC
```

> Do NOT raise MIN_DOMINANT to 4 — filters too aggressively with 9 indicators.
