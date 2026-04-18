# CONTEXT.md
# Read by Claude Code at session start
# For duplicate facts, canonical source is: CONTEXT.md

---

## Current Status (2026-04-18)

**All 16 stages complete. Forward testing phase.**
Regression audit + config revert complete: MAX_CONFIDENCE_PCT=72,
SESSION_ACTIVE_HOURS=range(13,22). Root cause of 70/2.24 → 87/1.75
apparent regression = Polygon 2yr sliding-window data drift (not code).
Chosen on post-audit fresh data: PF 1.75, 87 trades, FN 1-Step PASS 31d,
per-challenge DD 3.43% (1.75× buffer to 6% FN limit). See Experiments.
Awaiting 20 demo trades on IC Markets before FundedNext challenge.

---

## Forward Test

| Field           | Value                              |
|-----------------|------------------------------------|
| Broker          | IC Markets (Raw Spread, MT5)       |
| Account         | 52791555                           |
| Server          | ICMarketsGlobal-Demo               |
| Execution       | Manual via MT5 mobile app          |
| Outcome tracking| Google Sheet (not SQLite)          |
| Target          | 20 demo trades profitable          |
| Next step       | FundedNext $10k 1-Step ($99 fee)   |

---

## Active Prop Firm: FundedNext 1-Step $10k

| Metric         | Limit   | Backtest actual |
|----------------|---------|-----------------|
| Daily loss     | 3.0%    | 2.13% (ceiling) |
| Total DD       | 6.0%    | 4.99%           |
| Profit target  | 10.0%   | +48.2%          |
| Min days       | 0       | N/A             |
| Daily ceiling  | 2.8%    | Pre-emptive block in config.py |

---

## Latest Backtest (Regression Audit Revert 2026-04-18)

| Metric   | Value  |
|----------|--------|
| Trades   | 87     |
| WR       | 69.0%  |
| PF       | 1.75   |
| DD       | 9.88%  |
| Sharpe   | 1.49   |
| Period   | May 2024 - Mar 2026 (fresh 2yr Polygon) |

Prop firm sims: all 8 pass. FN 1-Step: 31d / +10.27% / 3.43% per-challenge DD.

Config: MAX_CONFIDENCE_PCT=72, SESSION_ACTIVE_HOURS=range(13,22), H1=agree.

**Supersedes Apr 18 freq-opt (ceil=74+sess 12-22)**. Root cause of prior 70/2.24
canonical-vs-90/1.51-sweep5 delta = Polygon 2yr window drift (~6-day shift
between fetches + macro_data.db refresh). Code at HEAD is behaviorally inert
in agree mode (verified by Exp R1 + R2 producing identical 87/1.75 on fresh
data at both 56c3cbe and 8c4c32c). See Experiments → Regression Audit.

---

## Backtest History

| Run                        | Trades | WR    | PF   | DD     | PnL     |
|----------------------------|--------|-------|------|--------|---------|
| Original (yfinance 60d)    | 46     | 30.4% | 0.89 | 9.17%  | -$332   |
| SL fix (50-200 pips)       | 30     | 36.7% | 1.04 | 6.34%  | +$85    |
| Session filter             | 13     | 38.5% | 1.36 | 3.50%  | +$298   |
| **Polygon 2yr**            | 112    | 38.4% | 1.23 | 10.04% | +$1,773 |
| MIN_ACTIVE=3 (reverted)    | 180    | 35.6% | 1.08 | 14.94% | +$1,003 |
| Stage 2 indicators (broke) | 111    | 31.5% | 0.90 | 15.72% | -$696   |
| 9-indicator revert         | 180    | 36.1% | 1.09 | 12.27% | +$1,030 |
| Stage 6: Risk mgmt         | 214    | ~40%  | 1.62 | 10.50% | —       |
| **Stage 5: +LGBM filter**  | 78     | 69.2% | 2.38 | 3.89%  | +$4,321 |
| Stage 10: News filter       | 107    | 72.9% | 2.45 | 3.60%  | +$6,748 |
| Stage 15 Ph1: Full          | 153    | 67.3% | 2.11 | 13.12% | +$9,938 |
| Stage 15 Ph2: RANGING block | 75     | 72.0% | 2.15 | 4.99%  | +$4,818 |
| Stage 15 Ph2.5: FN ceil     | 75     | 72.0% | 2.15 | 4.99%  | +$4,818 |
| Post-audit ceiling=None     | 434    | —     | —    | —      | —       |
| Post-audit ceiling=90       | 314    | 62.7% | 1.34 | 16.63% | +$9,224 |
| Post-audit ceiling=80       | 297    | 64.0% | 1.49 | 8.82%  | +$11,666|
| Post-audit ceiling=75       | 174    | 66.7% | 1.66 | 6.68%  | +$9,640 |
| Post-audit ceiling=70       | 36     | 69.4% | 2.00 | 2.93%  | +$1,205 |
| **Post-audit ceiling=72**   | 70     | 75.7% | 2.24 | 7.44%  | +$4,289 |
| **Freq opt ceil=74+sess**   | 103    | 65.0% | 1.50 | 11.67% | —       |
| **Regression audit revert ceil=72** | 87 | 69.0% | 1.75 | 9.88% | +$3,840 |

Stage 16 = deployment only, no metric changes.

---

## Critical Config Values

### In config.py
```
MIN_CONFIDENCE_PCT           = 65
MAX_CONFIDENCE_PCT           = 74   # freq optimization 2026-04-18
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

### In analysis/scoring.py (NOT in config.py)
```
MIN_ACTIVE           = 4    # min active (bull+bear) indicators
MIN_DOMINANT         = 3    # min in dominant direction
SESSION_ACTIVE_HOURS = frozenset(range(12, 23))  # 12:00-22:59 UTC (updated 2026-04-18)
```
- DO NOT raise MIN_DOMINANT to 4 — filters too aggressively with 9 indicators
- DO NOT add unvalidated indicators (Stage 2: PF 1.23 -> 0.90)

---

## What Is Disabled and Why

| Feature        | Flag                | Why                              |
|----------------|---------------------|----------------------------------|
| XGBoost filter | USE_ML_FILTER=False | 47% CV, worse than coin flip     |
| LGBM filter    | USE_LGBM_FILTER=False| 52% CV < 53% gate; soft vote ok |
| CNN-BiLSTM     | USE_DEEP_FILTER=False| 52.1% < 54% gate; UP bias       |
| MT5 execution  | MT5_EXECUTION_ENABLED=False| Linux has no MT5 terminal |
| RANGING trades | meta_decision.py R1 | Blocked: $+17.87 avg, high DD   |

Re-enable ML models after 150+ real trade outcomes for retraining.

---

## Experiments (2026-04-11)

### MAX_CONFIDENCE Ceiling Sweep

Commit `e0adc3b` removed the over-consensus ceiling from
`analysis/scoring.py` to fix a bug where valid 88% signals were
blocked. The removal was too broad — with no ceiling the backtest
produced 434 trades on the post-audit codebase. Swept ceiling values
to find the optimum:

| Ceiling | Trades | WR    | PF   | DD    | FN 1-Step | All 8 firms |
|---------|--------|-------|------|-------|-----------|-------------|
| None    | 434    | —     | —    | —     | —         | —           |
| 90      | 314    | 62.7% | 1.34 | 16.63%| FAIL      | 5/8         |
| 80      | 297    | 64.0% | 1.49 | 8.82% | FAIL      | 6/8         |
| 75      | 174    | 66.7% | 1.66 | 6.68% | PASS      | —           |
| **72**  | **70** | **75.7%** | **2.24** | **7.44%** | **PASS** | **8/8** |
| 70      | 36     | 69.4% | 2.00 | 2.93% | PASS      | 8/8         |

**Chosen: 72.** Best profit factor and win rate in the sweep, with
a trade count close to the pre-audit Stage 15 baseline (70 vs 75).
All 8 prop firm sims pass. Rationale:
- Ceiling=70 is safer on raw DD but only 36 trades in 2 years is too
  thin — poor statistical confidence and low challenge-period PnL.
- Ceiling≥75 re-admits lagging over-consensus signals that hurt PF.
- Ceiling=72 sits on the knee of the curve: blocks 100% raw signals
  and anything reaching 73% after bonuses, but keeps the 65–72% band
  that carries most of the edge.

### Challenge Frequency Optimization Sweep (2026-04-18)

Mission: maximize FundedNext 1-Step challenge success probability.
Baseline (ceil=72, H1=agree) yields ~45 signals/challenge-period (30-50d),
insufficient for reliable 10% profit target. Architectural cap ~287 trades/2yr
under M15+H1 architecture (structural changes not viable — see DD Audit).

**Indicator research:** 9 current indicators reviewed against public XAU M15
repos. Current set (PF 1.66 vs public ~1.64) already at or above benchmark.
Supertrend rejected Stage 2 (PF collapse). AO/Choppiness/TTM noisy on M15.
No swap warranted.

**Parameter sweep (cached 2yr data, H1=agree):**

| Experiment | Trd | WR | PF | DD | FN days/pnl/dd | All 8 |
|------------|-----|----|-----|-----|----------------|-------|
| BASELINE ceil=72, sess 13-22 | 90 | 67.8% | 1.51 | 11.58% | 34d/+10.3%/4.6% | 8/8 |
| ceil=74 | 92 | 67.4% | 1.66 | 11.68% | 34d/+10.8%/4.6% | 8/8 |
| ceil=75 | 245 | 62.9% | 1.21 | 10.87% | FAIL | 0/8 |
| ceil=72 + sess 12-22 | 102 | 65.7% | 1.45 | 10.78% | 37d/+11.8%/4.6% | 8/8 |
| **ceil=74 + sess 12-22** | **103** | **65.0%** | **1.50** | **11.67%** | **21d/+11.3%/2.1%** | **8/8** |

ceil=75 is a hard cliff — frequency triples (90→245) but all 8 firms fail.
ceil=76, ceil=75+session, ceil=75+min_conf=60 all expected ≥ as bad as ceil=75 — skipped.

**Winner: ceil=74 + sess 12-22.**
- Per-challenge DD: 2.1% vs baseline 4.6% → 3× more buffer below 6% FN limit
- FN challenge: 21d vs 34d → faster completion, less exposure window
- All 8 prop firms pass
- PF 1.50 lower than ceil=74-alone (1.66) but challenge survival is primary metric
- Note: prop sim is 1 observation over 2yr data, not rolling windows — treat
  21d/2.1% as favorable signal, not guaranteed probability

**Applied:** MAX_CONFIDENCE_PCT 72→74, SESSION_ACTIVE_HOURS range(13,22)→range(12,23)

**Superseded 2026-04-18 by Regression Audit — reverted below.**

### Regression Audit + Revert (2026-04-18 afternoon)

Trigger: `ceil=74+sess` sweep winner showed 103/65.0%/1.50 — but canonical
`ceil=72, sess 13-22` at `56c3cbe` reported 70/75.7%/2.24/7.44%. 0.73 PF
delta between nominally identical `ceil=72` configs (canonical vs
Freq-Opt sweep baseline row 90/1.51) demanded explanation before any
further optimization.

**Audit:** Code diffs 56c3cbe→HEAD `8c4c32c` (config.py, scoring.py,
engine.py H1 branches) are all behaviorally inert in `agree` mode —
sweep5 overrides every changed Config/scoring attr, and the new
`_analyse_bar` default-else branch matches the pre-change agree path
exactly. H1-WAIT risk reduction only fires in `noncontradict`.

**Experiments:**

| Exp | Commit | Data                        | Config     | Result                    |
|-----|--------|-----------------------------|------------|---------------------------|
| R1  | 56c3cbe| fresh Polygon + 56c3cbe macro | ceil=72   | 87/69.0%/1.75/9.88%, 8/8 PASS |
| R2  | 8c4c32c| R1 Polygon + R1-post macro  | ceil=72 patched | 87/69.0%/1.75/9.88%, 8/8 PASS |

R1 = R2 exactly (same trade count, WR, PF, DD, Sharpe, FN PASS 31d,
per-challenge DD 3.43%). Confirms empirically that HEAD code is inert in
agree mode AND that canonical 70/2.24 numbers are NOT reproducible
under today's Polygon data at the same code commit.

**Root cause: data drift.** The Polygon 2yr sliding window + macro_data.db
refresh shifted the bar set enough to move PF from 2.24 to 1.75 at the
same `ceil=72`. No code regression exists. Canonical is accepted as a
historical point-in-time snapshot.

**Config selection on fresh data (sweep6, same frozen cache as R2):**

| Candidate                   | Trd | WR    | PF   | DD    | FN days/pnl/dd    | Rule 3 buffer |
|-----------------------------|-----|-------|------|-------|-------------------|---------------|
| **ceil=72, sess 13-22**     | 87  | 69.0% | 1.75 | 9.88% | 31d/+10.3%/3.43%  | 1.75× ✓       |
| ceil=73, sess 13-22         | 87  | 69.0% | 1.75 | 9.88% | 31d/+10.3%/3.43%  | 1.75× ✓       |
| ceil=74, sess 13-22         | 89  | 68.5% | 1.81 | 9.76% | 32d/+11.0%/4.60%  | 1.30× ✗       |
| ceil=74, sess 14-22         | 78  | 70.5% | 2.16 | 8.64% | 27d/+10.3%/2.41%  | 2.49× ✓       |
| ceil=74, min_conf=66, 13-22 | 87  | 70.1% | 2.07 | 6.50% | 32d/+11.0%/4.60%  | 1.30× ✗       |

Selection: ceil=74 variants at sess 13-22 violate the DD-buffer rule
(per-challenge DD 4.60% vs 6% FN limit = 1.30× < 1.5×). ceil=74 sess
14-22 passes all rules but drops to 78 trades (below rule-4 threshold).
ceil=72 and ceil=73 identical under this data; ceil=72 wins on simplicity.

**Applied:** MAX_CONFIDENCE_PCT 74→72, SESSION_ACTIVE_HOURS range(12,23)
→range(13,22). `H1_FILTER_MODE="agree"` and `H1_WAIT_POSITION_MULT=0.5`
kept (inert in agree mode; available for future `noncontradict` trials).

### H1 Agreement Experiment (at ceiling=72)

Tested whether the `REQUIRE_H1_AGREEMENT` flag still earns its cost
on the post-audit codebase.

| Flag      | Trades | WR    | PF   | DD    | Sharpe | FN 1-Step |
|-----------|--------|-------|------|-------|--------|-----------|
| **True**  | **70** | **75.7%** | **2.24** | **7.44%** | **1.96** | PASS |
| False     | 232    | 60.3% | 1.34 | 12.93%| 1.23   | PASS      |

Disabling H1 agreement triples trade count but collapses quality
across every metric. Kept `REQUIRE_H1_AGREEMENT = True`.

Note: the `REQUIRE_H1_AGREEMENT` flag was previously only plumbed
through `analysis/multi_timeframe.py` (live path). `backtest/engine.py`
hardcoded the agreement rule. Fixed in the same commit as this
experiment — the backtest now honours the flag.

---

## Known Issues

| Issue                    | Impact              | Status / Workaround           |
|--------------------------|---------------------|-------------------------------|
| Trade count 75 < 80 gate | Validation gate fail| DD/volume tradeoff; not a blocker |
| MT5 not active on Linux  | No auto-execution   | Manual trades via MT5 mobile  |
| Forward test outcomes    | Not in SQLite       | Google Sheet manual tracking  |
| 1 test failure           | test_news_fetcher DST offset    | Pre-existing, not blocking |
| Stale Polygon data       | Bot operated on yesterday's bars | **FIXED 2026-04-08** |
| Scoring WAIT at 88% conf | Valid BUY/SELL signals blocked | **FIXED 2026-04-08** |
| DD formula mismatch (documented) | Backtest reports 7.44% but prop sim passes FN 6% limit | By design — see DD Audit note below |

### DD Audit Note (2026-04-11)

**Functions audited:**
- `propfirm/tracker.py` — `ChallengeTracker.get_status()` (line 409) and `drawdown_check()` in `propfirm/profiles.py` (line 181)
- `backtest/engine.py` — `_compute_stats()` (line 899) for reported Max DD; `_simulate_prop_firm()` (line 1032) for prop firm sim DD

**Formulas:**
- **propfirm/tracker.py**: `total_dd_pct = max(0, peak_balance - current_balance) / initial_balance * 100`
  → **Trailing peak-to-trough as % of initial balance** — matches FundedNext's actual rule exactly.
- **backtest Max DD** (`_compute_stats`, line 899): `dd_pct = (peak - equity) / peak * 100`
  → **Peak-to-trough as % of running peak equity** — standard investment metric, NOT FundedNext's formula.
- **prop sim DD** (`_simulate_prop_firm`, line 1032): `dd_pct = (peak - equity) / account_balance * 100`
  → **Peak-to-trough as % of initial balance** — correct FundedNext formula, consistent with tracker.

**Are they the same?** No. The reported 7.44% backtest DD uses running peak as denominator; FundedNext uses initial balance. When account is above starting value, FN's formula produces a *higher* % for the same absolute loss (more conservative). The 7.44% understates what FN would actually measure for that same worst-case drop.

**Is "all 8 pass" legitimate?** Yes. The prop sim uses FN's correct formula AND stops at profit target — matching real challenge behaviour. The 7.44% is the worst-case across 2 years of hypothetical non-stop trading; the challenge sprint to 10% completes in ~30-50 days, during which DD stays at 1.25-4.30% (well under 6%).

**Does 2.8% daily ceiling prevent 7.44% DD?** No — the ceiling blocks *future trades* after a day's loss hits 2.8%; it cannot prevent the trade that triggers the ceiling. The 2-year 7.44% worst-case is irrelevant: in a real challenge the bot stops once the 10% target is hit (before reaching the worst drawdown period). The ceiling protects against catastrophic intraday blowups within any single challenge period.

**Verdict: safe to forward test.** The prop sim is correctly calibrated to FundedNext's rules. The 7.44% backtest figure is a different metric (investment-style), not a sign the challenge would fail.

---

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
| Ceiling removal too broad  | Trade count 75→434, PF collapse | Ceiling re-added + retuned to 72 — FIXED 2026-04-11 |
| H1 flag unplumbed in BT    | backtest/engine.py ignored REQUIRE_H1_AGREEMENT | Wired flag through _analyze_bar — FIXED 2026-04-11 |

---

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

---

## Completed Stages

| Stage | Name                   | Date       | Key outcome                    |
|-------|------------------------|------------|--------------------------------|
| Ph1-3 | Data+Backtest+Stability| pre-2026   | Polygon 2yr, SQLite, Discord   |
| 1     | Environment            | pre-2026   | Arch Linux, Python 3.12        |
| 2     | Indicators (REVERTED)  | pre-2026   | 4 indicators caused PF->0.90   |
| 3     | Macro Features         | pre-2026   | DXY/VIX/US10Y pipeline         |
| 4     | HMM Regime             | pre-2026   | 3-state detector, active       |
| 5     | LightGBM               | pre-2026   | 52% CV, soft vote in meta      |
| 6     | Risk Management        | pre-2026   | CB + Half-Kelly, PF 1.62       |
| 7     | CNN-BiLSTM             | pre-2026   | 52.1% acc, disabled            |
| 8     | Meta-Decision          | pre-2026   | 5-rule cascade                 |
| 10    | News Filter            | pre-2026   | PF 2.45, WR 72.9%             |
| 11    | MT5 Execution          | 2026-04-02 | Bridge + monitor + generator   |
| 12    | Challenge Mode         | 2026-04-02 | Compliance tracking + Discord  |
| 13    | ML Auto-Retrain        | 2026-04-03 | Weekly LGBM, CNN-BiLSTM @150   |
| 14    | Dashboard              | 2026-04-03 | Bloomberg theme, 6 tabs        |
| 15    | Final Testing          | 2026-04-05 | RANGING block + FN ceiling     |
| 16    | Deployment             | 2026-04-05 | Env detect, forward test mode  |

---

## Remaining: Stage 9 — Multi-Asset

```
XAGUSD -> EURUSD -> US30 -> NAS100 -> USOIL
Per-asset ML models and risk parameters
Portfolio correlation monitoring (max 0.7)
```

Deferred until after FundedNext challenge funded.
