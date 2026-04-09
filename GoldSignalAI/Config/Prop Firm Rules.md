# Prop Firm Rules

Source: REFERENCES.md + CONTEXT.md

## Active Challenge: FundedNext 1-Step $10k

| Metric        | Limit | Backtest Actual | Notes                        |
|---------------|-------|-----------------|------------------------------|
| Daily loss    | 3.0%  | 2.13% (ceiling) | Pre-emptive 2.8% ceiling     |
| Total DD      | 6.0%  | 4.99%           | Trailing from peak           |
| Profit target | 10.0% | +48.2%          | Challenge passed in backtest |
| Min days      | 0     | N/A             | No minimum                   |
| Daily ceiling | 2.8%  | Enforced        | Pre-emptive block in config.py|

Cost: $99 fee. Next step after 20 profitable demo trades on IC Markets.

## All Supported Firm Limits

| Firm           | Daily Loss | Total DD | Profit | Min Days |
|----------------|-----------|----------|--------|----------|
| FundedNext 1S  | 3.0%      | 6.0%     | 10.0%  | 0        |
| FundedNext 2S  | 5.0%      | 10.0%    | 8.0%   | 5        |
| FTMO           | 5.0%      | 10.0%    | 10.0%  | 4        |
| The5ers        | 4.0%      | 6.0%     | 6.0%   | 0        |
| E8 Funding     | 5.0%      | 8.0%     | 8.0%   | 0        |
| MyForexFunds   | 5.0%      | 12.0%    | 8.0%   | 0        |
| Apex           | 3.0%      | 6.0%     | 9.0%   | 9        |

All 8 prop firm simulations pass in backtest (Stage 15 Ph2.5).

## Enforcement Logic

- **Daily ceiling (2.8%):** checked in `propfirm/tracker.py` → `is_trading_allowed()`
  before the 3.0% hard limit. Fires when `daily_loss_pct >= 2.8`.
- **Max DD (6.0%):** checked first in `is_trading_allowed()` — takes priority over daily.
- **Challenge state:** persisted in `state/challenge_state.json`
- **Backtest enforcement:** `_simulate_prop_firm()` in `backtest/engine.py`

## Forward Test (Current Phase)

| Field            | Value                             |
|------------------|-----------------------------------|
| Broker           | IC Markets (Raw Spread, MT5)      |
| Account          | 52791555                          |
| Server           | ICMarketsGlobal-Demo              |
| Execution        | Manual via MT5 mobile app         |
| Outcome tracking | Google Sheet (not SQLite)         |
| Target           | 20 demo trades profitable         |
| Next step        | FundedNext $10k 1-Step ($99 fee)  |
