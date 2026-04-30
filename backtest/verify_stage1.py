"""
Stage 1 verification — re-run committed config on FRESH data.

Re-fetches M15+H1 (overwrites stage17 cache) and runs ONLY the committed
config (D3_C58_agree_S13-21). Faster than full sweep; gives apples-to-apples
result vs `backtest.engine` on today's data.

Usage:
    venv/bin/python -m backtest.verify_stage1
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for mod in ("backtest.engine", "analysis", "signals", "ml", "data", "hmmlearn",
            "infrastructure", "urllib3", "polygon"):
    logging.getLogger(mod).setLevel(logging.WARNING)

from config import Config
import analysis.scoring as scoring
from backtest.engine import BacktestConfig, _fetch_historical_data, run_backtest

CACHE_PATH = Path("data/historical/stage17_sweep_cache.pkl")


def main():
    if CACHE_PATH.exists():
        print(f"[Verify] Loading cached data from {CACHE_PATH}")
        with CACHE_PATH.open("rb") as f:
            m15, h1 = pickle.load(f)
        print(f"[Verify] Cache loaded: M15={len(m15):,}, H1={len(h1):,}\n")
    else:
        print("[Verify] Re-fetching fresh M15+H1 data...")
        base_cfg = BacktestConfig()
        m15, h1 = _fetch_historical_data(base_cfg)
        if m15 is None or h1 is None:
            print("FATAL: data fetch failed"); sys.exit(1)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("wb") as f:
            pickle.dump((m15, h1), f)
        print(f"[Verify] Cache saved: M15={len(m15):,}, H1={len(h1):,}\n")

    # Apply committed config (D3_C65_agree_S13-21 — control)
    Config.MIN_CONFIDENCE_PCT = 65
    Config.H1_FILTER_MODE     = "agree"
    scoring.MIN_CONFIDENCE    = 65
    scoring.MIN_DOMINANT      = 3
    scoring.SESSION_ACTIVE_HOURS = frozenset(range(13, 22))

    print("[Verify] Running backtest with committed config: "
          "MIN_DOMINANT=3, MIN_CONFIDENCE=65, H1=agree, SESSION=13-21\n")
    cfg = BacktestConfig()
    result = run_backtest(cfg, m15_data=m15.copy(), h1_data=h1.copy())

    fn = next((s for s in result.prop_firm_sims if "1-Step" in s.firm_name), None)
    fn_dd = fn.max_drawdown_pct if fn else 0.0
    fn_buf = (6.0 / fn_dd) if fn_dd > 0 else 0.0
    all_pass = sum(1 for s in result.prop_firm_sims if s.passed)

    print("\n========== STAGE 1 VERIFICATION RESULT ==========")
    print(f"  Trades:        {result.total_trades}")
    print(f"  Win Rate:      {result.win_rate_pct:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Max DD:        {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe:        {result.sharpe_ratio:.2f}")
    print(f"  FN-1Step:      {'PASS' if fn and fn.passed else 'FAIL'} | "
          f"per-challenge DD={fn_dd:.2f}% | buffer={fn_buf:.2f}×")
    print(f"  All firms:     {all_pass}/8 PASS")
    print("=================================================\n")
    for s in result.prop_firm_sims:
        print(f"  {s.firm_name:30s} {'PASS' if s.passed else 'FAIL':4s}  "
              f"DD={s.max_drawdown_pct:.2f}%  PnL={s.final_pnl_pct:+.2f}%  "
              f"days={s.days_to_complete}")


if __name__ == "__main__":
    main()
