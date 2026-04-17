"""
Sweep5 — challenge success probability optimization.
Builds on sweep3 (H1 variants) + funnel diagnosis. Targets configurable
freq levers: ceiling, session, MIN_CONFIDENCE. Tracks FN 1-Step speed.

Usage: venv/bin/python -m backtest.sweep5
"""
import logging, sys, os, pickle
logging.basicConfig(level=logging.WARNING)
for m in ("backtest.engine","analysis","signals","ml","data","hmmlearn",
          "infrastructure","urllib3"):
    logging.getLogger(m).setLevel(logging.WARNING)

from config import Config
import analysis.scoring as scoring

CACHE = os.path.join(Config.BASE_DIR, "data", "historical", "_sweep_cache.pkl")
from backtest.engine import _fetch_historical_data, BacktestConfig, run_backtest

if os.path.exists(CACHE):
    print("[Sweep5] Loading cached data...")
    with open(CACHE, "rb") as f:
        _m15, _h1 = pickle.load(f)
    print(f"[Sweep5] Loaded: M15={len(_m15):,}, H1={len(_h1):,}")
else:
    print("[Sweep5] Fetching Polygon data (will cache)...")
    _m15, _h1 = _fetch_historical_data(BacktestConfig())
    if _m15 is None or _h1 is None:
        print("FATAL: data fetch failed"); sys.exit(1)
    with open(CACHE, "wb") as f:
        pickle.dump((_m15, _h1), f)


def run(label, ceil=72, session=None, min_conf=65,
        h1_mode="agree", h1_wait_mult=0.5):
    Config.MAX_CONFIDENCE_PCT = ceil
    scoring.MAX_CONFIDENCE = ceil
    Config.MIN_CONFIDENCE_PCT = min_conf
    scoring.MIN_CONFIDENCE = min_conf
    Config.H1_FILTER_MODE = h1_mode
    Config.H1_WAIT_POSITION_MULT = h1_wait_mult
    Config.REQUIRE_H1_AGREEMENT = True
    scoring.SESSION_ACTIVE_HOURS = frozenset(session or range(13, 22))

    r = run_backtest(BacktestConfig(), m15_data=_m15.copy(), h1_data=_h1.copy())
    fn = next((s for s in r.prop_firm_sims if "1-Step" in s.firm_name), None)
    ap = sum(1 for s in r.prop_firm_sims if s.passed)
    fn_days = fn.days_to_complete if fn and fn.passed else None
    fn_pnl = fn.final_pnl_pct if fn else 0.0
    fn_dd = fn.max_drawdown_pct if fn else 0.0
    fn_str = f"{fn_days}d/{fn_pnl:+.1f}%/{fn_dd:.1f}%" if fn_days else "FAIL"
    print(f"| {label:50s} | {r.total_trades:>5} | {r.win_rate_pct:>5.1f}% | "
          f"{r.profit_factor:>5.2f} | {r.max_drawdown_pct:>5.2f}% | "
          f"{r.sharpe_ratio:>5.2f} | {fn_str:>18s} | {ap}/8 |",
          flush=True)


if __name__ == "__main__":
    H = f"| {'Experiment':50s} | {'Trd':>5} | {'WR':>6} | {'PF':>5} | {'DD':>6} | {'Shrp':>6} | {'FN days/pnl/dd':>18s} | {'All':>3} |"
    S = f"|{'-'*52}|{'-'*7}|{'-'*8}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*20}|{'-'*5}|"
    print(f"\n{'='*len(H)}\n{H}\n{S}")

    # Baseline reference (current config)
    run("BASELINE (ceil=72, min_conf=65, sess 13-22)")
    # Ceiling lift — frequency expansion
    run("ceil=74", ceil=74)
    run("ceil=75", ceil=75)
    run("ceil=76", ceil=76)
    # Session widen — add hour 12 (pre-NY) and hour 22 (late NY)
    run("ceil=72 + sess 12-22", session=range(12, 23))
    run("ceil=75 + sess 12-22", ceil=75, session=range(12, 23))
    # MIN_CONFIDENCE floor — open 60-65 band
    run("ceil=75 + min_conf=60", ceil=75, min_conf=60)
    # Combined winners
    run("ceil=75 + min_conf=60 + sess 12-22",
        ceil=75, min_conf=60, session=range(12, 23))

    print(f"{'='*len(H)}\n")
