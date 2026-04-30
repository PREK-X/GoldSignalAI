"""
Stage 17 frequency sweep — fetch once, sweep 24 + control = 25 configs.

Grid:
    MIN_DOMINANT          ∈ {2, 3}
    MIN_CONFIDENCE_PCT    ∈ {58, 60, 62}
    H1_FILTER_MODE        ∈ {"noncontradict", "agree"}
    SESSION_ACTIVE_HOURS  ∈ {range(12,23), range(13,22)}
    + 1 control (current committed config)

Hard gates: trades ≥ 500, PF ≥ 1.50, WR ≥ 60%, FN buffer ≥ 1.30×, 8/8 firms PASS.

Usage:
    venv/bin/python -m backtest.sweep_stage17
    venv/bin/python -m backtest.sweep_stage17 --refresh   # force re-fetch cache
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from itertools import product
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
for mod in ("backtest.engine", "analysis", "signals", "ml", "data", "hmmlearn",
            "infrastructure", "urllib3", "polygon"):
    logging.getLogger(mod).setLevel(logging.WARNING)

from config import Config
import analysis.scoring as scoring
from backtest.engine import (
    BacktestConfig,
    _fetch_historical_data,
    run_backtest,
)

CACHE_PATH   = Path("data/historical/stage17_sweep_cache.pkl")
RESULTS_PATH = Path("data/historical/stage17_sweep_results.json")


# ── Step 1.1: Freeze data ───────────────────────────────────────────────
def load_or_fetch(refresh: bool = False):
    if not refresh and CACHE_PATH.exists():
        print(f"[Sweep] Loading cached data from {CACHE_PATH}")
        with CACHE_PATH.open("rb") as f:
            m15, h1 = pickle.load(f)
        print(f"[Sweep] Loaded: M15={len(m15):,} bars, H1={len(h1):,} bars\n")
        return m15, h1

    print("[Sweep] Fetching M15 + H1 from data source (one-time)...")
    base_cfg = BacktestConfig()
    m15, h1 = _fetch_historical_data(base_cfg)
    if m15 is None or h1 is None:
        print("FATAL: data fetch failed")
        sys.exit(1)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as f:
        pickle.dump((m15, h1), f)
    print(f"[Sweep] Cached to {CACHE_PATH}: M15={len(m15):,}, H1={len(h1):,}\n")
    return m15, h1


# ── Step 1.2: Sweep harness ─────────────────────────────────────────────
def patch_and_run(m15_data, h1_data, *, min_dominant: int,
                  min_confidence: int, h1_mode: str,
                  session_hours: range):
    """Monkey-patch all 4 levers, run backtest, return summary dict."""
    Config.MIN_CONFIDENCE_PCT = min_confidence
    Config.H1_FILTER_MODE     = h1_mode
    scoring.MIN_CONFIDENCE    = min_confidence       # captured at module import
    scoring.MIN_DOMINANT      = min_dominant
    scoring.SESSION_ACTIVE_HOURS = frozenset(session_hours)

    cfg = BacktestConfig()
    result = run_backtest(cfg, m15_data=m15_data.copy(), h1_data=h1_data.copy())

    fn = next((s for s in result.prop_firm_sims if "1-Step" in s.firm_name), None)
    # PropFirmSimulation.max_drawdown_pct is per-challenge (audit c88c837).
    fn_dd_per_challenge = fn.max_drawdown_pct if fn else 0.0
    fn_buffer = (6.0 / fn_dd_per_challenge) if fn_dd_per_challenge > 0 else 0.0
    all_pass = sum(1 for s in result.prop_firm_sims if s.passed)

    return {
        "trades":     result.total_trades,
        "wr":         result.win_rate_pct,
        "pf":         result.profit_factor,
        "dd":         result.max_drawdown_pct,
        "sharpe":     result.sharpe_ratio,
        "fn_pass":    fn.passed if fn else False,
        "fn_dd":      fn_dd_per_challenge,
        "fn_buffer":  fn_buffer,
        "all_pass":   all_pass,
    }


def passes_gates(r: dict) -> bool:
    return (r["trades"]   >= 500
        and r["pf"]       >= 1.50
        and r["wr"]       >= 60.0
        and r["fn_buffer"] >= 1.30
        and r["all_pass"] == 8)


def composite_score(r: dict) -> float:
    """Pareto rank when no row passes all gates."""
    return r["trades"] * r["pf"] * (r["wr"] / 100.0) * max(r["fn_buffer"], 0.01)


def row_key(md: int, mc: int, hm: str, sh_start: int, sh_stop: int) -> str:
    return f"D{md}_C{mc}_H{hm}_S{sh_start}-{sh_stop - 1}"


def load_results_cache() -> dict:
    if RESULTS_PATH.exists():
        try:
            with RESULTS_PATH.open("r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_results_cache(cache: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, indent=2, default=str)
    tmp.replace(RESULTS_PATH)


# ── Step 1.3: Run sweep ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch (ignore cache)")
    args = parser.parse_args()

    m15, h1 = load_or_fetch(refresh=args.refresh)

    # Snapshot original to restore between rows / for control
    orig = {
        "min_conf":   Config.MIN_CONFIDENCE_PCT,
        "h1_mode":    Config.H1_FILTER_MODE,
        "min_dom":    scoring.MIN_DOMINANT,
        "sess":       sorted(scoring.SESSION_ACTIVE_HOURS),
    }
    print(f"[Sweep] Control snapshot: {orig}\n")

    results_cache = load_results_cache()
    print(f"[Sweep] Results cache has {len(results_cache)} prior rows. "
          f"Resuming — done rows will be skipped.\n")

    rows = []

    # 24-row grid
    grid = list(product(
        [2, 3],                     # MIN_DOMINANT
        [58, 60, 62],               # MIN_CONFIDENCE_PCT
        ["noncontradict", "agree"], # H1_FILTER_MODE
        [range(12, 23), range(13, 22)],  # SESSION_ACTIVE_HOURS
    ))

    hdr = (f"| {'#':>2} | {'MinDom':>6} | {'MinConf':>7} | {'H1Mode':>13} | "
           f"{'Sess':>7} | {'Trd':>5} | {'WR':>5} | {'PF':>5} | "
           f"{'DD':>5} | {'Shrp':>5} | {'FNDD':>5} | {'Buf':>5} | "
           f"{'8Frm':>4} | {'Gate':>4} |")
    sep = "|" + "|".join("-" * w for w in
                         [4, 8, 9, 15, 9, 7, 7, 7, 7, 7, 7, 7, 6, 6]) + "|"
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for i, (md, mc, hm, sh) in enumerate(grid, 1):
        key = row_key(md, mc, hm, sh.start, sh.stop)
        sess_str = f"{sh.start}-{sh.stop-1}"
        if key in results_cache:
            r = results_cache[key]
            tag = "(cache)"
        else:
            r = patch_and_run(m15, h1,
                              min_dominant=md, min_confidence=mc,
                              h1_mode=hm, session_hours=sh)
            r["label"] = f"D{md}_C{mc}_H{hm[:5]}_S{sh.start}-{sh.stop-1}"
            r["params"] = [md, mc, hm, list(sh)]
            results_cache[key] = r
            save_results_cache(results_cache)
            tag = "       "
        rows.append(r)
        print(f"| {i:>2} | {md:>6} | {mc:>7} | {hm:>13} | {sess_str:>7} | "
              f"{r['trades']:>5} | {r['wr']:>4.1f}% | {r['pf']:>5.2f} | "
              f"{r['dd']:>4.2f}% | {r['sharpe']:>5.2f} | {r['fn_dd']:>4.2f}% | "
              f"{r['fn_buffer']:>5.2f} | {r['all_pass']:>1d}/8 | "
              f"{'PASS' if passes_gates(r) else 'FAIL':>4s} | {tag} |", flush=True)

    # Control: current committed config
    ctrl_sh = range(orig["sess"][0], orig["sess"][-1] + 1)
    ctrl_key = row_key(orig["min_dom"], orig["min_conf"], orig["h1_mode"],
                       ctrl_sh.start, ctrl_sh.stop) + "_CTRL"
    if ctrl_key in results_cache:
        r0 = results_cache[ctrl_key]
        tag = "(cache)"
    else:
        r0 = patch_and_run(m15, h1,
                           min_dominant=orig["min_dom"],
                           min_confidence=orig["min_conf"],
                           h1_mode=orig["h1_mode"],
                           session_hours=ctrl_sh)
        r0["label"] = "CONTROL_committed"
        r0["params"] = [orig["min_dom"], orig["min_conf"], orig["h1_mode"],
                        orig["sess"]]
        results_cache[ctrl_key] = r0
        save_results_cache(results_cache)
        tag = "       "
    rows.append(r0)
    sess_str = f"{orig['sess'][0]}-{orig['sess'][-1]}"
    print(sep)
    print(f"| 25 | {orig['min_dom']:>6} | {orig['min_conf']:>7} | "
          f"{orig['h1_mode']:>13} | {sess_str:>7} | "
          f"{r0['trades']:>5} | {r0['wr']:>4.1f}% | {r0['pf']:>5.2f} | "
          f"{r0['dd']:>4.2f}% | {r0['sharpe']:>5.2f} | {r0['fn_dd']:>4.2f}% | "
          f"{r0['fn_buffer']:>5.2f} | {r0['all_pass']:>1d}/8 | "
          f"{'PASS' if passes_gates(r0) else 'FAIL':>4s} | {tag} |", flush=True)
    print("=" * len(hdr))

    # ── Decision branch ──────────────────────────────────────────────
    passing = [r for r in rows if passes_gates(r)]
    print(f"\n[Sweep] {len(passing)}/25 rows pass all 4 gates "
          f"(trades≥500, PF≥1.50, WR≥60%, FN buffer≥1.30×, 8/8 firms).\n")

    if passing:
        winner = max(passing, key=lambda r: r["pf"])
        print(f"[Sweep] WINNER (highest PF): {winner['label']}")
        print(f"        params={winner['params']}")
        print(f"        trades={winner['trades']}, PF={winner['pf']:.2f}, "
              f"WR={winner['wr']:.1f}%, buffer={winner['fn_buffer']:.2f}×, "
              f"firms={winner['all_pass']}/8")
    else:
        top3 = sorted(rows, key=composite_score, reverse=True)[:3]
        print("[Sweep] No row passes all gates. Top-3 Pareto candidates "
              "(by trades × PF × WR × buffer):")
        for n, r in enumerate(top3, 1):
            print(f"  {n}. {r['label']} → trades={r['trades']} PF={r['pf']:.2f} "
                  f"WR={r['wr']:.1f}% buffer={r['fn_buffer']:.2f}× "
                  f"firms={r['all_pass']}/8")
        print("\n[Sweep] Per plan Step 1.4: pause and surface to user. "
              "Do NOT silently relax gates.")


if __name__ == "__main__":
    main()
