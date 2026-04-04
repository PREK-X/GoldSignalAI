#!/usr/bin/env python3
"""
Stage 15 Phase 2: Diagnose DD regression.

Runs the full backtest and dumps per-trade diagnostics,
focusing on the Nov 2025 - Jan 2026 losing period.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_backtest, BacktestConfig
import pandas as pd

def main():
    print("=" * 60)
    print(" Stage 15 Phase 2: DD Regression Diagnostic")
    print("=" * 60)

    cfg = BacktestConfig()
    result = run_backtest(cfg)

    # Print headline stats
    print(result.summary())

    # Convert trades to DataFrame
    df = result.trades_to_dataframe()
    if df.empty:
        print("No trades generated!")
        return

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["month"] = df["entry_time"].dt.strftime("%Y-%m")

    # ── Monthly breakdown ──
    print("\n" + "=" * 60)
    print(" Monthly Breakdown")
    print("=" * 60)
    monthly = df.groupby("month").agg(
        trades=("pnl_usd", "count"),
        wins=("is_winner", "sum"),
        pnl_usd=("pnl_usd", "sum"),
    )
    monthly["win_rate"] = (monthly["wins"] / monthly["trades"] * 100).round(1)
    for month, row in monthly.iterrows():
        flag = " <<<" if row["pnl_usd"] < -200 else ""
        print(f"  {month}: {int(row['trades']):3d} trades | WR {row['win_rate']:5.1f}% | PnL ${row['pnl_usd']:+8.2f}{flag}")

    # ── Focus on bad months: Nov 2025, Dec 2025, Jan 2026 ──
    bad_months = ["2025-11", "2025-12", "2026-01"]
    bad_trades = df[df["month"].isin(bad_months)]

    print("\n" + "=" * 60)
    print(f" LOSING PERIOD DETAIL (Nov 2025 - Jan 2026): {len(bad_trades)} trades")
    print("=" * 60)

    if not bad_trades.empty:
        losers_only = bad_trades[~bad_trades["is_winner"]]
        print(f"\n  Total trades: {len(bad_trades)} | Losers: {len(losers_only)}")

        # Q1: What % of losses occurred in RANGING state?
        ranging_losses = losers_only[losers_only["regime"] == "RANGING"]
        trending_losses = losers_only[losers_only["regime"] == "TRENDING"]
        print(f"\n  Q1: Regime distribution of losses:")
        print(f"      RANGING losses:  {len(ranging_losses)} / {len(losers_only)} = {len(ranging_losses)/max(1,len(losers_only))*100:.1f}%")
        print(f"      TRENDING losses: {len(trending_losses)} / {len(losers_only)} = {len(trending_losses)/max(1,len(losers_only))*100:.1f}%")

        # Q3: SL hits vs time exits vs other
        print(f"\n  Q3: Exit reasons for losers:")
        for reason, count in losers_only["exit_reason"].value_counts().items():
            print(f"      {reason}: {count}")

        # Detail per trade
        print(f"\n  Per-trade detail (losers in bad months):")
        print(f"  {'Date':20s} {'Dir':5s} {'Conf':5s} {'Regime':10s} {'Exit':8s} {'PnL':>10s} {'Risk%':>6s}")
        print(f"  {'-'*70}")
        for _, t in bad_trades.iterrows():
            marker = " L" if not t["is_winner"] else " W"
            print(f"  {str(t['entry_time'])[:19]:20s} {t['direction']:5s} {t['confidence_pct']:5.1f} {t['regime']:10s} {t['exit_reason']:8s} ${t['pnl_usd']:+9.2f} {t['risk_pct']:5.2f}%{marker}")

    # ── RANGING trades overall ──
    print("\n" + "=" * 60)
    print(" RANGING vs TRENDING Performance (all trades)")
    print("=" * 60)
    for regime in ["TRENDING", "RANGING"]:
        regime_df = df[df["regime"] == regime]
        if regime_df.empty:
            continue
        wr = regime_df["is_winner"].mean() * 100
        pnl = regime_df["pnl_usd"].sum()
        avg_pnl = regime_df["pnl_usd"].mean()
        n = len(regime_df)
        n_losers = len(regime_df[~regime_df["is_winner"]])
        avg_loss = regime_df[~regime_df["is_winner"]]["pnl_usd"].mean() if n_losers > 0 else 0
        print(f"  {regime}: {n} trades | WR {wr:.1f}% | PnL ${pnl:+.2f} | Avg ${avg_pnl:+.2f} | Avg Loss ${avg_loss:.2f}")

    # ── Sortino, Sharpe ──
    print(f"\n  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  Max DD: {result.max_drawdown_pct:.2f}%")
    print(f"  PF: {result.profit_factor:.2f}")
    print(f"  WR: {result.win_rate_pct:.1f}%")
    print(f"  Trades: {result.total_trades}")
    print(f"  PnL: ${result.total_pnl_usd:+.2f}")

    # ── Prop firm sims ──
    print("\n" + "=" * 60)
    print(" Prop Firm Simulations")
    print("=" * 60)
    for sim in result.prop_firm_sims:
        status = "PASS" if sim.passed else "FAIL"
        print(f"  {sim.firm_name}: {status} | DD {sim.max_drawdown_pct:.2f}% | Daily {sim.max_daily_loss_pct:.2f}% | Days {sim.days_traded}")
        if sim.breach_reason:
            print(f"    Breach: {sim.breach_reason}")

if __name__ == "__main__":
    main()
