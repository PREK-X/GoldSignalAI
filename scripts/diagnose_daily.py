#!/usr/bin/env python3
"""Diagnose 2025-10-27 daily loss breach."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_backtest, BacktestConfig
import pandas as pd

cfg = BacktestConfig()
result = run_backtest(cfg)
df = result.trades_to_dataframe()
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["date"] = df["entry_time"].dt.strftime("%Y-%m-%d")

# All trades on 2025-10-27
day_trades = df[df["date"] == "2025-10-27"]
print(f"\n{'='*60}")
print(f" Trades on 2025-10-27 ({len(day_trades)} trades)")
print(f"{'='*60}")

cum_pnl = 0.0
for _, t in day_trades.iterrows():
    cum_pnl += t["pnl_usd"]
    daily_pct = abs(min(0, cum_pnl)) / 10000 * 100
    print(f"  {str(t['entry_time'])[:19]} {t['direction']:5s} conf={t['confidence_pct']:.1f} "
          f"regime={t['regime']:10s} exit={t['exit_reason']:12s} "
          f"PnL=${t['pnl_usd']:+8.2f} CumPnL=${cum_pnl:+8.2f} DailyLoss%={daily_pct:.2f}%")

print(f"\n  Daily total: ${cum_pnl:+.2f} = {abs(min(0,cum_pnl))/100:.2f}% daily loss")

# Also check surrounding days
print(f"\n{'='*60}")
print(f" Context: trades 2025-10-25 to 2025-10-30")
print(f"{'='*60}")
context = df[(df["date"] >= "2025-10-25") & (df["date"] <= "2025-10-30")]
for d, grp in context.groupby("date"):
    day_pnl = grp["pnl_usd"].sum()
    print(f"  {d}: {len(grp)} trades | PnL ${day_pnl:+.2f} | DailyLoss {abs(min(0,day_pnl))/100:.2f}%")

# Check what daily loss was BEFORE the last losing trade
if len(day_trades) > 1:
    pre_last = day_trades.iloc[:-1]["pnl_usd"].sum()
    pre_pct = abs(min(0, pre_last)) / 100
    print(f"\n  Before last trade: cumPnL=${pre_last:+.2f} = {pre_pct:.2f}% daily loss")
    print(f"  Would 2.8% ceiling have stopped it? {'YES' if pre_pct >= 2.8 else 'NO'}")
    print(f"  Would 2.5% ceiling have stopped it? {'YES' if pre_pct >= 2.5 else 'NO'}")
