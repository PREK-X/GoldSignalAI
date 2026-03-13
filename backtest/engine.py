"""
GoldSignalAI — backtest/engine.py
===================================
Backtesting engine that simulates the full GoldSignalAI strategy on
historical XAU/USD data.

Approach:
  We walk forward through M15 candles, and at each bar we:
    1. Slice the last N candles (lookback window) for M15 + H1
    2. Run the full analysis pipeline (indicators, S/R, Fib, scoring)
    3. If a BUY/SELL signal fires, open a simulated trade with SL/TP
    4. On subsequent bars, check if SL or TP1/TP2 were hit
    5. Record trade outcome, track PnL, drawdown, etc.

Why custom engine (not backtesting.py library):
  - Our signal pipeline has multi-timeframe confirmation, scoring engine,
    and risk management that don't fit a simple Strategy(bt.Strategy) class
  - We need prop firm compliance simulation (daily loss, drawdown, etc.)
  - We need to track TP1 partial close + TP2 full close
  - Full control over the simulation loop and realistic spread/slippage

Usage:
    from backtest.engine import run_backtest, BacktestConfig
    result = run_backtest(BacktestConfig(spread_pips=3.0))
    print(result.summary())
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd

from config import Config, PROP_FIRM_PROFILES
from data.fetcher import get_candles
from data.processor import process
from analysis.indicators import calculate_all
from analysis.sr_levels import detect_sr_levels
from analysis.fibonacci import calculate_fibonacci
from analysis.candlestick import detect_patterns
from analysis.scoring import score_signal
from signals.risk_manager import (
    calculate_risk,
    price_to_pips,
    pips_to_price,
    RiskParameters,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """
    All tunables for a backtest run.
    Change these to simulate different conditions.
    """
    # Data
    symbol: str = Config.SYMBOL
    lookback_candles: int = Config.LOOKBACK_CANDLES  # analysis window per bar
    min_candles: int = Config.MIN_CANDLES_FOR_SIGNAL

    # Costs & slippage
    spread_pips: float = 3.0        # Gold spread simulation (typical retail)
    slippage_pips: float = 0.5      # Entry slippage simulation

    # Risk
    account_balance: float = Config.CHALLENGE_ACCOUNT_SIZE
    risk_per_trade_pct: float = Config.RISK_PER_TRADE_PCT
    max_open_trades: int = 1        # Only one position at a time

    # Prop firm simulation
    prop_firm_key: str = Config.ACTIVE_PROP_FIRM
    simulate_prop_firm: bool = True

    # Partial close: TP1 closes 50%, TP2 closes remaining 50%
    tp1_close_pct: float = 50.0
    tp2_close_pct: float = 50.0

    # Skip bars between signals to avoid overtrading
    min_bars_between_trades: int = 4  # At least 1 hour between entries (M15)

    # H1 data: how many M15 bars to aggregate for H1 analysis
    h1_resample_bars: int = 4  # 4 × M15 = 1 H1 candle


# ─────────────────────────────────────────────────────────────────────────────
# TRADE RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """A single completed backtest trade."""
    # Entry
    entry_time: datetime
    entry_price: float
    direction: str           # "BUY" or "SELL"
    confidence_pct: float
    lot_size: float

    # Risk levels
    stop_loss: float
    tp1_price: float
    tp2_price: float
    sl_pips: float
    tp1_pips: float
    tp2_pips: float

    # Exit
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""     # "TP1", "TP2", "SL", "END_OF_DATA"

    # PnL
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    is_winner: bool = False

    # TP1 partial close tracking
    tp1_hit: bool = False
    tp1_hit_time: Optional[datetime] = None
    tp1_pnl_pips: float = 0.0
    tp1_pnl_usd: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for CSV/DataFrame export."""
        return {
            "entry_time": self.entry_time.isoformat() if self.entry_time else "",
            "exit_time": self.exit_time.isoformat() if self.exit_time else "",
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "stop_loss": round(self.stop_loss, 2),
            "tp1_price": round(self.tp1_price, 2),
            "tp2_price": round(self.tp2_price, 2),
            "sl_pips": round(self.sl_pips, 1),
            "tp1_pips": round(self.tp1_pips, 1),
            "tp2_pips": round(self.tp2_pips, 1),
            "confidence_pct": round(self.confidence_pct, 1),
            "lot_size": round(self.lot_size, 2),
            "exit_reason": self.exit_reason,
            "pnl_pips": round(self.pnl_pips, 1),
            "pnl_usd": round(self.pnl_usd, 2),
            "is_winner": self.is_winner,
            "tp1_hit": self.tp1_hit,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PropFirmSimulation:
    """Prop firm challenge simulation result."""
    firm_name: str
    passed: bool
    days_to_complete: Optional[int] = None
    final_pnl_pct: float = 0.0
    max_daily_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    closest_daily_breach_pct: float = 0.0
    closest_drawdown_breach_pct: float = 0.0
    days_traded: int = 0
    breach_reason: str = ""  # Why it failed (empty if passed)


@dataclass
class MonthlyBreakdown:
    """Performance for a single month."""
    month: str           # "2024-01"
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest output with all statistics."""
    # Config used
    config: BacktestConfig

    # Raw trades
    trades: list[BacktestTrade] = field(default_factory=list)

    # Headline stats
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate_pct: float = 0.0

    # Pips
    total_pnl_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0

    # USD
    total_pnl_usd: float = 0.0
    final_balance: float = 0.0

    # Ratios
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_rr_achieved: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0

    # Streaks
    best_streak: int = 0
    worst_streak: int = 0
    current_streak: int = 0

    # Best / worst trade
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0

    # Timespan
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

    # Monthly
    monthly: list[MonthlyBreakdown] = field(default_factory=list)

    # Prop firm simulations
    prop_firm_sims: list[PropFirmSimulation] = field(default_factory=list)

    # Equity curve for charting
    equity_curve: list[float] = field(default_factory=list)
    equity_dates: list[datetime] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"{'═' * 50}",
            f" GoldSignalAI Backtest Results",
            f"{'═' * 50}",
            f" Period:           {self.start_date:%Y-%m-%d} → {self.end_date:%Y-%m-%d}" if self.start_date and self.end_date else "",
            f" Trading Days:     {self.trading_days}",
            f" Total Trades:     {self.total_trades}",
            f" Win Rate:         {self.win_rate_pct:.1f}%",
            f" Winners/Losers:   {self.winners} / {self.losers}",
            f"{'─' * 50}",
            f" Total PnL:        {self.total_pnl_pips:+.1f} pips | ${self.total_pnl_usd:+.2f}",
            f" Final Balance:    ${self.final_balance:,.2f}",
            f" Avg Win:          {self.avg_win_pips:+.1f} pips",
            f" Avg Loss:         {self.avg_loss_pips:.1f} pips",
            f" Best Trade:       {self.best_trade_pips:+.1f} pips",
            f" Worst Trade:      {self.worst_trade_pips:.1f} pips",
            f"{'─' * 50}",
            f" Profit Factor:    {self.profit_factor:.2f}",
            f" Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f" Max Drawdown:     {self.max_drawdown_pct:.2f}% (${self.max_drawdown_usd:,.2f})",
            f" Best Streak:      {self.best_streak} wins",
            f" Worst Streak:     {self.worst_streak} losses",
            f"{'═' * 50}",
        ]
        return "\n".join(line for line in lines if line)

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Export all trades as a pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def export_csv(self, path: Optional[str] = None) -> str:
        """Export trade history to CSV. Returns file path."""
        if path is None:
            os.makedirs(Config.REPORTS_DIR, exist_ok=True)
            path = os.path.join(Config.REPORTS_DIR, "backtest_trades.csv")
        df = self.trades_to_dataframe()
        df.to_csv(path, index=False)
        logger.info("Trade history exported to %s", path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# H1 RESAMPLING — build H1 candles from M15 data
# ─────────────────────────────────────────────────────────────────────────────

def _resample_to_h1(m15_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample M15 OHLCV data into H1 candles.

    Uses pandas resample with 'h' frequency on the DatetimeIndex.
    Only returns complete candles (drops the last partial one).
    """
    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    h1 = m15_df.resample("1h").agg(ohlcv).dropna()
    return h1


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-BAR ANALYSIS (stripped-down for speed)
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_bar(
    m15_slice: pd.DataFrame,
    h1_slice: pd.DataFrame,
) -> tuple[Optional[str], float, Optional[RiskParameters], int, int]:
    """
    Run the full analysis pipeline on a single bar's lookback window.

    Returns:
        (direction, confidence_pct, risk_params, bullish_count, bearish_count)
        direction is None if analysis fails.
    """
    try:
        # ── M15 analysis ────────────────────────────────────────────────
        m15_ind = calculate_all(m15_slice)
        m15_sr = detect_sr_levels(m15_slice)
        m15_fib = calculate_fibonacci(m15_slice)
        m15_cand = detect_patterns(m15_slice)
        m15_score = score_signal(m15_ind, m15_sr, m15_fib, m15_cand)

        # ── H1 analysis ────────────────────────────────────────────────
        h1_ind = calculate_all(h1_slice)
        h1_sr = detect_sr_levels(h1_slice)
        h1_fib = calculate_fibonacci(h1_slice)
        h1_cand = detect_patterns(h1_slice)
        h1_score = score_signal(h1_ind, h1_sr, h1_fib, h1_cand)

        # ── Multi-timeframe agreement ──────────────────────────────────
        m15_dir = m15_score.direction
        h1_dir = h1_score.direction

        agree = (m15_dir == h1_dir) and m15_dir in ("BUY", "SELL")

        if not agree:
            return "WAIT", 0.0, None, m15_score.bullish_count, m15_score.bearish_count

        confidence = min(m15_score.confidence_pct, h1_score.confidence_pct)

        if confidence < Config.MIN_CONFIDENCE_PCT:
            return "WAIT", confidence, None, m15_score.bullish_count, m15_score.bearish_count

        # ── Risk calculation ───────────────────────────────────────────
        entry_price = m15_ind.latest_close
        atr_val = m15_ind.atr.value
        risk = calculate_risk(
            entry_price=entry_price,
            direction=m15_dir,
            atr_value=atr_val,
            sr_levels=m15_sr,
        )

        return m15_dir, confidence, risk, m15_score.bullish_count, m15_score.bearish_count

    except Exception as exc:
        logger.debug("Analysis failed at bar: %s", exc)
        return None, 0.0, None, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_spread(entry_price: float, direction: str, spread_pips: float) -> float:
    """
    Adjust entry price for spread.
    BUY: we enter at the ask (higher), so add half spread.
    SELL: we enter at the bid (lower), so subtract half spread.
    """
    half_spread = pips_to_price(spread_pips / 2)
    if direction == "BUY":
        return entry_price + half_spread
    return entry_price - half_spread


def _apply_slippage(entry_price: float, direction: str, slippage_pips: float) -> float:
    """
    Adjust entry price for slippage (always adverse).
    BUY: price slips up. SELL: price slips down.
    """
    slip = pips_to_price(slippage_pips)
    if direction == "BUY":
        return entry_price + slip
    return entry_price - slip


def _check_exit(
    trade: BacktestTrade,
    bar_high: float,
    bar_low: float,
    bar_time: datetime,
) -> bool:
    """
    Check if a trade's SL or TP was hit during this bar.

    For partial close logic:
      - TP1 hit → record partial profit, move SL to breakeven
      - TP2 hit → close remaining position
      - SL hit  → full loss (or breakeven if TP1 already hit)

    Returns True if the trade is fully closed.
    """
    if trade.direction == "BUY":
        # Check SL first (worst case)
        if bar_low <= trade.stop_loss:
            trade.exit_price = trade.stop_loss
            trade.exit_time = bar_time
            if trade.tp1_hit:
                # SL was moved to breakeven after TP1
                trade.exit_reason = "SL_BE"
                trade.pnl_pips = trade.tp1_pnl_pips  # only TP1 profit
                trade.pnl_usd = trade.tp1_pnl_usd
                trade.is_winner = True
            else:
                trade.exit_reason = "SL"
                trade.pnl_pips = -trade.sl_pips
                pip_value = Config.GOLD_PIP_VALUE
                trade.pnl_usd = -trade.sl_pips * pip_value * trade.lot_size
                trade.is_winner = False
            return True

        # Check TP2 (if TP1 already hit)
        if trade.tp1_hit and bar_high >= trade.tp2_price:
            trade.exit_price = trade.tp2_price
            trade.exit_time = bar_time
            trade.exit_reason = "TP2"
            pip_value = Config.GOLD_PIP_VALUE
            # TP2 profit is on the remaining 50% of the position
            tp2_lot = trade.lot_size * 0.5
            tp2_pnl_pips = trade.tp2_pips
            tp2_pnl_usd = tp2_pnl_pips * pip_value * tp2_lot
            trade.pnl_pips = trade.tp1_pnl_pips + tp2_pnl_pips
            trade.pnl_usd = trade.tp1_pnl_usd + tp2_pnl_usd
            trade.is_winner = True
            return True

        # Check TP1 (first target)
        if not trade.tp1_hit and bar_high >= trade.tp1_price:
            trade.tp1_hit = True
            trade.tp1_hit_time = bar_time
            pip_value = Config.GOLD_PIP_VALUE
            tp1_lot = trade.lot_size * 0.5  # close 50%
            trade.tp1_pnl_pips = trade.tp1_pips
            trade.tp1_pnl_usd = trade.tp1_pips * pip_value * tp1_lot
            # Move SL to breakeven for remaining position
            trade.stop_loss = trade.entry_price
            trade.sl_pips = 0.0
            return False  # trade still open with TP2 target

    else:  # SELL
        # Check SL first
        if bar_high >= trade.stop_loss:
            trade.exit_price = trade.stop_loss
            trade.exit_time = bar_time
            if trade.tp1_hit:
                trade.exit_reason = "SL_BE"
                trade.pnl_pips = trade.tp1_pnl_pips
                trade.pnl_usd = trade.tp1_pnl_usd
                trade.is_winner = True
            else:
                trade.exit_reason = "SL"
                trade.pnl_pips = -trade.sl_pips
                pip_value = Config.GOLD_PIP_VALUE
                trade.pnl_usd = -trade.sl_pips * pip_value * trade.lot_size
                trade.is_winner = False
            return True

        # Check TP2
        if trade.tp1_hit and bar_low <= trade.tp2_price:
            trade.exit_price = trade.tp2_price
            trade.exit_time = bar_time
            trade.exit_reason = "TP2"
            pip_value = Config.GOLD_PIP_VALUE
            tp2_lot = trade.lot_size * 0.5
            tp2_pnl_pips = trade.tp2_pips
            tp2_pnl_usd = tp2_pnl_pips * pip_value * tp2_lot
            trade.pnl_pips = trade.tp1_pnl_pips + tp2_pnl_pips
            trade.pnl_usd = trade.tp1_pnl_usd + tp2_pnl_usd
            trade.is_winner = True
            return True

        # Check TP1
        if not trade.tp1_hit and bar_low <= trade.tp1_price:
            trade.tp1_hit = True
            trade.tp1_hit_time = bar_time
            pip_value = Config.GOLD_PIP_VALUE
            tp1_lot = trade.lot_size * 0.5
            trade.tp1_pnl_pips = trade.tp1_pips
            trade.tp1_pnl_usd = trade.tp1_pips * pip_value * tp1_lot
            trade.stop_loss = trade.entry_price
            trade.sl_pips = 0.0
            return False

    return False  # trade still open


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_statistics(
    trades: list[BacktestTrade],
    cfg: BacktestConfig,
) -> BacktestResult:
    """
    Compute all performance statistics from the list of completed trades.
    """
    result = BacktestResult(config=cfg, trades=trades)

    if not trades:
        result.final_balance = cfg.account_balance
        return result

    result.total_trades = len(trades)
    result.winners = sum(1 for t in trades if t.is_winner)
    result.losers = result.total_trades - result.winners
    result.win_rate_pct = (result.winners / result.total_trades * 100) if result.total_trades > 0 else 0.0

    # ── PnL ────────────────────────────────────────────────────────────
    pnl_list = [t.pnl_pips for t in trades]
    usd_list = [t.pnl_usd for t in trades]
    result.total_pnl_pips = sum(pnl_list)
    result.total_pnl_usd = sum(usd_list)
    result.final_balance = cfg.account_balance + result.total_pnl_usd

    wins = [t.pnl_pips for t in trades if t.is_winner]
    losses = [t.pnl_pips for t in trades if not t.is_winner]
    result.avg_win_pips = np.mean(wins) if wins else 0.0
    result.avg_loss_pips = np.mean(losses) if losses else 0.0

    result.best_trade_pips = max(pnl_list) if pnl_list else 0.0
    result.worst_trade_pips = min(pnl_list) if pnl_list else 0.0

    # ── Profit factor ──────────────────────────────────────────────────
    gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
    result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # ── Sharpe ratio (annualized, using daily returns) ──────────────────
    if len(usd_list) > 1:
        returns = np.array(usd_list) / cfg.account_balance
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret > 0:
            # Annualize: assume ~252 trading days, ~2 trades per day at most
            result.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252)
        else:
            result.sharpe_ratio = 0.0
    else:
        result.sharpe_ratio = 0.0

    # ── Average R/R achieved ───────────────────────────────────────────
    rr_list = []
    for t in trades:
        if t.sl_pips > 0 and t.pnl_pips != 0:
            rr_list.append(t.pnl_pips / t.sl_pips if t.sl_pips > 0 else 0)
    result.avg_rr_achieved = np.mean(rr_list) if rr_list else 0.0

    # ── Equity curve & drawdown ────────────────────────────────────────
    equity = cfg.account_balance
    peak = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    curve = [equity]
    dates = [trades[0].entry_time] if trades else []

    for t in trades:
        equity += t.pnl_usd
        curve.append(equity)
        dates.append(t.exit_time or t.entry_time)
        if equity > peak:
            peak = equity
        dd_usd = peak - equity
        dd_pct = (dd_usd / peak * 100) if peak > 0 else 0
        if dd_usd > max_dd_usd:
            max_dd_usd = dd_usd
            max_dd_pct = dd_pct

    result.equity_curve = curve
    result.equity_dates = dates
    result.max_drawdown_usd = max_dd_usd
    result.max_drawdown_pct = max_dd_pct

    # ── Streaks ────────────────────────────────────────────────────────
    best_win_streak = 0
    worst_loss_streak = 0
    current_streak = 0

    for t in trades:
        if t.is_winner:
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
            best_win_streak = max(best_win_streak, current_streak)
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1
            worst_loss_streak = max(worst_loss_streak, abs(current_streak))

    result.best_streak = best_win_streak
    result.worst_streak = worst_loss_streak
    result.current_streak = current_streak

    # ── Time span ──────────────────────────────────────────────────────
    result.start_date = trades[0].entry_time
    result.end_date = trades[-1].exit_time or trades[-1].entry_time
    unique_days = set()
    for t in trades:
        if t.entry_time:
            unique_days.add(t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time)
    result.trading_days = len(unique_days)

    # ── Monthly breakdown ──────────────────────────────────────────────
    monthly_map: dict[str, MonthlyBreakdown] = {}
    for t in trades:
        key = t.entry_time.strftime("%Y-%m") if t.entry_time else "unknown"
        if key not in monthly_map:
            monthly_map[key] = MonthlyBreakdown(month=key)
        m = monthly_map[key]
        m.trades += 1
        if t.is_winner:
            m.wins += 1
        else:
            m.losses += 1
        m.pnl_pips += t.pnl_pips
        m.pnl_usd += t.pnl_usd

    for m in monthly_map.values():
        m.win_rate = (m.wins / m.trades * 100) if m.trades > 0 else 0.0

    result.monthly = sorted(monthly_map.values(), key=lambda x: x.month)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROP FIRM CHALLENGE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_prop_firm(
    trades: list[BacktestTrade],
    firm_key: str,
    account_balance: float,
) -> PropFirmSimulation:
    """
    Simulate whether the backtest trades would have passed a prop firm challenge.

    Checks:
      - Daily loss limit (any single day)
      - Maximum total drawdown (from peak)
      - Profit target reached
      - Minimum trading days met
    """
    profile = PROP_FIRM_PROFILES.get(firm_key)
    if profile is None:
        return PropFirmSimulation(
            firm_name=firm_key, passed=False,
            breach_reason=f"Unknown firm key: {firm_key}"
        )

    equity = account_balance
    peak = equity
    max_dd_pct = 0.0
    max_daily_loss_pct = 0.0
    passed = False
    days_to_complete = None
    breach_reason = ""

    # Group trades by date
    daily_pnl: dict[str, float] = {}
    unique_days = set()
    target_equity = account_balance * (1 + profile.profit_target / 100)

    for trade in trades:
        day_key = trade.entry_time.strftime("%Y-%m-%d") if trade.entry_time else "unknown"
        unique_days.add(day_key)

        if day_key not in daily_pnl:
            daily_pnl[day_key] = 0.0
        daily_pnl[day_key] += trade.pnl_usd
        equity += trade.pnl_usd

        # Daily loss check
        daily_loss_pct = abs(min(0, daily_pnl[day_key])) / account_balance * 100
        if daily_loss_pct > max_daily_loss_pct:
            max_daily_loss_pct = daily_loss_pct

        if daily_loss_pct >= profile.daily_loss_limit:
            breach_reason = (
                f"Daily loss limit breached on {day_key}: "
                f"{daily_loss_pct:.2f}% >= {profile.daily_loss_limit:.1f}%"
            )
            break

        # Total drawdown check
        if equity > peak:
            peak = equity
        dd_pct = (peak - equity) / account_balance * 100
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        if dd_pct >= profile.max_total_drawdown:
            breach_reason = (
                f"Max drawdown breached on {day_key}: "
                f"{dd_pct:.2f}% >= {profile.max_total_drawdown:.1f}%"
            )
            break

        # Profit target check
        if equity >= target_equity and not passed:
            passed = True
            days_to_complete = len(unique_days)

    # Final checks
    if not breach_reason and not passed:
        final_pnl = (equity - account_balance) / account_balance * 100
        breach_reason = f"Profit target not reached: {final_pnl:.2f}% < {profile.profit_target:.1f}%"

    if passed and profile.min_trading_days > 0 and len(unique_days) < profile.min_trading_days:
        breach_reason = (
            f"Minimum trading days not met: {len(unique_days)} < {profile.min_trading_days}"
        )
        passed = False

    return PropFirmSimulation(
        firm_name=profile.name,
        passed=passed and not breach_reason,
        days_to_complete=days_to_complete,
        final_pnl_pct=(equity - account_balance) / account_balance * 100,
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_dd_pct,
        closest_daily_breach_pct=profile.daily_loss_limit - max_daily_loss_pct if max_daily_loss_pct < profile.daily_loss_limit else 0,
        closest_drawdown_breach_pct=profile.max_total_drawdown - max_dd_pct if max_dd_pct < profile.max_total_drawdown else 0,
        days_traded=len(unique_days),
        breach_reason=breach_reason if not passed or breach_reason else "",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_historical_data(
    cfg: BacktestConfig,
) -> Optional[pd.DataFrame]:
    """
    Fetch 2 years of M15 data for backtesting.

    Uses yfinance (get_candles with fallback). The data is fetched in
    chunks if necessary to respect yfinance's 60-day limit for M15.
    """
    logger.info("Fetching historical M15 data for backtest...")

    # yfinance M15 limit is 60 days per request — we need ~730 days
    # Fetch in chunks of 59 days
    chunk_days = 59
    total_days = Config.HISTORICAL_YEARS * 365
    all_chunks = []

    end_dt = datetime.now(timezone.utc)
    remaining = total_days

    while remaining > 0:
        fetch_days = min(chunk_days, remaining)
        start_dt = end_dt - timedelta(days=fetch_days)

        try:
            raw = get_candles(
                timeframe="M15",
                n_candles=fetch_days * 24 * 4,  # approximate M15 bars
                symbol=cfg.symbol,
            )
            if raw is not None and not raw.empty:
                # Filter to the date range we want
                mask = (raw.index >= start_dt) & (raw.index <= end_dt)
                chunk = raw[mask]
                if not chunk.empty:
                    all_chunks.append(chunk)
        except Exception as exc:
            logger.warning("Chunk fetch failed for %s → %s: %s", start_dt, end_dt, exc)

        end_dt = start_dt
        remaining -= fetch_days

    if not all_chunks:
        logger.error("No historical data retrieved.")
        return None

    # Combine and deduplicate
    combined = pd.concat(all_chunks)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    logger.info(
        "Historical data: %d bars from %s to %s",
        len(combined), combined.index[0], combined.index[-1],
    )
    return combined


def run_backtest(
    cfg: Optional[BacktestConfig] = None,
    m15_data: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Run the full backtest simulation.

    Args:
        cfg:      Backtest configuration. Uses defaults if None.
        m15_data: Pre-loaded M15 DataFrame. If None, fetches from yfinance.

    Returns:
        BacktestResult with all statistics, trades, and prop firm simulations.

    Usage:
        # Simple — use defaults
        result = run_backtest()

        # With custom config
        result = run_backtest(BacktestConfig(spread_pips=2.0))

        # With pre-loaded data
        result = run_backtest(m15_data=my_dataframe)
    """
    if cfg is None:
        cfg = BacktestConfig()

    logger.info("Starting backtest with config: %s", cfg)

    # ── Step 1: Get data ───────────────────────────────────────────────
    if m15_data is None:
        m15_data = _fetch_historical_data(cfg)
    if m15_data is None or m15_data.empty:
        logger.error("No data available for backtest.")
        return BacktestResult(config=cfg, final_balance=cfg.account_balance)

    # Process the raw data
    m15_processed = process(m15_data, timeframe="M15", label="BT_M15")
    if m15_processed is None or m15_processed.empty:
        logger.error("Data processing failed.")
        return BacktestResult(config=cfg, final_balance=cfg.account_balance)

    # Build H1 data from M15
    h1_data = _resample_to_h1(m15_processed)

    logger.info(
        "Backtest data ready: M15=%d bars, H1=%d bars",
        len(m15_processed), len(h1_data),
    )

    # ── Step 2: Walk-forward simulation ────────────────────────────────
    trades: list[BacktestTrade] = []
    open_trade: Optional[BacktestTrade] = None
    bars_since_last_trade = cfg.min_bars_between_trades  # allow first trade immediately
    lookback = cfg.lookback_candles
    min_bars = cfg.min_candles

    total_bars = len(m15_processed)
    start_idx = max(lookback, min_bars)

    logger.info(
        "Simulating %d bars (from index %d to %d)...",
        total_bars - start_idx, start_idx, total_bars,
    )

    progress_step = max(1, (total_bars - start_idx) // 20)  # log every 5%

    for i in range(start_idx, total_bars):
        bar = m15_processed.iloc[i]
        bar_time = m15_processed.index[i]
        bar_high = bar["high"]
        bar_low = bar["low"]

        # Progress logging
        if (i - start_idx) % progress_step == 0:
            pct = (i - start_idx) / (total_bars - start_idx) * 100
            logger.info("Backtest progress: %.0f%% (%d/%d bars)", pct, i - start_idx, total_bars - start_idx)

        # ── Check open trade exit ──────────────────────────────────────
        if open_trade is not None:
            closed = _check_exit(open_trade, bar_high, bar_low, bar_time)
            if closed:
                trades.append(open_trade)
                open_trade = None
                bars_since_last_trade = 0
                continue  # don't open a new trade on the same bar

        bars_since_last_trade += 1

        # ── Skip if we already have an open trade ──────────────────────
        if open_trade is not None:
            continue

        # ── Skip if too soon after last trade ──────────────────────────
        if bars_since_last_trade < cfg.min_bars_between_trades:
            continue

        # ── Run analysis on lookback window ────────────────────────────
        m15_slice = m15_processed.iloc[max(0, i - lookback):i + 1]

        # Get corresponding H1 data up to this point
        h1_mask = h1_data.index <= bar_time
        h1_slice = h1_data[h1_mask].tail(lookback // 4)  # roughly lookback/4 H1 bars

        if len(m15_slice) < min_bars or len(h1_slice) < 50:
            continue

        # ── Analyse ────────────────────────────────────────────────────
        direction, confidence, risk, bull_count, bear_count = _analyse_bar(m15_slice, h1_slice)

        if direction is None or direction == "WAIT" or risk is None:
            continue

        # ── Open trade ─────────────────────────────────────────────────
        entry = risk.entry_price
        entry = _apply_spread(entry, direction, cfg.spread_pips)
        entry = _apply_slippage(entry, direction, cfg.slippage_pips)

        # Recalculate lot size for backtest account balance
        current_balance = cfg.account_balance + sum(t.pnl_usd for t in trades)
        risk_usd = current_balance * (cfg.risk_per_trade_pct / 100)
        pip_value = Config.GOLD_PIP_VALUE
        lot_size = risk_usd / (risk.sl_pips * pip_value) if risk.sl_pips > 0 else 0.01
        lot_size = round(max(0.01, lot_size), 2)

        open_trade = BacktestTrade(
            entry_time=bar_time,
            entry_price=entry,
            direction=direction,
            confidence_pct=confidence,
            lot_size=lot_size,
            stop_loss=risk.stop_loss,
            tp1_price=risk.tp1_price,
            tp2_price=risk.tp2_price,
            sl_pips=risk.sl_pips,
            tp1_pips=risk.tp1_pips,
            tp2_pips=risk.tp2_pips,
        )
        bars_since_last_trade = 0

    # ── Close any remaining open trade at last bar's close ─────────────
    if open_trade is not None:
        last_bar = m15_processed.iloc[-1]
        last_time = m15_processed.index[-1]
        open_trade.exit_price = last_bar["close"]
        open_trade.exit_time = last_time
        open_trade.exit_reason = "END_OF_DATA"
        if open_trade.direction == "BUY":
            open_trade.pnl_pips = price_to_pips(last_bar["close"] - open_trade.entry_price)
        else:
            open_trade.pnl_pips = price_to_pips(open_trade.entry_price - last_bar["close"])
        pip_value = Config.GOLD_PIP_VALUE
        open_trade.pnl_usd = open_trade.pnl_pips * pip_value * open_trade.lot_size
        open_trade.is_winner = open_trade.pnl_pips > 0
        trades.append(open_trade)

    logger.info("Backtest complete: %d trades generated.", len(trades))

    # ── Step 3: Compute statistics ─────────────────────────────────────
    result = _compute_statistics(trades, cfg)

    # ── Step 4: Prop firm simulations ──────────────────────────────────
    if cfg.simulate_prop_firm:
        for firm_key in PROP_FIRM_PROFILES:
            sim = _simulate_prop_firm(trades, firm_key, cfg.account_balance)
            result.prop_firm_sims.append(sim)
            status = "PASSED" if sim.passed else "FAILED"
            logger.info(
                "Prop firm [%s]: %s — PnL=%.2f%%, MaxDD=%.2f%%, Days=%d",
                sim.firm_name, status, sim.final_pnl_pct,
                sim.max_drawdown_pct, sim.days_traded,
            )

    logger.info("\n%s", result.summary())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("Running GoldSignalAI backtest...\n")
    result = run_backtest()
    print(result.summary())

    if result.trades:
        csv_path = result.export_csv()
        print(f"\nTrade history exported to: {csv_path}")

    if result.prop_firm_sims:
        print("\n" + "═" * 50)
        print(" Prop Firm Challenge Simulations")
        print("═" * 50)
        for sim in result.prop_firm_sims:
            status = "✅ PASSED" if sim.passed else "❌ FAILED"
            print(f"  {sim.firm_name:25s} {status}")
            if sim.passed:
                print(f"    Days to complete: {sim.days_to_complete}")
            else:
                print(f"    Reason: {sim.breach_reason}")
            print(f"    PnL: {sim.final_pnl_pct:+.2f}% | Max DD: {sim.max_drawdown_pct:.2f}% | Max Daily Loss: {sim.max_daily_loss_pct:.2f}%")
