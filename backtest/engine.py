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
from data.fetcher import get_candles, get_market_data
from data.processor import process
from analysis.indicators import calculate_all, PrecomputedIndicators
from analysis.sr_levels import detect_sr_levels
from analysis.fibonacci import calculate_fibonacci
from analysis.candlestick import detect_patterns
from analysis.scoring import score_signal
from signals.risk_manager import (
    calculate_risk,
    price_to_pips,
    pips_to_price,
    RiskParameters,
    calculate_half_kelly_risk_pct,
    should_friday_close,
    should_time_exit,
    update_trailing_stop,
)
from infrastructure.circuit_breaker import CircuitBreaker, NORMAL, CAUTION, RESTRICTED, HALTED

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

    # Timeframe selection
    # "M15" = 60-day backtest (yfinance hard limit for 15m data)
    # "H1"  = up to 2-year backtest (yfinance allows 730 days of 1h data)
    data_timeframe: str = "M15"

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

    # Confidence threshold — matches live trading at 65%.
    min_confidence_pct: float = Config.MIN_CONFIDENCE_PCT

    # ML integration: train models on available data and use for filtering.
    use_ml: bool = True

    # Stage 5: LGBM direction classifier filter
    use_lgbm: bool = True

    # Limit order entry at S/R levels
    use_limit_orders: bool = False
    limit_max_distance_pips: float = 50.0  # max distance from current price to S/R
    limit_sl_pips: float = 60.0            # tighter SL when entering at S/R
    limit_expiry_bars: int = 8             # cancel pending order after N bars (2 hours)


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

    # Regime state at entry
    regime_state: int = -1       # 0=TRENDING, 1=RANGING, 2=CRISIS, -1=unknown
    regime_label: str = ""

    # Stage 6: risk management
    entry_bar_idx: int = 0       # M15 bar index at entry (for 48-bar time exit)
    initial_sl_pips: float = 0.0 # original SL before trailing (1R reference)
    trailing_stop: Optional[float] = None  # active trailing stop price
    risk_pct_used: float = 0.0   # actual risk % after Half-Kelly + circuit breaker

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
            "regime": self.regime_label,
            "risk_pct": round(self.risk_pct_used, 2),
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

    # Regime distribution (% of H1 bars in each state)
    regime_distribution: dict = field(default_factory=dict)
    regime_filtered: int = 0   # signals suppressed by CRISIS regime

    # Stage 6: circuit breaker & risk stats
    cb_state_counts: dict = field(default_factory=dict)   # {NORMAL: N, CAUTION: N, ...}
    cb_days_halted: int = 0
    cb_total_dd_overrides: int = 0
    avg_risk_pct: float = 0.0
    exit_friday_close: int = 0
    exit_time_48bar: int = 0
    exit_trailing_stop: int = 0

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
        ]
        if self.cb_state_counts:
            lines.append(f"{'─' * 50}")
            lines.append(f" Circuit Breaker:")
            for state, count in self.cb_state_counts.items():
                lines.append(f"   {state:12s} {count:5d} signals")
            lines.append(f"   Days halted:        {self.cb_days_halted}")
            lines.append(f"   Total DD overrides: {self.cb_total_dd_overrides}")
            lines.append(f"   Avg risk %:         {self.avg_risk_pct:.2f}%")
        if self.exit_friday_close or self.exit_time_48bar or self.exit_trailing_stop:
            lines.append(f"{'─' * 50}")
            lines.append(f" Exit Reasons (new):")
            lines.append(f"   Friday close:    {self.exit_friday_close}")
            lines.append(f"   48-bar timeout:  {self.exit_time_48bar}")
            lines.append(f"   Trailing stop:   {self.exit_trailing_stop}")
        if self.regime_distribution:
            lines.append(f"{'─' * 50}")
            lines.append(f" Regime Distribution (H1 bars):")
            for label, pct in self.regime_distribution.items():
                lines.append(f"   {label:12s} {pct:5.1f}%")
            if self.regime_filtered > 0:
                lines.append(f"   Signals filtered by CRISIS: {self.regime_filtered}")
        lines.append(f"{'═' * 50}")
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

class _AnalysisResult:
    """Bundle of analysis outputs for a single bar."""
    __slots__ = ("direction", "confidence", "risk", "bull_count", "bear_count", "sr_levels", "current_price")

    def __init__(self, direction, confidence, risk, bull_count, bear_count, sr_levels=None, current_price=0.0):
        self.direction = direction
        self.confidence = confidence
        self.risk = risk
        self.bull_count = bull_count
        self.bear_count = bear_count
        self.sr_levels = sr_levels
        self.current_price = current_price


def _analyse_bar(
    m15_slice: pd.DataFrame,
    h1_slice: pd.DataFrame,
    bar_time: Optional[datetime] = None,
    m15_precomp: Optional[PrecomputedIndicators] = None,
    m15_bar_idx: Optional[int] = None,
    h1_precomp: Optional[PrecomputedIndicators] = None,
    h1_bar_idx: Optional[int] = None,
) -> _AnalysisResult:
    """
    Run the full analysis pipeline on a single bar's lookback window.

    If precomputed indicator objects are provided, uses O(1) array lookups
    instead of re-running all rolling computations (10-20x faster per bar).

    Returns _AnalysisResult with direction, confidence, risk, counts, and S/R levels.
    """
    try:
        # ── M15 analysis ────────────────────────────────────────────────
        if m15_precomp is not None and m15_bar_idx is not None:
            m15_ind = m15_precomp.at(m15_bar_idx)
        else:
            m15_ind = calculate_all(m15_slice)
        m15_sr = detect_sr_levels(m15_slice)
        m15_fib = calculate_fibonacci(m15_slice)
        m15_cand = detect_patterns(m15_slice)
        # Pass bar_time for session gate (applied on M15 score only — it's the signal TF)
        m15_score = score_signal(m15_ind, m15_sr, m15_fib, m15_cand, bar_time=bar_time)

        # ── H1 analysis ────────────────────────────────────────────────
        if h1_precomp is not None and h1_bar_idx is not None:
            h1_ind = h1_precomp.at(h1_bar_idx)
        else:
            h1_ind = calculate_all(h1_slice)
        h1_sr = detect_sr_levels(h1_slice)
        h1_fib = calculate_fibonacci(h1_slice)
        h1_cand = detect_patterns(h1_slice)
        h1_score = score_signal(h1_ind, h1_sr, h1_fib, h1_cand)  # no session gate on H1

        # ── Multi-timeframe agreement ──────────────────────────────────
        m15_dir = m15_score.direction
        h1_dir = h1_score.direction

        agree = (m15_dir == h1_dir) and m15_dir in ("BUY", "SELL")

        if not agree:
            return _AnalysisResult("WAIT", 0.0, None, m15_score.bullish_count, m15_score.bearish_count)

        confidence = min(m15_score.confidence_pct, h1_score.confidence_pct)

        # ── Risk calculation ───────────────────────────────────────────
        entry_price = m15_ind.latest_close
        atr_val = m15_ind.atr.value
        risk = calculate_risk(
            entry_price=entry_price,
            direction=m15_dir,
            atr_value=atr_val,
            sr_levels=m15_sr,
        )

        return _AnalysisResult(
            m15_dir, confidence, risk,
            m15_score.bullish_count, m15_score.bearish_count,
            sr_levels=m15_sr, current_price=entry_price,
        )

    except Exception as exc:
        logger.debug("Analysis failed at bar: %s", exc)
        return _AnalysisResult(None, 0.0, None, 0, 0)


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


def _close_trade_at_market(
    trade: BacktestTrade,
    exit_price: float,
    exit_time: datetime,
    reason: str,
) -> None:
    """Close a trade at a given price with a custom exit reason."""
    trade.exit_price = exit_price
    trade.exit_time = exit_time
    trade.exit_reason = reason
    pip_value = Config.GOLD_PIP_VALUE
    if trade.direction == "BUY":
        trade.pnl_pips = price_to_pips(exit_price - trade.entry_price)
    else:
        trade.pnl_pips = price_to_pips(trade.entry_price - exit_price)
    # If TP1 was already hit, add that partial profit
    if trade.tp1_hit:
        total_pnl_pips = trade.tp1_pnl_pips + trade.pnl_pips
        remaining_lot = trade.lot_size * 0.5
        trade.pnl_usd = trade.tp1_pnl_usd + trade.pnl_pips * pip_value * remaining_lot
        trade.pnl_pips = total_pnl_pips
    else:
        trade.pnl_usd = trade.pnl_pips * pip_value * trade.lot_size
    trade.is_winner = trade.pnl_pips > 0


# ─────────────────────────────────────────────────────────────────────────────
# LIMIT ORDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class _PendingOrder:
    """A pending limit order waiting to be filled."""
    __slots__ = (
        "direction", "limit_price", "stop_loss", "tp1_price", "tp2_price",
        "sl_pips", "tp1_pips", "tp2_pips", "confidence_pct", "lot_size",
        "created_bar", "created_time", "expiry_bars",
    )

    def __init__(
        self, direction, limit_price, stop_loss, tp1_price, tp2_price,
        sl_pips, tp1_pips, tp2_pips, confidence_pct, lot_size,
        created_bar, created_time, expiry_bars,
    ):
        self.direction = direction
        self.limit_price = limit_price
        self.stop_loss = stop_loss
        self.tp1_price = tp1_price
        self.tp2_price = tp2_price
        self.sl_pips = sl_pips
        self.tp1_pips = tp1_pips
        self.tp2_pips = tp2_pips
        self.confidence_pct = confidence_pct
        self.lot_size = lot_size
        self.created_bar = created_bar
        self.created_time = created_time
        self.expiry_bars = expiry_bars

    def is_expired(self, current_bar: int) -> bool:
        return (current_bar - self.created_bar) >= self.expiry_bars

    def is_filled(self, bar_high: float, bar_low: float) -> bool:
        """Check if price reached our limit during this bar."""
        if self.direction == "BUY":
            return bar_low <= self.limit_price
        else:  # SELL
            return bar_high >= self.limit_price


def _find_limit_entry(
    direction: str,
    current_price: float,
    sr_levels,
    max_distance_pips: float,
    sl_pips: float,
) -> Optional[_PendingOrder]:
    """
    Find the nearest S/R level for a limit order entry.

    For BUY: find nearest support below current price (buy the dip).
    For SELL: find nearest resistance above current price (sell the rally).

    Checks strong zones first, then falls back to ALL zones if no strong
    zone is within range.

    Returns a partially filled _PendingOrder (without lot_size/bar info),
    or None if no suitable S/R level within max_distance_pips.
    """
    if sr_levels is None:
        return None

    if direction == "BUY":
        # Try strong zone first, then any zone
        zone = sr_levels.nearest_support
        if zone is None or zone.price >= current_price:
            # Fallback: find nearest support zone (any strength) below price
            candidates = [
                z for z in sr_levels.zones
                if z.zone_type == "support" and z.price < current_price
            ]
            if not candidates:
                return None
            zone = min(candidates, key=lambda z: current_price - z.price)

        dist_pips = price_to_pips(current_price - zone.price)
        if dist_pips > max_distance_pips or dist_pips < 1.0:
            return None
        limit_price = zone.price
        stop_loss = limit_price - pips_to_price(sl_pips)
        tp1_price = limit_price + pips_to_price(sl_pips)
        tp2_price = limit_price + pips_to_price(sl_pips * 2)

    elif direction == "SELL":
        zone = sr_levels.nearest_resistance
        if zone is None or zone.price <= current_price:
            candidates = [
                z for z in sr_levels.zones
                if z.zone_type == "resistance" and z.price > current_price
            ]
            if not candidates:
                return None
            zone = min(candidates, key=lambda z: z.price - current_price)

        dist_pips = price_to_pips(zone.price - current_price)
        if dist_pips > max_distance_pips or dist_pips < 1.0:
            return None
        limit_price = zone.price
        stop_loss = limit_price + pips_to_price(sl_pips)
        tp1_price = limit_price - pips_to_price(sl_pips)
        tp2_price = limit_price - pips_to_price(sl_pips * 2)

    else:
        return None

    return _PendingOrder(
        direction=direction,
        limit_price=limit_price,
        stop_loss=stop_loss,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        sl_pips=sl_pips,
        tp1_pips=sl_pips,        # 1:1 RR for TP1
        tp2_pips=sl_pips * 2,    # 2:1 RR for TP2
        confidence_pct=0.0,      # set later
        lot_size=0.0,            # set later
        created_bar=0,           # set later
        created_time=None,       # set later
        expiry_bars=0,           # set later
    )


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

def _yf_fetch(
    yf_symbol: str,
    interval: str,
    max_days: int,
    label: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch a single timeframe from yfinance.  Returns normalised DataFrame or None.
    """
    import yfinance as yf
    from data.fetcher import _normalise_columns, _validate_ohlcv

    now_dt = datetime.now(timezone.utc)
    end_dt = now_dt + timedelta(days=1)
    start_dt = now_dt - timedelta(days=max_days)

    try:
        raw = yf.Ticker(yf_symbol).history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            prepost=False,
        )
    except Exception as exc:
        print(f"[Backtest] ERROR: yfinance {label} fetch failed — {exc}")
        return None

    if raw is None or raw.empty:
        print(f"[Backtest] ERROR: yfinance returned no {label} data.")
        return None

    df = _normalise_columns(raw)
    df = _validate_ohlcv(df, f"backtest:{yf_symbol}:{interval}")
    if df.empty:
        return None

    print(f"[Backtest] {label}: {len(df):,} bars "
          f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def _fetch_historical_data(
    cfg: BacktestConfig,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch M15 + H1 data using the full source hierarchy (Polygon → MT5 → yfinance).

    With Polygon as primary source we get ~2 years of M15 data.
    Bar counts reduced to match Polygon free-tier limits (~2 years).
    Each fetch is wrapped with a 60-second timeout.
    """
    import concurrent.futures

    print("\n[Backtest] Fetching dual-timeframe data via Polygon (2 years)...")
    print("[Backtest]   M15: ~47,000 bars (2 years)")
    print("[Backtest]   H1 : ~12,000 bars (2 years)\n")

    def _fetch_with_timeout(symbol, tf, bars, label):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(get_market_data, symbol, tf, bars)
            try:
                return fut.result(timeout=300)
            except concurrent.futures.TimeoutError:
                print(f"[Backtest] WARNING: {label} fetch timed out after 300s")
                return None
            except Exception as e:
                print(f"[Backtest] WARNING: {label} fetch failed: {e}")
                return None

    m15_df = _fetch_with_timeout("XAUUSD", "M15", 47000, "M15")
    h1_df  = _fetch_with_timeout("XAUUSD", "H1",  12000, "H1")

    return m15_df, h1_df


def _train_ml_for_backtest(
    m15_processed: pd.DataFrame,
    h1_processed: pd.DataFrame,
    force_retrain: bool = True,
) -> bool:
    """
    Train ML models on H1 data (1 year, ~5,900 bars).

    Why H1 only (not mixed):
      - Mixed M15+H1 confuses the model because the target variable
        (15 candles ahead) means 3.75h on M15 vs 15h on H1.
      - H1 alone gives a consistent dataset with 5,900+ bars — plenty
        for walk-forward CV.
      - The trained model filters M15 signals by predicting whether the
        broader H1 trend confirms the M15 direction.

    Returns True if models are ready.
    """
    from ml.predictor import is_model_ready, invalidate_cache

    if not force_retrain and is_model_ready():
        print("[Backtest] ML models already trained — using existing models.")
        return True

    print(f"[Backtest] Training ML on H1 data ({len(h1_processed):,} bars, ~1 year)...")

    try:
        from ml.trainer import train
        result = train(h1_processed, save=True)
        if result.rejected:
            print(f"[Backtest] ML training rejected (accuracy too low: "
                  f"XGB={result.xgb_accuracy:.1%}, RF={result.rf_accuracy:.1%})")
            print("[Backtest] Continuing without ML filter.")
            return False
        print(f"[Backtest] ML trained: XGB={result.xgb_accuracy:.1%} "
              f"RF={result.rf_accuracy:.1%} | "
              f"{result.n_samples} samples, {result.n_features} features")
        invalidate_cache()
        return True
    except Exception as exc:
        print(f"[Backtest] ML training failed: {exc}")
        print("[Backtest] Continuing without ML filter.")
        return False


def run_backtest(
    cfg: Optional[BacktestConfig] = None,
    m15_data: Optional[pd.DataFrame] = None,
    h1_data: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Run the full backtest simulation.

    Args:
        cfg:      Backtest configuration. Uses defaults if None.
        m15_data: Pre-loaded M15 DataFrame. If None, fetches from yfinance.
        h1_data:  Pre-loaded H1 DataFrame. If None, fetches from yfinance
                  (1 year of real H1 data for proper indicator warm-up).

    Returns:
        BacktestResult with all statistics, trades, and prop firm simulations.
    """
    if cfg is None:
        cfg = BacktestConfig()

    logger.info("Starting backtest with config: %s", cfg)

    # ── Step 1: Get data ───────────────────────────────────────────────
    if m15_data is None or h1_data is None:
        fetched_m15, fetched_h1 = _fetch_historical_data(cfg)
        if m15_data is None:
            m15_data = fetched_m15
        if h1_data is None:
            h1_data = fetched_h1

    if m15_data is None or m15_data.empty:
        print("\n[Backtest] FAILED: Could not fetch M15 data.")
        return BacktestResult(config=cfg, final_balance=cfg.account_balance)
    if h1_data is None or h1_data.empty:
        print("\n[Backtest] FAILED: Could not fetch H1 data.")
        return BacktestResult(config=cfg, final_balance=cfg.account_balance)

    # Process both timeframes
    m15_processed = process(m15_data, timeframe="M15", label="BT_M15")
    h1_processed = process(h1_data, timeframe="H1", label="BT_H1")

    if m15_processed is None or m15_processed.empty:
        print("\n[Backtest] FAILED: M15 data processing returned empty DataFrame.")
        return BacktestResult(config=cfg, final_balance=cfg.account_balance)
    if h1_processed is None or h1_processed.empty:
        print("\n[Backtest] FAILED: H1 data processing returned empty DataFrame.")
        return BacktestResult(config=cfg, final_balance=cfg.account_balance)

    print(f"[Backtest] Processed: M15={len(m15_processed):,} bars, H1={len(h1_processed):,} bars")

    # ── Step 1b: Fetch & cache macro data (DXY, VIX, US10Y) ───────────
    try:
        from data.macro_fetcher import fetch_and_cache_macro
        print("[Backtest] Fetching macro data (DXY, VIX, US10Y)...")
        macro_result = fetch_and_cache_macro()
        for name, count in macro_result.items():
            print(f"[Backtest]   {name}: {count} daily bars cached")
    except Exception as exc:
        print(f"[Backtest] WARNING: Macro data fetch failed: {exc}")
        print("[Backtest] ML features will not include macro context.")

    # ── Step 2: Train ML models ────────────────────────────────────────
    ml_available = False

    ml_batch: Optional[pd.DataFrame] = None
    if cfg.use_ml and Config.USE_ML_FILTER:
        ml_available = _train_ml_for_backtest(m15_processed, h1_processed)
        if ml_available:
            from ml.predictor import predict_batch as _predict_batch
            print("[Backtest] Precomputing ML predictions for all bars (batch)...")
            ml_batch = _predict_batch(m15_processed)
            if ml_batch.empty:
                print("[Backtest] ML batch prediction failed — ML filter disabled.")
                ml_available = False
                ml_batch = None
            else:
                print(f"[Backtest] ML batch ready: {len(ml_batch):,} rows")

    # ── Step 2a: Train & precompute LGBM direction classifier ──────────
    lgbm_available = False
    lgbm_batch: Optional[pd.DataFrame] = None
    if cfg.use_lgbm and Config.USE_LGBM_FILTER:
        try:
            from ml.trainer import train_lgbm
            from ml.predictor import predict_lgbm_batch as _predict_lgbm_batch

            trades_csv = os.path.join(Config.REPORTS_DIR, "backtest_trades.csv")
            print("[Backtest] Training LGBM direction classifier…")
            lgbm_result = train_lgbm(m15_processed, save=True, trades_csv=trades_csv)
            print(f"[Backtest] LGBM CV accuracy: {lgbm_result.cv_accuracy:.1%} "
                  f"({'PASS' if lgbm_result.gate_passed else 'FAIL'} ≥{Config.LGBM_MIN_CV_ACCURACY:.0%})")

            if lgbm_result.saved:
                from ml.predictor import invalidate_lgbm_cache
                invalidate_lgbm_cache()
                print("[Backtest] Precomputing LGBM predictions for all bars (batch)…")
                lgbm_batch = _predict_lgbm_batch(m15_processed)
                if lgbm_batch.empty:
                    print("[Backtest] LGBM batch prediction failed — LGBM filter disabled.")
                else:
                    n_pass = lgbm_batch["lgbm_pass"].sum()
                    n_total = lgbm_batch["lgbm_prob"].notna().sum()
                    print(f"[Backtest] LGBM batch ready: {n_total:,} rows | "
                          f"pass={n_pass:,} ({n_pass / n_total * 100:.1f}%) "
                          f"skip={n_total - n_pass:,}")
                    lgbm_available = True
        except ImportError as exc:
            print(f"[Backtest] LGBM not available: {exc}")
        except Exception as exc:
            print(f"[Backtest] LGBM training failed: {exc}")

    # ── Step 2b: Precompute indicator time-series for all bars ─────────
    print("[Backtest] Precomputing M15 indicators (full dataset)...")
    m15_precomp = PrecomputedIndicators(m15_processed)
    print("[Backtest] Precomputing H1  indicators (full dataset)...")
    h1_precomp  = PrecomputedIndicators(h1_processed)
    print("[Backtest] Indicator precomputation complete.")

    # ── Step 2c: Train HMM regime detector on first 30% of H1 bars ────
    h1_regime_states: Optional[np.ndarray] = None
    regime_labels_map = {0: "TRENDING", 1: "RANGING", 2: "CRISIS"}
    regime_multipliers = {0: 1.0, 1: 0.5, 2: 0.0}
    try:
        from analysis.regime_filter import RegimeDetector
        warmup_end = int(len(h1_processed) * 0.30)
        if warmup_end >= 200:
            print(f"[Backtest] Training HMM regime detector on first {warmup_end:,} H1 bars "
                  f"({warmup_end / len(h1_processed) * 100:.0f}% warm-up)...")
            hmm_detector = RegimeDetector()
            hmm_ok = hmm_detector.fit(h1_processed.iloc[:warmup_end])
            if hmm_ok:
                h1_regime_states = hmm_detector.predict_all(h1_processed)
                # Count distribution
                unique, counts = np.unique(h1_regime_states, return_counts=True)
                dist = {regime_labels_map.get(int(s), f"S{s}"): int(c) for s, c in zip(unique, counts)}
                total_h1 = len(h1_regime_states)
                print("[Backtest] HMM regime distribution:")
                for label, c in dist.items():
                    print(f"[Backtest]   {label:12s} {c:6,} bars ({c / total_h1 * 100:.1f}%)")
            else:
                print("[Backtest] HMM training failed — regime filter disabled.")
        else:
            print(f"[Backtest] Not enough H1 bars for HMM warm-up ({warmup_end} < 200) — regime filter disabled.")
    except ImportError:
        print("[Backtest] hmmlearn not installed — regime filter disabled.")
    except Exception as exc:
        print(f"[Backtest] HMM regime filter failed: {exc}")

    # ── Step 3 pre: Vectorised ATR-14 + 28-bar rolling avg ─────────────
    # Avoids calling m15_precomp.at(i) (which runs full calculate_all) in the
    # inner loop — critical for performance on 48k-bar datasets.
    _high = m15_processed["high"].values
    _low  = m15_processed["low"].values
    _close = m15_processed["close"].values
    _prev_close = np.roll(_close, 1)
    _prev_close[0] = _close[0]
    _tr = np.maximum(
        _high - _low,
        np.maximum(np.abs(_high - _prev_close), np.abs(_low - _prev_close))
    )
    # Wilder's ATR (EWM alpha = 1/14)
    _atr14_series = pd.Series(_tr).ewm(span=14, adjust=False).mean().values
    # 28-bar rolling average of ATR14 (shift forward so index i = avg of bars [i-28..i-1])
    _atr28_avg_series = pd.Series(_atr14_series).rolling(28, min_periods=1).mean().values

    # ── Step 3: Walk-forward simulation ────────────────────────────────
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

    signals_found = 0
    ml_filtered = 0
    lgbm_filtered = 0  # count of signals filtered by LGBM classifier
    regime_filtered = 0  # count of signals suppressed by CRISIS regime
    circuit_breaker_paused = 0  # count of signals blocked by circuit breaker
    limit_orders_placed = 0    # count of limit orders placed
    limit_orders_filled = 0    # count of limit orders filled
    limit_orders_expired = 0   # count of limit orders expired
    limit_no_sr = 0            # count of signals skipped (no S/R within range)

    # Stage 6: PnL-based circuit breaker (replaces old consecutive-loss breaker)
    cb = CircuitBreaker()
    daily_open_balance = cfg.account_balance
    current_day_str = ""
    peak_balance = cfg.account_balance
    cb_state_counts = {NORMAL: 0, CAUTION: 0, RESTRICTED: 0, HALTED: 0}
    cb_days_halted_set: set = set()
    cb_total_dd_override_count = 0
    exit_friday_close_count = 0
    exit_time_48bar_count = 0
    exit_trailing_stop_count = 0

    # Pending limit order (only one at a time, like open trades)
    pending_order: Optional[_PendingOrder] = None

    for i in range(start_idx, total_bars):
        bar = m15_processed.iloc[i]
        bar_time = m15_processed.index[i]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        # Progress logging
        bars_done = i - start_idx
        if bars_done % 1000 == 0 and bars_done > 0:
            pct = bars_done / (total_bars - start_idx) * 100
            print(f"[Backtest] Processing bar {bars_done}/{total_bars - start_idx} ({pct:.1f}%)", flush=True)

        # ── Track daily PnL + total DD for circuit breaker ────────────
        this_day = bar_time.strftime("%Y-%m-%d")
        if this_day != current_day_str:
            current_day_str = this_day
            running_balance = cfg.account_balance + sum(t.pnl_usd for t in trades)
            daily_open_balance = running_balance
            if running_balance > peak_balance:
                peak_balance = running_balance

        current_balance = cfg.account_balance + sum(t.pnl_usd for t in trades)
        if current_balance > peak_balance:
            peak_balance = current_balance

        daily_pnl_pct = ((current_balance - daily_open_balance) / cfg.account_balance * 100) if cfg.account_balance > 0 else 0.0
        total_dd_pct = ((peak_balance - current_balance) / cfg.account_balance * 100) if cfg.account_balance > 0 else 0.0

        # ── Check pending limit order fill / expiry ───────────────────
        if pending_order is not None and open_trade is None:
            if pending_order.is_expired(i):
                limit_orders_expired += 1
                pending_order = None
            elif pending_order.is_filled(bar_high, bar_low):
                limit_orders_filled += 1
                entry = pending_order.limit_price
                entry = _apply_spread(entry, pending_order.direction, cfg.spread_pips)

                open_trade = BacktestTrade(
                    entry_time=bar_time,
                    entry_price=entry,
                    direction=pending_order.direction,
                    confidence_pct=pending_order.confidence_pct,
                    lot_size=pending_order.lot_size,
                    stop_loss=pending_order.stop_loss,
                    tp1_price=pending_order.tp1_price,
                    tp2_price=pending_order.tp2_price,
                    sl_pips=pending_order.sl_pips,
                    tp1_pips=pending_order.tp1_pips,
                    tp2_pips=pending_order.tp2_pips,
                    entry_bar_idx=i,
                    initial_sl_pips=pending_order.sl_pips,
                )
                pending_order = None
                bars_since_last_trade = 0
                continue  # don't check exit on fill bar

        # ── Manage open trade: new exits + original SL/TP ─────────────
        if open_trade is not None:
            # Use vectorised ATR (O(1))
            cur_atr = float(_atr14_series[i])

            bars_held = i - open_trade.entry_bar_idx

            # a) Friday 20:00 UTC close (weekend gap protection)
            if should_friday_close(bar_time):
                _close_trade_at_market(open_trade, bar_close, bar_time, "FRIDAY_CLOSE")
                trades.append(open_trade)
                exit_friday_close_count += 1
                open_trade = None
                bars_since_last_trade = 0
                continue

            # b) 48-bar time exit (12 hours)
            if should_time_exit(bars_held):
                _close_trade_at_market(open_trade, bar_close, bar_time, "TIME_EXIT_48")
                trades.append(open_trade)
                exit_time_48bar_count += 1
                open_trade = None
                bars_since_last_trade = 0
                continue

            # c) Trailing stop update + check
            if cur_atr > 0:
                open_trade.trailing_stop = update_trailing_stop(
                    direction=open_trade.direction,
                    entry_price=open_trade.entry_price,
                    current_price=bar_close,
                    current_trail_stop=open_trade.trailing_stop,
                    sl_pips=open_trade.initial_sl_pips,
                    atr_value=cur_atr,
                )
            if open_trade.trailing_stop is not None:
                trail_hit = False
                if open_trade.direction == "BUY" and bar_low <= open_trade.trailing_stop:
                    trail_hit = True
                elif open_trade.direction == "SELL" and bar_high >= open_trade.trailing_stop:
                    trail_hit = True
                if trail_hit:
                    _close_trade_at_market(open_trade, open_trade.trailing_stop, bar_time, "TRAIL_STOP")
                    trades.append(open_trade)
                    exit_trailing_stop_count += 1
                    open_trade = None
                    bars_since_last_trade = 0
                    continue

            # d) Original SL/TP exit
            closed = _check_exit(open_trade, bar_high, bar_low, bar_time)
            if closed:
                trades.append(open_trade)
                open_trade = None
                bars_since_last_trade = 0
                continue

        bars_since_last_trade += 1

        # ── Skip if we already have an open trade or pending order ────
        if open_trade is not None or pending_order is not None:
            continue

        # ── Skip if too soon after last trade ─────────────────────────
        if bars_since_last_trade < cfg.min_bars_between_trades:
            continue

        # ── Run analysis on lookback window ───────────────────────────
        m15_slice = m15_processed.iloc[max(0, i - lookback):i + 1]

        h1_idx = h1_processed.index.searchsorted(bar_time, side="right")
        h1_slice = h1_processed.iloc[max(0, h1_idx - lookback):h1_idx]

        if len(m15_slice) < min_bars or len(h1_slice) < 50:
            continue

        # ── Analyse — use precomputed indicators for O(1) lookups ─────
        h1_bar_idx = h1_idx - 1 if h1_idx > 0 else 0
        analysis = _analyse_bar(
            m15_slice, h1_slice, bar_time=bar_time,
            m15_precomp=m15_precomp, m15_bar_idx=i,
            h1_precomp=h1_precomp,  h1_bar_idx=h1_bar_idx,
        )

        if analysis.direction is None or analysis.direction == "WAIT" or analysis.risk is None:
            continue

        direction = analysis.direction
        confidence = analysis.confidence
        risk = analysis.risk
        signals_found += 1

        # ── ML filter — use precomputed batch (O(1) lookup) ───────────
        if ml_available and ml_batch is not None:
            ml_row = ml_batch.iloc[i]
            pred_available = not pd.isna(ml_row["xgb_prob"])
            if pred_available and not (
                bool(ml_row["ml_agree"]) and
                ((direction == "BUY" and ml_row["ml_direction"] == "UP") or
                 (direction == "SELL" and ml_row["ml_direction"] == "DOWN"))
            ):
                ml_filtered += 1
                continue

        # ── LGBM filter — use precomputed batch (O(1) lookup) ────────
        if lgbm_available and lgbm_batch is not None:
            lgbm_row = lgbm_batch.iloc[i]
            lgbm_prob_available = not pd.isna(lgbm_row["lgbm_prob"])
            if lgbm_prob_available and not bool(lgbm_row["lgbm_pass"]):
                lgbm_filtered += 1
                continue

        # ── HMM regime filter ─────────────────────────────────────────
        bar_regime_state = 1  # default RANGING
        bar_regime_label = "RANGING"
        bar_regime_mult = 1.0
        if h1_regime_states is not None:
            h1_regime_idx = h1_processed.index.searchsorted(bar_time, side="right") - 1
            h1_regime_idx = max(0, min(h1_regime_idx, len(h1_regime_states) - 1))
            bar_regime_state = int(h1_regime_states[h1_regime_idx])
            bar_regime_label = regime_labels_map.get(bar_regime_state, f"S{bar_regime_state}")
            bar_regime_mult = regime_multipliers.get(bar_regime_state, 1.0)

            if bar_regime_mult == 0.0:
                regime_filtered += 1
                continue

        # ── Circuit breaker gate ──────────────────────────────────────
        cb_state = cb.get_circuit_state(daily_pnl_pct, total_dd_pct)
        if not cb.is_signal_allowed(daily_pnl_pct, total_dd_pct, confidence):
            circuit_breaker_paused += 1
            cb_state_counts[cb_state] = cb_state_counts.get(cb_state, 0) + 1
            if cb_state == HALTED:
                cb_days_halted_set.add(this_day)
            continue
        cb_state_counts[cb_state] = cb_state_counts.get(cb_state, 0) + 1

        # Track total DD override activations
        cb_mult = cb.get_size_multiplier(daily_pnl_pct, total_dd_pct)
        if cb.total_dd_override_active:
            cb_total_dd_override_count += 1

        # ── Half-Kelly ATR-adjusted position sizing ───────────────────
        pip_value = Config.GOLD_PIP_VALUE

        # Use vectorised ATR arrays (O(1) — precomputed before loop)
        cur_atr = float(_atr14_series[i])
        avg_atr_28 = float(_atr28_avg_series[i])

        # Build recent trades window for Kelly calculation
        recent_for_kelly = [
            {"pnl_pips": t.pnl_pips, "is_winner": t.is_winner, "sl_pips": t.initial_sl_pips or t.sl_pips}
            for t in trades[-50:]
        ]

        risk_pct = calculate_half_kelly_risk_pct(
            recent_trades=recent_for_kelly,
            current_atr=cur_atr,
            avg_atr_28=avg_atr_28,
            circuit_multiplier=cb_mult,
        )

        # Apply regime multiplier on top
        if bar_regime_mult < 1.0:
            risk_pct *= bar_regime_mult
            risk_pct = max(0.1, risk_pct)

        # ── Limit order entry (at S/R levels) ─────────────────────────
        if cfg.use_limit_orders:
            pending = _find_limit_entry(
                direction=direction,
                current_price=analysis.current_price,
                sr_levels=analysis.sr_levels,
                max_distance_pips=cfg.limit_max_distance_pips,
                sl_pips=cfg.limit_sl_pips,
            )
            if pending is None:
                limit_no_sr += 1
                continue

            sl_for_lot = cfg.limit_sl_pips
            risk_usd = current_balance * (risk_pct / 100)
            lot_size = risk_usd / (sl_for_lot * pip_value) if sl_for_lot > 0 else 0.01
            lot_size = round(max(0.01, lot_size), 2)

            pending.lot_size = lot_size
            pending.confidence_pct = confidence
            pending.created_bar = i
            pending.created_time = bar_time
            pending.expiry_bars = cfg.limit_expiry_bars

            pending_order = pending
            limit_orders_placed += 1
            bars_since_last_trade = 0

        else:
            # ── Market order entry ────────────────────────────────────
            entry = risk.entry_price
            entry = _apply_spread(entry, direction, cfg.spread_pips)
            entry = _apply_slippage(entry, direction, cfg.slippage_pips)

            risk_usd = current_balance * (risk_pct / 100)
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
                regime_state=bar_regime_state,
                regime_label=bar_regime_label,
                entry_bar_idx=i,
                initial_sl_pips=risk.sl_pips,
                risk_pct_used=risk_pct,
            )
            bars_since_last_trade = 0

    # ── Close any remaining open trade at last bar's close ─────────────
    if open_trade is not None:
        last_bar = m15_processed.iloc[-1]
        last_time = m15_processed.index[-1]
        _close_trade_at_market(open_trade, last_bar["close"], last_time, "END_OF_DATA")
        trades.append(open_trade)

    # Cancel any remaining pending order
    if pending_order is not None:
        limit_orders_expired += 1
        pending_order = None

    if cfg.use_limit_orders:
        print(f"[Backtest] Signals found: {signals_found} | ML filtered: {ml_filtered} "
              f"| LGBM filtered: {lgbm_filtered} "
              f"| Regime filtered: {regime_filtered} "
              f"| No S/R: {limit_no_sr} | Limits placed: {limit_orders_placed} "
              f"| Filled: {limit_orders_filled} | Expired: {limit_orders_expired} "
              f"| CB blocked: {circuit_breaker_paused} | Trades: {len(trades)}")
    else:
        print(f"[Backtest] Signals found: {signals_found} | ML filtered: {ml_filtered} "
              f"| LGBM filtered: {lgbm_filtered} "
              f"| Regime filtered: {regime_filtered} "
              f"| CB blocked: {circuit_breaker_paused} | Trades taken: {len(trades)}")
    print(f"[Backtest] Exits — Friday: {exit_friday_close_count} | "
          f"48-bar: {exit_time_48bar_count} | Trail: {exit_trailing_stop_count}")
    logger.info("Backtest complete: %d trades generated.", len(trades))

    # ── Step 4: Compute statistics ─────────────────────────────────────
    result = _compute_statistics(trades, cfg)

    # ── Step 4b: Add regime distribution to result ─────────────────────
    if h1_regime_states is not None:
        total_h1 = len(h1_regime_states)
        unique, counts = np.unique(h1_regime_states, return_counts=True)
        result.regime_distribution = {
            regime_labels_map.get(int(s), f"S{s}"): round(int(c) / total_h1 * 100, 1)
            for s, c in zip(unique, counts)
        }
        result.regime_filtered = regime_filtered

    # ── Step 4c: Add circuit breaker stats to result ───────────────────
    result.cb_state_counts = dict(cb_state_counts)
    result.cb_days_halted = len(cb_days_halted_set)
    result.cb_total_dd_overrides = cb_total_dd_override_count
    result.exit_friday_close = exit_friday_close_count
    result.exit_time_48bar = exit_time_48bar_count
    result.exit_trailing_stop = exit_trailing_stop_count
    if trades:
        risk_pcts = [t.risk_pct_used for t in trades if t.risk_pct_used > 0]
        result.avg_risk_pct = float(np.mean(risk_pcts)) if risk_pcts else 0.0

    # ── Step 5: Prop firm simulations ──────────────────────────────────
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
    print(f"Confidence threshold: {Config.MIN_CONFIDENCE_PCT}% (same as live)")
    print(f"ML filter: enabled\n")
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
