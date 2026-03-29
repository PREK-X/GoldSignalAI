"""
GoldSignalAI — signals/risk_manager.py
========================================
Calculates Stop Loss, Take Profit, lot size, and risk metrics for
every signal.

SL/TP rules:
  Stop Loss:
    - ATR-based: Entry ± (1.5 × ATR14)
    - Clamped: minimum 10 pips, maximum 30 pips (Gold)
    - Placed just beyond nearest S/R level when available
  Take Profit:
    - TP1: 1:2 risk/reward (close 50% of position)
    - TP2: 1:3 risk/reward (close remaining 50%)

Lot size:
    - Based on user's account balance and risk %
    - Formula: lot = (balance × risk%) / (sl_pips × pip_value)
    - Default: 1% risk per trade (Config.RISK_PER_TRADE_PCT)

All calculations use pips. For Gold (XAU/USD):
    1 pip = $0.10 movement per 0.01 lot
    1 pip = $1.00 movement per 0.1 lot
    1 pip = $10.00 movement per 1.0 standard lot
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from config import Config
from analysis.sr_levels import SRLevels
from infrastructure.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskParameters:
    """
    Complete risk management output for one signal.

    All price values are absolute (e.g. 2312.50).
    All pip values are in Gold pips (1 pip = 0.10 price movement).
    """
    # Entry
    entry_price:      float

    # Stop Loss
    stop_loss:        float
    sl_pips:          float
    sl_usd_per_lot:   float  # $ risk per standard lot at this SL

    # Take Profit 1 (1:2 R/R — close 50%)
    tp1_price:        float
    tp1_pips:         float
    tp1_rr:           float  # actual R/R ratio (e.g. 2.0)

    # Take Profit 2 (1:3 R/R — close remaining 50%)
    tp2_price:        float
    tp2_pips:         float
    tp2_rr:           float

    # Lot sizing
    suggested_lot:    float  # lot size for configured risk %
    risk_usd:         float  # $ amount at risk
    potential_tp1_usd: float # $ profit if TP1 hit
    potential_tp2_usd: float # $ profit if TP2 hit

    # Metadata
    direction:        str    # "BUY" or "SELL"
    atr_value:        float  # raw ATR used for calculation
    sl_method:        str    # "ATR" or "ATR+SR"

    def summary(self) -> str:
        return (
            f"SL={self.stop_loss:.2f} ({self.sl_pips:.1f} pips) | "
            f"TP1={self.tp1_price:.2f} ({self.tp1_pips:.1f} pips, 1:{self.tp1_rr:.1f}) | "
            f"TP2={self.tp2_price:.2f} ({self.tp2_pips:.1f} pips, 1:{self.tp2_rr:.1f}) | "
            f"Lot={self.suggested_lot:.2f} Risk=${self.risk_usd:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PRICE ↔ PIP CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def price_to_pips(price_distance: float) -> float:
    """Convert an absolute price distance to pips. Gold: 1 pip = 0.10."""
    return abs(price_distance) / Config.PIP_SIZE


def pips_to_price(pips: float) -> float:
    """Convert pips to absolute price distance. Gold: 1 pip = 0.10."""
    return abs(pips) * Config.PIP_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# STOP LOSS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_sl(
    entry:     float,
    direction: str,
    atr:       float,
    sr_levels: Optional[SRLevels] = None,
) -> tuple[float, float, str]:
    """
    Calculate the stop loss price.

    Logic:
      1. Base SL = entry ± (ATR × multiplier)
      2. Clamp to [MIN_SL_PIPS, MAX_SL_PIPS]
      3. If an S/R level exists just beyond the base SL, extend SL
         to sit just beyond that level (gives the trade room to breathe
         without getting stopped on a wick to S/R)

    Args:
        entry:     Entry price
        direction: "BUY" or "SELL"
        atr:       Current ATR value
        sr_levels: Optional S/R analysis for SL refinement

    Returns:
        (sl_price, sl_pips, method_label)
    """
    multiplier = Config.ATR_SL_MULTIPLIER
    min_sl     = Config.MIN_SL_PIPS
    max_sl     = Config.MAX_SL_PIPS

    # Base ATR distance
    atr_distance = atr * multiplier
    sl_pips      = price_to_pips(atr_distance)

    # Clamp
    sl_pips = max(min_sl, min(max_sl, sl_pips))
    sl_distance = pips_to_price(sl_pips)

    if direction == "BUY":
        sl_price = entry - sl_distance
    else:
        sl_price = entry + sl_distance

    method = "ATR"

    # ── S/R refinement ────────────────────────────────────────────────────
    # If there's a strong S/R level near the SL, extend SL just beyond it.
    # This prevents getting stopped right AT a support/resistance bounce.
    if sr_levels is not None:
        buffer_pips = 2.0  # place SL 2 pips beyond the S/R level
        buffer_price = pips_to_price(buffer_pips)

        if direction == "BUY" and sr_levels.nearest_support is not None:
            sr_price = sr_levels.nearest_support.price
            # Only use if the S/R is between entry and our calculated SL
            # or just slightly below the SL (within 5 pips)
            sr_sl_candidate = sr_price - buffer_price
            candidate_pips  = price_to_pips(entry - sr_sl_candidate)

            if min_sl <= candidate_pips <= max_sl:
                sl_price = sr_sl_candidate
                sl_pips  = candidate_pips
                method   = "ATR+SR"

        elif direction == "SELL" and sr_levels.nearest_resistance is not None:
            sr_price = sr_levels.nearest_resistance.price
            sr_sl_candidate = sr_price + buffer_price
            candidate_pips  = price_to_pips(sr_sl_candidate - entry)

            if min_sl <= candidate_pips <= max_sl:
                sl_price = sr_sl_candidate
                sl_pips  = candidate_pips
                method   = "ATR+SR"

    return sl_price, sl_pips, method


# ─────────────────────────────────────────────────────────────────────────────
# TAKE PROFIT CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_tp(
    entry:     float,
    sl_pips:   float,
    direction: str,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate TP1 (1:2 R/R) and TP2 (1:3 R/R) prices.

    Args:
        entry:     Entry price
        sl_pips:   Stop loss distance in pips
        direction: "BUY" or "SELL"

    Returns:
        (tp1_price, tp1_pips, tp1_rr, tp2_price, tp2_pips, tp2_rr)
    """
    tp1_rr = Config.TP1_RR_RATIO
    tp2_rr = Config.TP2_RR_RATIO

    tp1_pips = sl_pips * tp1_rr
    tp2_pips = sl_pips * tp2_rr

    tp1_distance = pips_to_price(tp1_pips)
    tp2_distance = pips_to_price(tp2_pips)

    if direction == "BUY":
        tp1_price = entry + tp1_distance
        tp2_price = entry + tp2_distance
    else:
        tp1_price = entry - tp1_distance
        tp2_price = entry - tp2_distance

    return tp1_price, tp1_pips, tp1_rr, tp2_price, tp2_pips, tp2_rr


# ─────────────────────────────────────────────────────────────────────────────
# LOT SIZE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_lot_size(
    sl_pips:       float,
    account_balance: float = Config.CHALLENGE_ACCOUNT_SIZE,
    risk_pct:        float = Config.RISK_PER_TRADE_PCT,
) -> tuple[float, float]:
    """
    Calculate the appropriate lot size based on account balance and risk %.

    Formula:
        risk_usd = account_balance × (risk_pct / 100)
        lot_size = risk_usd / (sl_pips × pip_value_per_lot)

    For Gold (XAU/USD):
        1 pip = $10 per standard lot (1.0)
        1 pip = $1  per mini lot (0.1)
        1 pip = $0.10 per micro lot (0.01)

    Args:
        sl_pips:         Stop loss in pips
        account_balance: Account balance in USD
        risk_pct:        % of account to risk

    Returns:
        (lot_size, risk_usd)
    """
    if sl_pips <= 0:
        return 0.01, 0.0   # safety minimum

    risk_usd = account_balance * (risk_pct / 100)
    pip_value_per_lot = Config.GOLD_PIP_VALUE   # $10 per pip per standard lot

    lot_size = risk_usd / (sl_pips * pip_value_per_lot)

    # Round to 2 decimals (most brokers accept 0.01 increments)
    lot_size = round(lot_size, 2)

    # Enforce minimum 0.01 lot
    lot_size = max(0.01, lot_size)

    return lot_size, risk_usd


# ─────────────────────────────────────────────────────────────────────────────
# HALF-KELLY ATR-ADJUSTED POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────

# Hard caps for dynamic risk %
_MIN_RISK_PCT = 0.1
_MAX_RISK_PCT = 2.0
_KELLY_MIN_TRADES = 20   # fallback to flat 1% if fewer trades in window
_KELLY_WINDOW = 50       # rolling window of closed trades

# ATR adjustment thresholds
_ATR_HIGH_RATIO = 1.5    # high-vol: scale down
_ATR_LOW_RATIO = 0.7     # low-vol: scale up
_ATR_HIGH_MULT = 0.7
_ATR_LOW_MULT = 1.2


def calculate_half_kelly_risk_pct(
    recent_trades: list[dict],
    current_atr: float,
    avg_atr_28: float,
    circuit_multiplier: float = 1.0,
) -> float:
    """
    Compute dynamic risk % using Half-Kelly with ATR adjustment.

    Args:
        recent_trades: Last N closed trades, each with keys
                       {"pnl_pips", "is_winner", "sl_pips"}.
                       Most recent last.
        current_atr:   Current ATR-14 value.
        avg_atr_28:    Rolling 28-bar average ATR.
        circuit_multiplier: From circuit breaker (0.0 – 1.0).

    Returns:
        Risk % per trade (clamped to [0.1, 2.0]).
    """
    # Step 1: Half-Kelly fraction
    window = recent_trades[-_KELLY_WINDOW:]
    if len(window) < _KELLY_MIN_TRADES:
        half_kelly_pct = Config.RISK_PER_TRADE_PCT  # fallback 1.0%
    else:
        wins = [t for t in window if t["is_winner"]]
        losses = [t for t in window if not t["is_winner"]]
        win_rate = len(wins) / len(window)
        loss_rate = 1.0 - win_rate

        avg_win = np.mean([abs(t["pnl_pips"]) for t in wins]) if wins else 0.0
        avg_loss = np.mean([abs(t["pnl_pips"]) for t in losses]) if losses else 1.0

        if avg_win <= 0:
            half_kelly_pct = _MIN_RISK_PCT
        else:
            kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
            half_kelly_pct = max(kelly * 0.5, 0.0) * 100  # convert fraction to %
            if half_kelly_pct <= 0:
                half_kelly_pct = _MIN_RISK_PCT

    # Step 2: ATR adjustment
    if avg_atr_28 > 0:
        atr_ratio = current_atr / avg_atr_28
        if atr_ratio > _ATR_HIGH_RATIO:
            half_kelly_pct *= _ATR_HIGH_MULT
        elif atr_ratio < _ATR_LOW_RATIO:
            half_kelly_pct *= _ATR_LOW_MULT

    # Step 3: Apply circuit breaker multiplier
    half_kelly_pct *= circuit_multiplier

    # Step 4: Hard caps
    return max(_MIN_RISK_PCT, min(_MAX_RISK_PCT, half_kelly_pct))


# ─────────────────────────────────────────────────────────────────────────────
# EXIT CONDITION HELPERS (for backtest engine)
# ─────────────────────────────────────────────────────────────────────────────

def should_friday_close(bar_time: datetime) -> bool:
    """Return True if bar_time is Friday >= 20:00 UTC (weekend gap protection)."""
    # weekday(): Monday=0 ... Friday=4
    return bar_time.weekday() == 4 and bar_time.hour >= 20


def should_time_exit(bars_since_entry: int, max_bars: int = 48) -> bool:
    """Return True if trade has been open for >= max_bars M15 bars (12 hours)."""
    return bars_since_entry >= max_bars


def update_trailing_stop(
    direction: str,
    entry_price: float,
    current_price: float,
    current_trail_stop: Optional[float],
    sl_pips: float,
    atr_value: float,
) -> Optional[float]:
    """
    Compute updated trailing stop.

    Activates when unrealised profit >= 1R (1x initial risk in pips).
    Trail distance = 0.5 * ATR14.
    Never moves against the position direction.

    Args:
        direction:          "BUY" or "SELL"
        entry_price:        Trade entry price
        current_price:      Current bar close
        current_trail_stop: Existing trailing stop (None if not yet active)
        sl_pips:            Initial risk in pips (1R)
        atr_value:          Current ATR14

    Returns:
        Updated trailing stop price, or None if not yet activated.
    """
    initial_risk_price = pips_to_price(sl_pips)
    trail_distance = atr_value * 0.5

    if direction == "BUY":
        unrealised = current_price - entry_price
        if unrealised < initial_risk_price:
            return current_trail_stop  # not yet at 1R
        new_trail = current_price - trail_distance
        if current_trail_stop is not None:
            return max(current_trail_stop, new_trail)  # never move down
        return new_trail
    else:  # SELL
        unrealised = entry_price - current_price
        if unrealised < initial_risk_price:
            return current_trail_stop
        new_trail = current_price + trail_distance
        if current_trail_stop is not None:
            return min(current_trail_stop, new_trail)  # never move up
        return new_trail


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def calculate_risk(
    entry_price:     float,
    direction:       str,
    atr_value:       float,
    sr_levels:       Optional[SRLevels] = None,
    account_balance: float = Config.CHALLENGE_ACCOUNT_SIZE,
    risk_pct:        float = Config.RISK_PER_TRADE_PCT,
) -> RiskParameters:
    """
    Calculate all risk parameters for a signal.

    This is the single function called by signals/generator.py for
    every BUY or SELL signal.

    Args:
        entry_price:     Current price (will be the entry)
        direction:       "BUY" or "SELL"
        atr_value:       Current ATR14 value from indicators
        sr_levels:       S/R analysis for SL refinement (optional)
        account_balance: Account size in USD
        risk_pct:        % of account to risk per trade

    Returns:
        RiskParameters with all SL/TP/lot data.
    """
    # ── Stop Loss ─────────────────────────────────────────────────────────
    sl_price, sl_pips, sl_method = _calculate_sl(
        entry_price, direction, atr_value, sr_levels
    )

    # ── Take Profits ──────────────────────────────────────────────────────
    tp1_price, tp1_pips, tp1_rr, tp2_price, tp2_pips, tp2_rr = _calculate_tp(
        entry_price, sl_pips, direction
    )

    # ── Lot Size ──────────────────────────────────────────────────────────
    lot_size, risk_usd = _calculate_lot_size(sl_pips, account_balance, risk_pct)

    # ── USD values ────────────────────────────────────────────────────────
    pip_value = Config.GOLD_PIP_VALUE  # $10 per pip per standard lot
    sl_usd    = sl_pips  * pip_value * lot_size
    tp1_usd   = tp1_pips * pip_value * lot_size
    tp2_usd   = tp2_pips * pip_value * lot_size

    result = RiskParameters(
        entry_price=entry_price,
        stop_loss=sl_price,
        sl_pips=sl_pips,
        sl_usd_per_lot=sl_pips * pip_value,
        tp1_price=tp1_price,
        tp1_pips=tp1_pips,
        tp1_rr=tp1_rr,
        tp2_price=tp2_price,
        tp2_pips=tp2_pips,
        tp2_rr=tp2_rr,
        suggested_lot=lot_size,
        risk_usd=risk_usd,
        potential_tp1_usd=tp1_usd,
        potential_tp2_usd=tp2_usd,
        direction=direction,
        atr_value=atr_value,
        sl_method=sl_method,
    )

    logger.info(
        "Risk [%s @ %.2f]: %s",
        direction, entry_price, result.summary()
    )
    return result
