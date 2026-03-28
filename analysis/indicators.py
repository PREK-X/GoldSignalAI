"""
GoldSignalAI — analysis/indicators.py
======================================
Calculates all 13 technical indicators used in the signal scoring engine.

Each indicator returns an IndicatorResult dataclass containing:
  - signal   : "bullish" | "bearish" | "neutral"
  - value    : primary numeric value (e.g. RSI = 62.4)
  - values   : dict of all computed sub-values for display & ML features
  - reason   : human-readable explanation of the signal decision

The top-level function `calculate_all()` returns an AllIndicators
object holding all results and is the only thing other modules need
to import.

Indicator → Signal mapping (12 voted + BBands ML-only):
  1.  HMA/EMA Alignment   — price vs HMA20/EMA50/EMA200 stack
  2.  ADX                  — trend strength gate
  3.  Ichimoku Cloud       — price vs cloud + kumo twist
  4.  RSI 14               — oversold / overbought + divergence
  5.  MACD                 — crossover + histogram momentum
  6.  Williams %R (14)     — overbought / oversold momentum
  7.  CCI 20               — below -100 / above +100
  8.  Bollinger Bands      — ML feature only (excluded from voting)
  9.  Supertrend (10,3)    — trend direction via ATR bands
  10. Connors RSI          — composite 3-part momentum oscillator
  11. Keltner Channels     — squeeze/breakout detection
  12. ATR                  — volatility context (neutral signal, value used for SL)
  13. Volume               — current vs 20-period average
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    ta = None
    PANDAS_TA_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPES
# ─────────────────────────────────────────────────────────────────────────────

BULLISH  = "bullish"
BEARISH  = "bearish"
NEUTRAL  = "neutral"

Signal = str   # one of the three constants above


@dataclass
class IndicatorResult:
    """
    Container for a single indicator's output.
    Keeps raw values separate from the scoring decision so the dashboard
    can display numbers and the scoring engine only looks at `signal`.
    """
    name:   str
    signal: Signal                          # "bullish" | "bearish" | "neutral"
    value:  float                           # Primary display value
    values: dict = field(default_factory=dict)  # All sub-values
    reason: str  = ""                       # Why this signal was chosen

    def score(self) -> int:
        """Convert signal to +1 / 0 / -1 for the scoring engine."""
        return 1 if self.signal == BULLISH else (-1 if self.signal == BEARISH else 0)


@dataclass
class AllIndicators:
    """
    Holds all 13 indicator results plus convenience properties.
    Passed to scoring.py, formatter.py, and the ML feature builder.
    """
    ema:         IndicatorResult     # HMA-20 / EMA-50 / EMA-200 alignment
    adx:         IndicatorResult
    ichimoku:    IndicatorResult
    rsi:         IndicatorResult
    macd:        IndicatorResult
    williams_r:  IndicatorResult     # Replaces Stochastic
    cci:         IndicatorResult
    bbands:      IndicatorResult     # Kept for ML features; excluded from voting
    supertrend:  IndicatorResult     # NEW — trend direction
    connors_rsi: IndicatorResult     # NEW — composite momentum
    keltner:     IndicatorResult     # NEW — squeeze/breakout
    atr:         IndicatorResult     # directional-neutral; value used for SL
    volume:      IndicatorResult

    # Cached latest close (set by calculate_all)
    latest_close: float = 0.0

    def as_list(self) -> list[IndicatorResult]:
        """Return the 13 voted indicators (BBands excluded — negative predictive accuracy).
        ATR is directionally neutral but included for the scoring engine's neutral count."""
        return [
            self.ema, self.adx, self.ichimoku, self.rsi,
            self.macd, self.williams_r, self.cci,
            self.supertrend, self.connors_rsi, self.keltner,
            self.atr, self.volume,
        ]

    def bullish_count(self) -> int:
        return sum(1 for r in self.as_list() if r.signal == BULLISH)

    def bearish_count(self) -> int:
        return sum(1 for r in self.as_list() if r.signal == BEARISH)

    def neutral_count(self) -> int:
        return sum(1 for r in self.as_list() if r.signal == NEUTRAL)

    def net_score(self) -> int:
        return sum(r.score() for r in self.as_list())

    def summary_line(self) -> str:
        b = self.bullish_count()
        br = self.bearish_count()
        n = len(self.as_list())
        return f"{b}/{n} Bullish, {br}/{n} Bearish"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    """Convert any scalar to float, returning nan on failure."""
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _last(series: pd.Series) -> float:
    """Return the last non-NaN value of a Series, or nan."""
    dropped = series.dropna()
    return _safe_float(dropped.iloc[-1]) if len(dropped) > 0 else np.nan


def _prev(series: pd.Series) -> float:
    """Return the second-to-last non-NaN value of a Series, or nan."""
    dropped = series.dropna()
    return _safe_float(dropped.iloc[-2]) if len(dropped) > 1 else np.nan


def _neutral(name: str, reason: str = "Insufficient data") -> IndicatorResult:
    """Shorthand for a neutral result when data is missing."""
    return IndicatorResult(name=name, signal=NEUTRAL, value=np.nan,
                           values={}, reason=reason)


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 1 — EMA ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average (linearly weighted)."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma_half = _wma(series, half_period)
    wma_full = _wma(series, period)
    diff = 2 * wma_half - wma_full
    return _wma(diff, sqrt_period)


def calc_ema(df: pd.DataFrame) -> IndicatorResult:
    """
    HMA/EMA Alignment: price vs HMA20 vs EMA50 vs EMA200 stack.

    Uses Hull Moving Average (HMA-20) as the fast line for reduced lag,
    while keeping EMA-50 and EMA-200 as the mid/slow trend anchors.

    BUY  signal: price > HMA20 > EMA50 > EMA200  (perfect bull stack)
    SELL signal: price < HMA20 < EMA50 < EMA200  (perfect bear stack)
    NEUTRAL: mixed / EMA200 not yet calculable
    """
    name = "HMA/EMA Alignment"
    close = df["close"]

    if len(df) < Config.EMA_SLOW:
        return _neutral(name, f"Need {Config.EMA_SLOW}+ candles for EMA200")

    try:
        hma20  = _last(_hma(close, Config.EMA_FAST))
        ema50  = _last(close.ewm(span=Config.EMA_MID,   adjust=False).mean())
        ema200 = _last(close.ewm(span=Config.EMA_SLOW,  adjust=False).mean())
        price  = _last(close)

        if any(np.isnan(v) for v in [price, hma20, ema50, ema200]):
            return _neutral(name, "NaN in HMA/EMA calculation")

        values = {
            "price": price, "hma20": hma20,
            "ema50": ema50, "ema200": ema200,
        }

        if price > hma20 > ema50 > ema200:
            return IndicatorResult(
                name=name, signal=BULLISH, value=hma20, values=values,
                reason=f"Price({price:.2f}) > HMA20({hma20:.2f}) > EMA50({ema50:.2f}) > EMA200({ema200:.2f})"
            )
        if price < hma20 < ema50 < ema200:
            return IndicatorResult(
                name=name, signal=BEARISH, value=hma20, values=values,
                reason=f"Price({price:.2f}) < HMA20({hma20:.2f}) < EMA50({ema50:.2f}) < EMA200({ema200:.2f})"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=hma20, values=values,
            reason="HMA/EMA stack not aligned — mixed trend"
        )

    except Exception as exc:
        logger.exception("HMA/EMA calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 2 — ADX
# ─────────────────────────────────────────────────────────────────────────────

def calc_adx(df: pd.DataFrame) -> IndicatorResult:
    """
    ADX (Average Directional Index) — trend strength gate.

    This indicator is intentionally used as a GATE, not a direction signal:
      - ADX < 25   → NEUTRAL  (trend too weak, avoid trading)
      - ADX 25–40  → confirms direction from +DI/-DI (BULLISH or BEARISH)
      - ADX > 40   → BULLISH/BEARISH with extra strength note

    Direction comes from +DI vs -DI:
      +DI > -DI = uptrend (bullish)
      +DI < -DI = downtrend (bearish)

    If ADX < 25 the market is ranging; we output NEUTRAL regardless of
    +DI/-DI because ranging markets produce false trend signals.
    """
    name = "ADX"
    period = Config.ADX_PERIOD

    if len(df) < period * 2:
        return _neutral(name, f"Need {period*2}+ candles for ADX")

    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # True range components
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        # Directional movement
        up_move   = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s  = pd.Series(plus_dm,  index=df.index)
        minus_dm_s = pd.Series(minus_dm, index=df.index)

        # Wilder smoothing
        atr_s     = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di   = 100 * plus_dm_s.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
        minus_di  = 100 * minus_dm_s.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
        dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx_line  = dx.ewm(alpha=1/period, adjust=False).mean()

        adx_val   = _last(adx_line)
        plus_val  = _last(plus_di)
        minus_val = _last(minus_di)

        if any(np.isnan(v) for v in [adx_val, plus_val, minus_val]):
            return _neutral(name, "NaN in ADX calculation")

        values = {"adx": adx_val, "+di": plus_val, "-di": minus_val}
        strong = adx_val > Config.ADX_STRONG_THRESHOLD

        if adx_val < Config.ADX_TREND_THRESHOLD:
            return IndicatorResult(
                name=name, signal=NEUTRAL, value=adx_val, values=values,
                reason=f"ADX={adx_val:.1f} < {Config.ADX_TREND_THRESHOLD} — weak/ranging market"
            )

        suffix = " (very strong)" if strong else ""
        if plus_val > minus_val:
            return IndicatorResult(
                name=name, signal=BULLISH, value=adx_val, values=values,
                reason=f"ADX={adx_val:.1f}{suffix}, +DI({plus_val:.1f}) > -DI({minus_val:.1f})"
            )
        return IndicatorResult(
            name=name, signal=BEARISH, value=adx_val, values=values,
            reason=f"ADX={adx_val:.1f}{suffix}, -DI({minus_val:.1f}) > +DI({plus_val:.1f})"
        )

    except Exception as exc:
        logger.exception("ADX calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 3 — ICHIMOKU CLOUD
# ─────────────────────────────────────────────────────────────────────────────

def calc_ichimoku(df: pd.DataFrame) -> IndicatorResult:
    """
    Ichimoku Cloud (Tenkan/Kijun/Senkou A & B).

    BUY signals:
      - Price above both Senkou A and B (price above cloud)
      - Optional: Tenkan > Kijun (TK cross confirmation)

    SELL signals:
      - Price below both Senkou A and B (price below cloud)

    NEUTRAL:
      - Price inside the cloud (indecision zone)
      - Kumo twist (Senkou A crosses Senkou B) = imminent change, wait

    We use standard settings 9/26/52 as configured in Config.
    """
    name = "Ichimoku"
    tenkan_p = Config.ICHIMOKU_TENKAN
    kijun_p  = Config.ICHIMOKU_KIJUN
    senkou_p = Config.ICHIMOKU_SENKOU

    if len(df) < senkou_p + kijun_p:
        return _neutral(name, f"Need {senkou_p + kijun_p}+ candles for Ichimoku")

    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2
        # Kijun-sen (Base Line)
        kijun  = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2
        # Senkou Span A — plotted 26 periods ahead, so we look back 26 from now
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
        # Senkou Span B
        senkou_b = ((high.rolling(senkou_p).max() + low.rolling(senkou_p).min()) / 2).shift(kijun_p)

        price   = _last(close)
        tk      = _last(tenkan)
        kj      = _last(kijun)
        sa      = _last(senkou_a)
        sb      = _last(senkou_b)

        # Kumo twist detection: A crossing B in the near future window
        sa_prev = _prev(senkou_a)
        sb_prev = _prev(senkou_b)
        kumo_twist = (sa_prev < sb_prev) != (sa < sb)   # sign flip = twist

        if any(np.isnan(v) for v in [price, tk, kj, sa, sb]):
            return _neutral(name, "NaN in Ichimoku — insufficient history")

        cloud_top    = max(sa, sb)
        cloud_bottom = min(sa, sb)

        values = {
            "price": price, "tenkan": tk, "kijun": kj,
            "senkou_a": sa, "senkou_b": sb,
            "cloud_top": cloud_top, "cloud_bottom": cloud_bottom,
            "kumo_twist": int(kumo_twist),
        }

        if kumo_twist:
            return IndicatorResult(
                name=name, signal=NEUTRAL, value=price, values=values,
                reason="Kumo twist detected — trend change imminent, wait"
            )

        if price > cloud_top:
            sig    = BULLISH
            reason = f"Price({price:.2f}) above cloud ({cloud_bottom:.2f}–{cloud_top:.2f})"
            if tk > kj:
                reason += " + TK bullish cross"
        elif price < cloud_bottom:
            sig    = BEARISH
            reason = f"Price({price:.2f}) below cloud ({cloud_bottom:.2f}–{cloud_top:.2f})"
            if tk < kj:
                reason += " + TK bearish cross"
        else:
            sig    = NEUTRAL
            reason = f"Price({price:.2f}) inside cloud — indecision zone"

        return IndicatorResult(name=name, signal=sig, value=price, values=values, reason=reason)

    except Exception as exc:
        logger.exception("Ichimoku calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 4 — RSI
# ─────────────────────────────────────────────────────────────────────────────

def calc_rsi(df: pd.DataFrame) -> IndicatorResult:
    """
    RSI 14 with divergence detection.

    Standard signals:
      RSI < 30  → BULLISH (oversold, bounce expected)
      RSI > 70  → BEARISH (overbought, pullback expected)
      30–70     → NEUTRAL (mid-range, no edge)

    Divergence (weakens or reverses the standard signal):
      Bearish divergence: price makes higher high but RSI makes lower high
        → even if RSI < 70, warns of exhaustion → output NEUTRAL
      Bullish divergence: price makes lower low but RSI makes higher low
        → even if RSI > 30, suggests reversal → output BULLISH

    Divergence is detected over the last 5 candles.
    """
    name   = "RSI"
    period = Config.RSI_PERIOD

    if len(df) < period + 5:
        return _neutral(name, f"Need {period+5}+ candles for RSI")

    try:
        close = df["close"]
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        # Wilder's smoothing (equivalent to EWM with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        rsi_now  = _last(rsi)
        rsi_prev = _prev(rsi)
        price_now  = _last(close)
        price_prev = _prev(close)

        if np.isnan(rsi_now):
            return _neutral(name, "RSI returned NaN")

        # Divergence detection over last 5 candles
        window = 5
        if len(df) >= period + window:
            recent_close = close.iloc[-window:]
            recent_rsi   = rsi.iloc[-window:].dropna()

            bearish_div = (
                recent_close.iloc[-1] > recent_close.iloc[0] and
                len(recent_rsi) >= 2 and
                recent_rsi.iloc[-1] < recent_rsi.iloc[0]
            )
            bullish_div = (
                recent_close.iloc[-1] < recent_close.iloc[0] and
                len(recent_rsi) >= 2 and
                recent_rsi.iloc[-1] > recent_rsi.iloc[0]
            )
        else:
            bearish_div = bullish_div = False

        values = {
            "rsi": rsi_now, "rsi_prev": rsi_prev,
            "bullish_divergence": int(bullish_div),
            "bearish_divergence": int(bearish_div),
        }

        if bullish_div and rsi_now < 50:
            return IndicatorResult(
                name=name, signal=BULLISH, value=rsi_now, values=values,
                reason=f"RSI={rsi_now:.1f} — bullish divergence (price lower, RSI higher)"
            )

        if bearish_div and rsi_now > 50:
            return IndicatorResult(
                name=name, signal=NEUTRAL, value=rsi_now, values=values,
                reason=f"RSI={rsi_now:.1f} — bearish divergence warning"
            )

        if rsi_now < Config.RSI_OVERSOLD:
            return IndicatorResult(
                name=name, signal=BULLISH, value=rsi_now, values=values,
                reason=f"RSI={rsi_now:.1f} < {Config.RSI_OVERSOLD} (oversold)"
            )
        if rsi_now > Config.RSI_OVERBOUGHT:
            return IndicatorResult(
                name=name, signal=BEARISH, value=rsi_now, values=values,
                reason=f"RSI={rsi_now:.1f} > {Config.RSI_OVERBOUGHT} (overbought)"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=rsi_now, values=values,
            reason=f"RSI={rsi_now:.1f} — mid-range, no directional edge"
        )

    except Exception as exc:
        logger.exception("RSI calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 5 — MACD
# ─────────────────────────────────────────────────────────────────────────────

def calc_macd(df: pd.DataFrame) -> IndicatorResult:
    """
    MACD (12, 26, 9) with crossover and histogram momentum.

    BULLISH:
      - MACD line crosses above signal line (fresh bullish crossover)
      - OR MACD line above signal AND histogram increasing (momentum)

    BEARISH:
      - MACD line crosses below signal line
      - OR MACD line below signal AND histogram decreasing

    NEUTRAL:
      - Lines just crossed (crossover bar itself — confirmation needed)
      - Histogram flat
    """
    name = "MACD"
    fast = Config.MACD_FAST
    slow = Config.MACD_SLOW
    sig  = Config.MACD_SIGNAL

    if len(df) < slow + sig + 5:
        return _neutral(name, f"Need {slow+sig+5}+ candles for MACD")

    try:
        close     = df["close"]
        ema_fast  = close.ewm(span=fast, adjust=False).mean()
        ema_slow  = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        sig_line  = macd_line.ewm(span=sig, adjust=False).mean()
        histogram = macd_line - sig_line

        macd_now  = _last(macd_line)
        sig_now   = _last(sig_line)
        hist_now  = _last(histogram)
        hist_prev = _prev(histogram)
        macd_prev = _prev(macd_line)
        sig_prev  = _prev(sig_line)

        if any(np.isnan(v) for v in [macd_now, sig_now, hist_now]):
            return _neutral(name, "MACD returned NaN")

        # Crossover detection: sign changed between prev and current bar
        bullish_cross = (macd_prev <= sig_prev) and (macd_now > sig_now)
        bearish_cross = (macd_prev >= sig_prev) and (macd_now < sig_now)
        hist_rising   = (not np.isnan(hist_prev)) and (hist_now > hist_prev)
        hist_falling  = (not np.isnan(hist_prev)) and (hist_now < hist_prev)

        values = {
            "macd": macd_now, "signal": sig_now, "histogram": hist_now,
            "bullish_cross": int(bullish_cross), "bearish_cross": int(bearish_cross),
        }

        if bullish_cross:
            return IndicatorResult(
                name=name, signal=BULLISH, value=macd_now, values=values,
                reason=f"MACD bullish crossover (MACD={macd_now:.4f} crossed above signal={sig_now:.4f})"
            )
        if bearish_cross:
            return IndicatorResult(
                name=name, signal=BEARISH, value=macd_now, values=values,
                reason=f"MACD bearish crossover (MACD={macd_now:.4f} crossed below signal={sig_now:.4f})"
            )
        if macd_now > sig_now and hist_rising:
            return IndicatorResult(
                name=name, signal=BULLISH, value=macd_now, values=values,
                reason=f"MACD above signal & histogram expanding bullishly"
            )
        if macd_now < sig_now and hist_falling:
            return IndicatorResult(
                name=name, signal=BEARISH, value=macd_now, values=values,
                reason=f"MACD below signal & histogram expanding bearishly"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=macd_now, values=values,
            reason=f"MACD={macd_now:.4f} — no clear momentum"
        )

    except Exception as exc:
        logger.exception("MACD calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 6 — WILLIAMS %R  (replaces Stochastic)
# ─────────────────────────────────────────────────────────────────────────────

def calc_williams_r(df: pd.DataFrame) -> IndicatorResult:
    """
    Williams %R (14-period).

    Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) × -100

    Signals:
      %R > -20  → BEARISH (overbought, reversal expected)
      %R < -80  → BULLISH (oversold, bounce expected)
      -80 to -20 → NEUTRAL (mid-range)

    Williams %R is the inverse of the Fast Stochastic and reacts faster
    to price changes, making it better suited for Gold's volatile moves.
    """
    name   = "Williams %R"
    period = 14

    if len(df) < period + 5:
        return _neutral(name, f"Need {period+5}+ candles for Williams %R")

    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        highest_high = high.rolling(period).max()
        lowest_low   = low.rolling(period).min()
        denom        = (highest_high - lowest_low).replace(0, np.nan)

        wr = ((highest_high - close) / denom) * -100

        wr_now  = _last(wr)
        wr_prev = _prev(wr)

        if np.isnan(wr_now):
            return _neutral(name, "Williams %R returned NaN")

        values = {"williams_r": wr_now, "williams_r_prev": wr_prev}

        if wr_now > -20:
            return IndicatorResult(
                name=name, signal=BEARISH, value=wr_now, values=values,
                reason=f"%R={wr_now:.1f} > -20 (overbought)"
            )
        if wr_now < -80:
            return IndicatorResult(
                name=name, signal=BULLISH, value=wr_now, values=values,
                reason=f"%R={wr_now:.1f} < -80 (oversold)"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=wr_now, values=values,
            reason=f"%R={wr_now:.1f} — mid-range"
        )

    except Exception as exc:
        logger.exception("Williams %%R calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 7 — CCI
# ─────────────────────────────────────────────────────────────────────────────

def calc_cci(df: pd.DataFrame) -> IndicatorResult:
    """
    Commodity Channel Index (20 period).

    CCI < -100 → BULLISH (oversold, bounce expected)
    CCI > +100 → BEARISH (overbought, reversal expected)
    -100 to +100 → NEUTRAL

    CCI is particularly useful for Gold because commodities frequently
    overshoot traditional RSI levels during news-driven moves.
    """
    name   = "CCI"
    period = Config.CCI_PERIOD

    if len(df) < period + 5:
        return _neutral(name, f"Need {period+5}+ candles for CCI")

    try:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_mean = typical_price.rolling(period).mean()
        # Mean absolute deviation (pandas doesn't have a rolling MAD — compute manually)
        tp_mad  = typical_price.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci = (typical_price - tp_mean) / (0.015 * tp_mad.replace(0, np.nan))

        cci_now  = _last(cci)
        cci_prev = _prev(cci)

        if np.isnan(cci_now):
            return _neutral(name, "CCI returned NaN")

        values = {"cci": cci_now, "cci_prev": cci_prev}

        if cci_now < Config.CCI_OVERSOLD:
            return IndicatorResult(
                name=name, signal=BULLISH, value=cci_now, values=values,
                reason=f"CCI={cci_now:.1f} < {Config.CCI_OVERSOLD} (oversold)"
            )
        if cci_now > Config.CCI_OVERBOUGHT:
            return IndicatorResult(
                name=name, signal=BEARISH, value=cci_now, values=values,
                reason=f"CCI={cci_now:.1f} > {Config.CCI_OVERBOUGHT} (overbought)"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=cci_now, values=values,
            reason=f"CCI={cci_now:.1f} — within normal range"
        )

    except Exception as exc:
        logger.exception("CCI calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 8 — BOLLINGER BANDS
# ─────────────────────────────────────────────────────────────────────────────

def calc_bbands(df: pd.DataFrame) -> IndicatorResult:
    """
    Bollinger Bands (20, 2) with squeeze detection.

    BULLISH:  Price touches or breaches lower band AND RSI < 50
              (oversold + statistical extreme = high probability bounce)
    BEARISH:  Price touches or breaches upper band AND RSI > 50
    NEUTRAL:  Price in mid-band zone
    SQUEEZE:  Band width is at 6-month low → breakout imminent (NEUTRAL,
              noted in reason so dashboard can flag it)

    Squeeze is detected by comparing current bandwidth to its 120-period
    rolling minimum. If current width ≤ 110% of that minimum → squeeze.
    """
    name   = "Bollinger Bands"
    period = Config.BB_PERIOD
    std    = Config.BB_STDDEV

    if len(df) < period + 5:
        return _neutral(name, f"Need {period+5}+ candles for Bollinger Bands")

    try:
        close  = df["close"]
        basis  = close.rolling(period).mean()
        stddev = close.rolling(period).std()
        upper  = basis + std * stddev
        lower  = basis - std * stddev
        width  = (upper - lower) / basis.replace(0, np.nan)   # normalised bandwidth

        price  = _last(close)
        upper_ = _last(upper)
        lower_ = _last(lower)
        basis_ = _last(basis)
        width_ = _last(width)

        # Squeeze: current width near its rolling minimum (120 bars back)
        width_min = width.rolling(min(120, len(width))).min()
        squeeze   = (_last(width_min) > 0) and (width_ <= _last(width_min) * 1.1)

        if any(np.isnan(v) for v in [price, upper_, lower_, basis_]):
            return _neutral(name, "Bollinger Bands returned NaN")

        # RSI for confirmation (fast recalc — avoids circular import)
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        avg_loss = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        rsi_val  = _last(100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan))))

        values = {
            "price": price, "upper": upper_, "lower": lower_,
            "basis": basis_, "bandwidth": width_,
            "squeeze": int(squeeze), "rsi_confirm": rsi_val,
        }

        squeeze_note = " [SQUEEZE — breakout pending]" if squeeze else ""

        # Touch = within 0.1% of band
        near_lower = price <= lower_ * 1.001
        near_upper = price >= upper_ * 0.999

        if near_lower and (np.isnan(rsi_val) or rsi_val < 55):
            return IndicatorResult(
                name=name, signal=BULLISH, value=price, values=values,
                reason=f"Price({price:.2f}) at lower band({lower_:.2f}){squeeze_note}"
            )
        if near_upper and (np.isnan(rsi_val) or rsi_val > 45):
            return IndicatorResult(
                name=name, signal=BEARISH, value=price, values=values,
                reason=f"Price({price:.2f}) at upper band({upper_:.2f}){squeeze_note}"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=price, values=values,
            reason=f"Price({price:.2f}) mid-band | BB({lower_:.2f}–{upper_:.2f}){squeeze_note}"
        )

    except Exception as exc:
        logger.exception("Bollinger Bands calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 9 — SUPERTREND
# ─────────────────────────────────────────────────────────────────────────────

def calc_supertrend(df: pd.DataFrame) -> IndicatorResult:
    """
    Supertrend (period=10, multiplier=3.0) via pandas_ta.

    BULLISH: price above the Supertrend line (uptrend)
    BEARISH: price below the Supertrend line (downtrend)
    NEUTRAL: insufficient data or pandas_ta unavailable
    """
    name = "Supertrend"
    period = 10
    multiplier = 3.0

    if len(df) < period + 10:
        return _neutral(name, f"Need {period+10}+ candles for Supertrend")

    if not PANDAS_TA_AVAILABLE:
        return _neutral(name, "pandas_ta not installed")

    try:
        st = ta.supertrend(df["high"], df["low"], df["close"],
                           length=period, multiplier=multiplier)
        if st is None or st.empty:
            return _neutral(name, "Supertrend returned empty result")

        # pandas_ta returns columns: SUPERT_10_3.0, SUPERTd_10_3.0, ...
        dir_col = [c for c in st.columns if c.startswith("SUPERTd_")]
        val_col = [c for c in st.columns if c.startswith("SUPERT_") and not c.startswith("SUPERTd_") and not c.startswith("SUPERTl_") and not c.startswith("SUPERTs_")]

        if not dir_col or not val_col:
            return _neutral(name, "Unexpected Supertrend column names")

        direction = _last(st[dir_col[0]])
        st_value  = _last(st[val_col[0]])
        price     = _last(df["close"])

        if any(np.isnan(v) for v in [direction, st_value, price]):
            return _neutral(name, "Supertrend returned NaN")

        values = {"supertrend": st_value, "direction": direction, "price": price}

        # pandas_ta: direction = 1 → bullish (price above ST), -1 → bearish
        if direction > 0:
            return IndicatorResult(
                name=name, signal=BULLISH, value=st_value, values=values,
                reason=f"Price({price:.2f}) above Supertrend({st_value:.2f}) — uptrend"
            )
        else:
            return IndicatorResult(
                name=name, signal=BEARISH, value=st_value, values=values,
                reason=f"Price({price:.2f}) below Supertrend({st_value:.2f}) — downtrend"
            )

    except Exception as exc:
        logger.exception("Supertrend calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 10 — CONNORS RSI
# ─────────────────────────────────────────────────────────────────────────────

def calc_connors_rsi(df: pd.DataFrame) -> IndicatorResult:
    """
    Connors RSI — composite momentum oscillator.

    Components:
      1. RSI(close, 3)  — short-term RSI
      2. RSI(streak, 2) — RSI of consecutive up/down day streak
      3. PercentRank(ROC(1), 100) — percentile rank of 1-bar price change

    CRSI = average of the three components (0–100 scale).

    Signals:
      CRSI > 70 → BEARISH (overbought)
      CRSI < 30 → BULLISH (oversold)
      30–70     → NEUTRAL
    """
    name = "Connors RSI"

    if len(df) < 105:
        return _neutral(name, "Need 105+ candles for Connors RSI")

    try:
        close = df["close"]

        # Component 1: RSI(3)
        delta = close.diff()
        gain3 = delta.clip(lower=0).ewm(alpha=1/3, adjust=False).mean()
        loss3 = (-delta).clip(lower=0).ewm(alpha=1/3, adjust=False).mean()
        rsi3 = 100 - (100 / (1 + gain3 / loss3.replace(0, np.nan)))

        # Component 2: Streak RSI(2)
        # Vectorized via raw numpy arrays — avoids pandas iloc overhead in tight loop
        changes_arr = close.diff().to_numpy()
        streak_arr = np.zeros(len(changes_arr))
        for i in range(1, len(changes_arr)):
            c = changes_arr[i]
            if c > 0:
                streak_arr[i] = streak_arr[i - 1] + 1 if streak_arr[i - 1] >= 0 else 1
            elif c < 0:
                streak_arr[i] = streak_arr[i - 1] - 1 if streak_arr[i - 1] <= 0 else -1
            # else: streak_arr[i] stays 0
        streak = pd.Series(streak_arr, index=close.index)

        streak_delta = streak.diff()
        sg = streak_delta.clip(lower=0).ewm(alpha=1/2, adjust=False).mean()
        sl = (-streak_delta).clip(lower=0).ewm(alpha=1/2, adjust=False).mean()
        rsi_streak = 100 - (100 / (1 + sg / sl.replace(0, np.nan)))

        # Component 3: Percentile rank of 1-bar ROC over 100 periods
        # raw=True passes numpy array — ~10x faster than raw=False (no pandas overhead)
        roc1 = close.pct_change(1)
        pct_rank = roc1.rolling(100).apply(
            lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100
            if len(x) > 1 else 50.0, raw=True
        )

        # Connors RSI = average of the 3 components
        crsi = (rsi3 + rsi_streak + pct_rank) / 3
        crsi_now  = _last(crsi)
        crsi_prev = _prev(crsi)

        if np.isnan(crsi_now):
            return _neutral(name, "Connors RSI returned NaN")

        values = {
            "connors_rsi": crsi_now, "connors_rsi_prev": crsi_prev,
            "rsi3": _last(rsi3), "rsi_streak": _last(rsi_streak),
            "pct_rank": _last(pct_rank),
        }

        if crsi_now > 70:
            return IndicatorResult(
                name=name, signal=BEARISH, value=crsi_now, values=values,
                reason=f"CRSI={crsi_now:.1f} > 70 (overbought)"
            )
        if crsi_now < 30:
            return IndicatorResult(
                name=name, signal=BULLISH, value=crsi_now, values=values,
                reason=f"CRSI={crsi_now:.1f} < 30 (oversold)"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=crsi_now, values=values,
            reason=f"CRSI={crsi_now:.1f} — mid-range"
        )

    except Exception as exc:
        logger.exception("Connors RSI calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 11 — KELTNER CHANNELS (squeeze detector)
# ─────────────────────────────────────────────────────────────────────────────

def calc_keltner(df: pd.DataFrame) -> IndicatorResult:
    """
    Keltner Channels (EMA-20, 2.0× ATR-20) used as a squeeze detector
    in combination with Bollinger Bands.

    Squeeze detection:
      BB upper < Keltner upper AND BB lower > Keltner lower
      → Bollinger Bands are inside Keltner = volatility squeeze = NEUTRAL

    Breakout confirmation:
      BB outside Keltner → expanding volatility → confirm direction:
        - Price above Keltner upper → BULLISH breakout
        - Price below Keltner lower → BEARISH breakout
        - Price inside channels → NEUTRAL (no breakout)
    """
    name   = "Keltner Channels"
    period = 20
    kc_mult = 2.0
    bb_std  = Config.BB_STDDEV

    if len(df) < period + 10:
        return _neutral(name, f"Need {period+10}+ candles for Keltner Channels")

    try:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # Keltner Channels: EMA ± multiplier × ATR
        kc_mid = close.ewm(span=period, adjust=False).mean()
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_line   = tr.ewm(alpha=1/period, adjust=False).mean()
        kc_upper   = kc_mid + kc_mult * atr_line
        kc_lower   = kc_mid - kc_mult * atr_line

        # Bollinger Bands for squeeze comparison
        bb_mid    = close.rolling(period).mean()
        bb_stddev = close.rolling(period).std()
        bb_upper  = bb_mid + bb_std * bb_stddev
        bb_lower  = bb_mid - bb_std * bb_stddev

        price     = _last(close)
        kc_up     = _last(kc_upper)
        kc_lo     = _last(kc_lower)
        kc_m      = _last(kc_mid)
        bb_up     = _last(bb_upper)
        bb_lo     = _last(bb_lower)

        if any(np.isnan(v) for v in [price, kc_up, kc_lo, bb_up, bb_lo]):
            return _neutral(name, "Keltner/BB returned NaN")

        squeeze = (bb_up < kc_up) and (bb_lo > kc_lo)

        values = {
            "kc_upper": kc_up, "kc_lower": kc_lo, "kc_mid": kc_m,
            "bb_upper": bb_up, "bb_lower": bb_lo,
            "squeeze": int(squeeze), "price": price,
        }

        if squeeze:
            return IndicatorResult(
                name=name, signal=NEUTRAL, value=price, values=values,
                reason=f"BB inside Keltner — volatility squeeze, breakout pending"
            )

        # No squeeze — check for breakout direction
        if price > kc_up:
            return IndicatorResult(
                name=name, signal=BULLISH, value=price, values=values,
                reason=f"Price({price:.2f}) above Keltner upper({kc_up:.2f}) — bullish breakout"
            )
        if price < kc_lo:
            return IndicatorResult(
                name=name, signal=BEARISH, value=price, values=values,
                reason=f"Price({price:.2f}) below Keltner lower({kc_lo:.2f}) — bearish breakout"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=price, values=values,
            reason=f"Price({price:.2f}) inside Keltner — no breakout"
        )

    except Exception as exc:
        logger.exception("Keltner Channels calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 12 — ATR (volatility context, not directional)
# ─────────────────────────────────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame) -> IndicatorResult:
    """
    ATR 14 — Average True Range.

    Always returns NEUTRAL signal — ATR is not a directional indicator.
    Its value is used exclusively by risk_manager.py to set SL/TP levels.

    We also compute ATR as a % of price (normalised ATR) which is useful
    for the ML feature vector regardless of Gold's absolute price level.
    """
    name   = "ATR"
    period = Config.ATR_PERIOD

    if len(df) < period + 2:
        return _neutral(name, f"Need {period+2}+ candles for ATR")

    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_line = tr.ewm(alpha=1/period, adjust=False).mean()
        atr_now  = _last(atr_line)
        price    = _last(close)

        atr_pct  = (atr_now / price * 100) if price > 0 else np.nan
        sl_dist  = atr_now * Config.ATR_SL_MULTIPLIER

        values = {
            "atr": atr_now,
            "atr_pct": atr_pct,
            "sl_distance": sl_dist,
            "price": price,
        }

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=atr_now, values=values,
            reason=f"ATR={atr_now:.2f} ({atr_pct:.2f}% of price) | SL dist={sl_dist:.2f}"
        )

    except Exception as exc:
        logger.exception("ATR calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR 13 — VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def calc_volume(df: pd.DataFrame, overall_direction: Optional[Signal] = None) -> IndicatorResult:
    """
    Volume analysis: current bar vs 20-period average.

    Volume alone is direction-neutral. We use the *overall_direction*
    hint (from other indicators) to decide whether above-average volume
    confirms (BULLISH/BEARISH) or is suspicious (e.g. high sell volume
    during a supposed buy signal).

    Rules:
      - Above average (> 1× avg):           confirms direction (+1)
      - Surge (> VOLUME_SURGE_FACTOR× avg): strong confirmation (+2 effect
            captured by scoring engine separately)
      - Below average:                       NEUTRAL (weak confirmation)
      - No direction hint:                   NEUTRAL (can't score directionally)

    For Gold tick volume is a proxy for real volume — it's still a valid
    relative measure when comparing current vs rolling average.
    """
    name    = "Volume"
    lookback = Config.VOLUME_LOOKBACK

    if len(df) < lookback + 1:
        return _neutral(name, f"Need {lookback+1}+ candles for volume analysis")

    try:
        volume   = df["volume"]
        vol_now  = _last(volume)
        vol_avg  = _last(volume.rolling(lookback).mean())

        if np.isnan(vol_now) or np.isnan(vol_avg) or vol_avg == 0:
            return _neutral(name, "Volume data unavailable or zero average")

        ratio  = vol_now / vol_avg
        surge  = ratio >= Config.VOLUME_SURGE_FACTOR

        values = {
            "volume": vol_now, "avg_volume": vol_avg,
            "ratio": ratio, "surge": int(surge),
        }

        above_avg = ratio > 1.0

        if above_avg and overall_direction == BULLISH:
            sig    = BULLISH
            reason = f"Volume {ratio:.1f}× avg — confirms BUY{'  (SURGE)' if surge else ''}"
        elif above_avg and overall_direction == BEARISH:
            sig    = BEARISH
            reason = f"Volume {ratio:.1f}× avg — confirms SELL{'  (SURGE)' if surge else ''}"
        elif above_avg:
            # High volume but no direction context yet
            sig    = NEUTRAL
            reason = f"Volume {ratio:.1f}× avg — elevated but direction unknown"
        else:
            sig    = NEUTRAL
            reason = f"Volume {ratio:.2f}× avg — below average, weak signal"

        return IndicatorResult(name=name, signal=sig, value=ratio, values=values, reason=reason)

    except Exception as exc:
        logger.exception("Volume calculation error: %s", exc)
        return _neutral(name, f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CALCULATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_all(df: pd.DataFrame) -> AllIndicators:
    """
    Calculate all 13 indicators on the supplied DataFrame and return
    an AllIndicators object with every result populated.

    Indicators are computed in dependency order:
      - Directional indicators first (EMA/HMA, ADX, Ichimoku, RSI, MACD,
        Williams %R, CCI, Supertrend, Connors RSI, Keltner, BBands, ATR)
      - Volume last (needs direction hint from the others)

    Args:
        df: Clean, processed OHLCV DataFrame (output of processor.py)

    Returns:
        AllIndicators with all 13 results. No result will be None —
        indicators that can't compute return a NEUTRAL result with an
        explanatory reason string.
    """
    logger.debug("Calculating all indicators on %d candles…", len(df))

    if not PANDAS_TA_AVAILABLE:
        logger.warning(
            "pandas_ta not installed — Supertrend disabled. Run: pip install pandas-ta"
        )

    ema_r   = calc_ema(df)
    adx_r   = calc_adx(df)
    ich_r   = calc_ichimoku(df)
    rsi_r   = calc_rsi(df)
    macd_r  = calc_macd(df)
    wr_r    = calc_williams_r(df)
    cci_r   = calc_cci(df)
    bb_r    = calc_bbands(df)
    st_r    = calc_supertrend(df)
    crsi_r  = calc_connors_rsi(df)
    kelt_r  = calc_keltner(df)
    atr_r   = calc_atr(df)

    # Determine provisional direction from directional indicators for volume confirmation
    directional = [ema_r, adx_r, ich_r, rsi_r, macd_r, wr_r, cci_r, st_r, crsi_r, kelt_r, atr_r]
    bull_count = sum(1 for r in directional if r.signal == BULLISH)
    bear_count = sum(1 for r in directional if r.signal == BEARISH)
    if   bull_count > bear_count: provisional = BULLISH
    elif bear_count > bull_count: provisional = BEARISH
    else:                          provisional = None

    vol_r = calc_volume(df, overall_direction=provisional)

    latest_close = _last(df["close"])

    result = AllIndicators(
        ema=ema_r, adx=adx_r, ichimoku=ich_r,
        rsi=rsi_r, macd=macd_r, williams_r=wr_r,
        cci=cci_r, bbands=bb_r, supertrend=st_r,
        connors_rsi=crsi_r, keltner=kelt_r,
        atr=atr_r, volume=vol_r,
        latest_close=latest_close,
    )

    logger.info(
        "Indicators: %s | Price=%.2f",
        result.summary_line(), latest_close
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTED INDICATORS (backtest fast-path)
# ─────────────────────────────────────────────────────────────────────────────

class PrecomputedIndicators:
    """
    Precomputes ALL indicator time-series on a full DataFrame ONCE.
    Use .at(i) to reconstruct AllIndicators at bar index i in O(1).

    Speedup: avoids re-running rolling operations on overlapping 500-bar slices
    for every analysis bar during backtesting.  Running once on 48k bars takes
    ~1.5 s; running 12k times on 500-bar slices takes ~350 s.
    """

    def __init__(self, df: pd.DataFrame) -> None:  # noqa: PLR0915
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        n      = len(df)

        # ── EMA / HMA alignment ──────────────────────────────────────────────
        self._hma20  = _hma(close, Config.EMA_FAST).to_numpy()
        self._ema50  = close.ewm(span=Config.EMA_MID,  adjust=False).mean().to_numpy()
        self._ema200 = close.ewm(span=Config.EMA_SLOW, adjust=False).mean().to_numpy()
        self._close  = close.to_numpy()

        # ── ADX ──────────────────────────────────────────────────────────────
        _adx_p = Config.ADX_PERIOD
        _tr    = pd.concat([high - low,
                            (high - close.shift(1)).abs(),
                            (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        _up    = high - high.shift(1)
        _dn    = low.shift(1) - low
        _pdm   = pd.Series(np.where((_up > _dn) & (_up > 0), _up, 0.0), index=df.index)
        _mdm   = pd.Series(np.where((_dn > _up) & (_dn > 0), _dn, 0.0), index=df.index)
        _atr_s = _tr.ewm(alpha=1/_adx_p, adjust=False).mean()
        _pdi   = 100 * _pdm.ewm(alpha=1/_adx_p, adjust=False).mean() / _atr_s.replace(0, np.nan)
        _mdi   = 100 * _mdm.ewm(alpha=1/_adx_p, adjust=False).mean() / _atr_s.replace(0, np.nan)
        _dx    = 100 * (_pdi - _mdi).abs() / (_pdi + _mdi).replace(0, np.nan)
        self._adx      = _dx.ewm(alpha=1/_adx_p, adjust=False).mean().to_numpy()
        self._plus_di  = _pdi.to_numpy()
        self._minus_di = _mdi.to_numpy()

        # ── Ichimoku ─────────────────────────────────────────────────────────
        _tk_p  = Config.ICHIMOKU_TENKAN
        _kj_p  = Config.ICHIMOKU_KIJUN
        _sk_p  = Config.ICHIMOKU_SENKOU
        _tenkan  = (high.rolling(_tk_p).max() + low.rolling(_tk_p).min()) / 2
        _kijun   = (high.rolling(_kj_p).max() + low.rolling(_kj_p).min()) / 2
        _sa      = ((_tenkan + _kijun) / 2).shift(_kj_p)
        _sb      = ((high.rolling(_sk_p).max() + low.rolling(_sk_p).min()) / 2).shift(_kj_p)
        self._tenkan   = _tenkan.to_numpy()
        self._kijun    = _kijun.to_numpy()
        self._senkou_a = _sa.to_numpy()
        self._senkou_b = _sb.to_numpy()

        # ── RSI (Wilder, 14) ─────────────────────────────────────────────────
        _rsi_p   = Config.RSI_PERIOD
        _delta   = close.diff()
        _ag      = _delta.clip(lower=0).ewm(alpha=1/_rsi_p, adjust=False).mean()
        _al      = (-_delta).clip(lower=0).ewm(alpha=1/_rsi_p, adjust=False).mean()
        self._rsi = (100 - (100 / (1 + _ag / _al.replace(0, np.nan)))).to_numpy()

        # ── MACD ─────────────────────────────────────────────────────────────
        _ef       = close.ewm(span=Config.MACD_FAST,   adjust=False).mean()
        _es       = close.ewm(span=Config.MACD_SLOW,   adjust=False).mean()
        _ml       = _ef - _es
        _sl       = _ml.ewm(span=Config.MACD_SIGNAL,   adjust=False).mean()
        self._macd      = _ml.to_numpy()
        self._macd_sig  = _sl.to_numpy()
        self._macd_hist = (_ml - _sl).to_numpy()

        # ── Williams %R ──────────────────────────────────────────────────────
        _wr_p   = 14
        _hh     = high.rolling(_wr_p).max()
        _ll     = low.rolling(_wr_p).min()
        self._wr = ((_hh - close) / (_hh - _ll).replace(0, np.nan) * -100).to_numpy()

        # ── CCI ──────────────────────────────────────────────────────────────
        _cci_p  = Config.CCI_PERIOD
        _tp     = (high + low + close) / 3
        _tp_m   = _tp.rolling(_cci_p).mean()
        _tp_mad = _tp.rolling(_cci_p).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        self._cci = ((_tp - _tp_m) / (0.015 * _tp_mad.replace(0, np.nan))).to_numpy()

        # ── Bollinger Bands ───────────────────────────────────────────────────
        _bb_p      = Config.BB_PERIOD
        _bb_std    = Config.BB_STDDEV
        _bb_basis  = close.rolling(_bb_p).mean()
        _bb_sd     = close.rolling(_bb_p).std()
        _bb_up     = _bb_basis + _bb_std * _bb_sd
        _bb_lo     = _bb_basis - _bb_std * _bb_sd
        _bb_w      = (_bb_up - _bb_lo) / _bb_basis.replace(0, np.nan)
        self._bb_upper    = _bb_up.to_numpy()
        self._bb_lower    = _bb_lo.to_numpy()
        self._bb_width    = _bb_w.to_numpy()
        self._bb_wmin     = _bb_w.rolling(min(120, n)).min().to_numpy()
        self._rsi_for_bb  = self._rsi   # reuse main RSI array

        # ── Supertrend ───────────────────────────────────────────────────────
        self._st_dir = self._st_val = None
        if PANDAS_TA_AVAILABLE:
            try:
                _st = ta.supertrend(high, low, close, length=10, multiplier=3.0)
                if _st is not None and not _st.empty:
                    _dc = [c for c in _st.columns if c.startswith("SUPERTd_")]
                    _vc = [c for c in _st.columns
                           if c.startswith("SUPERT_")
                           and not c.startswith("SUPERTd_")
                           and not c.startswith("SUPERTl_")
                           and not c.startswith("SUPERTs_")]
                    if _dc and _vc:
                        self._st_dir = _st[_dc[0]].to_numpy()
                        self._st_val = _st[_vc[0]].to_numpy()
            except Exception:
                pass

        # ── Connors RSI ───────────────────────────────────────────────────────
        _g3  = _delta.clip(lower=0).ewm(alpha=1/3, adjust=False).mean()
        _l3  = (-_delta).clip(lower=0).ewm(alpha=1/3, adjust=False).mean()
        _r3  = 100 - (100 / (1 + _g3 / _l3.replace(0, np.nan)))
        _carr = close.diff().to_numpy()
        _sarr = np.zeros(n)
        for _j in range(1, n):
            _c = _carr[_j]
            if _c > 0:
                _sarr[_j] = _sarr[_j-1] + 1 if _sarr[_j-1] >= 0 else 1
            elif _c < 0:
                _sarr[_j] = _sarr[_j-1] - 1 if _sarr[_j-1] <= 0 else -1
        _streak = pd.Series(_sarr, index=close.index)
        _sd  = _streak.diff()
        _sg  = _sd.clip(lower=0).ewm(alpha=1/2, adjust=False).mean()
        _sl2 = (-_sd).clip(lower=0).ewm(alpha=1/2, adjust=False).mean()
        _rs2 = 100 - (100 / (1 + _sg / _sl2.replace(0, np.nan)))
        _roc = close.pct_change(1)
        _pr  = _roc.rolling(100).apply(
            lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100
            if len(x) > 1 else 50.0, raw=True
        )
        self._crsi = ((_r3 + _rs2 + _pr) / 3).to_numpy()

        # ── Keltner Channels ─────────────────────────────────────────────────
        _kc_p   = 20
        _kc_m_  = close.ewm(span=_kc_p, adjust=False).mean()
        _tr_kc  = pd.concat([high - low,
                              (high - close.shift(1)).abs(),
                              (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        _atr_kc = _tr_kc.ewm(alpha=1/_kc_p, adjust=False).mean()
        self._kc_upper = (_kc_m_ + 2.0 * _atr_kc).to_numpy()
        self._kc_lower = (_kc_m_ - 2.0 * _atr_kc).to_numpy()
        self._kc_mid   = _kc_m_.to_numpy()

        # ── ATR (14-period Wilder) ────────────────────────────────────────────
        _tr_atr = pd.concat([high - low,
                              (high - close.shift(1)).abs(),
                              (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        self._atr = _tr_atr.ewm(alpha=1/Config.ATR_PERIOD, adjust=False).mean().to_numpy()

        # ── Volume ───────────────────────────────────────────────────────────
        self._vol     = volume.to_numpy(dtype=float)
        self._vol_avg = volume.rolling(Config.VOLUME_LOOKBACK).mean().to_numpy()

    # ─────────────────────────────────────────────────────────────────────────
    def at(self, i: int) -> "AllIndicators":
        """Return AllIndicators for bar index i (O(1) array lookups)."""
        cl = float(self._close[i])
        ip = i - 1  # previous bar index (safe because we guard with NaN checks)

        # ── EMA / HMA ────────────────────────────────────────────────────────
        h20, e50, e200 = float(self._hma20[i]), float(self._ema50[i]), float(self._ema200[i])
        if any(np.isnan(v) for v in (h20, e50, e200)):
            ema_r = _neutral("HMA/EMA Alignment")
        else:
            _ev = {"price": cl, "hma20": h20, "ema50": e50, "ema200": e200}
            if cl > h20 > e50 > e200:
                ema_r = IndicatorResult("HMA/EMA Alignment", BULLISH, h20, _ev)
            elif cl < h20 < e50 < e200:
                ema_r = IndicatorResult("HMA/EMA Alignment", BEARISH, h20, _ev)
            else:
                ema_r = IndicatorResult("HMA/EMA Alignment", NEUTRAL, h20, _ev)

        # ── ADX ──────────────────────────────────────────────────────────────
        adxv, pdiv, mdiv = float(self._adx[i]), float(self._plus_di[i]), float(self._minus_di[i])
        if any(np.isnan(v) for v in (adxv, pdiv, mdiv)):
            adx_r = _neutral("ADX")
        else:
            _av = {"adx": adxv, "+di": pdiv, "-di": mdiv}
            if adxv < Config.ADX_TREND_THRESHOLD:
                adx_r = IndicatorResult("ADX", NEUTRAL, adxv, _av)
            elif pdiv > mdiv:
                adx_r = IndicatorResult("ADX", BULLISH, adxv, _av)
            else:
                adx_r = IndicatorResult("ADX", BEARISH, adxv, _av)

        # ── Ichimoku ─────────────────────────────────────────────────────────
        tk, kj = float(self._tenkan[i]), float(self._kijun[i])
        sa, sb = float(self._senkou_a[i]), float(self._senkou_b[i])
        if any(np.isnan(v) for v in (tk, kj, sa, sb)):
            ich_r = _neutral("Ichimoku")
        else:
            sa_p = float(self._senkou_a[ip]) if ip >= 0 else np.nan
            sb_p = float(self._senkou_b[ip]) if ip >= 0 else np.nan
            twist = (not np.isnan(sa_p) and not np.isnan(sb_p) and
                     ((sa_p < sb_p) != (sa < sb)))
            ct, cb = max(sa, sb), min(sa, sb)
            _iv = {"price": cl, "tenkan": tk, "kijun": kj, "senkou_a": sa, "senkou_b": sb,
                   "cloud_top": ct, "cloud_bottom": cb, "kumo_twist": int(twist)}
            if twist:
                ich_r = IndicatorResult("Ichimoku", NEUTRAL, cl, _iv)
            elif cl > ct:
                ich_r = IndicatorResult("Ichimoku", BULLISH, cl, _iv)
            elif cl < cb:
                ich_r = IndicatorResult("Ichimoku", BEARISH, cl, _iv)
            else:
                ich_r = IndicatorResult("Ichimoku", NEUTRAL, cl, _iv)

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi_v = float(self._rsi[i])
        if np.isnan(rsi_v):
            rsi_r = _neutral("RSI")
        else:
            rsi_p = float(self._rsi[ip]) if ip >= 0 else np.nan
            bull_div = bear_div = False
            if i >= 4:
                rc = self._close[i-4:i+1]
                rr = self._rsi[i-4:i+1]
                if not np.any(np.isnan(rr)):
                    bear_div = bool(rc[-1] > rc[0] and rr[-1] < rr[0])
                    bull_div = bool(rc[-1] < rc[0] and rr[-1] > rr[0])
            _rv = {"rsi": rsi_v, "rsi_prev": rsi_p,
                   "bullish_divergence": int(bull_div), "bearish_divergence": int(bear_div)}
            if bull_div and rsi_v < 50:
                rsi_r = IndicatorResult("RSI", BULLISH, rsi_v, _rv)
            elif bear_div and rsi_v > 50:
                rsi_r = IndicatorResult("RSI", NEUTRAL, rsi_v, _rv)
            elif rsi_v < Config.RSI_OVERSOLD:
                rsi_r = IndicatorResult("RSI", BULLISH, rsi_v, _rv)
            elif rsi_v > Config.RSI_OVERBOUGHT:
                rsi_r = IndicatorResult("RSI", BEARISH, rsi_v, _rv)
            else:
                rsi_r = IndicatorResult("RSI", NEUTRAL, rsi_v, _rv)

        # ── MACD ─────────────────────────────────────────────────────────────
        mv, sv, hv = float(self._macd[i]), float(self._macd_sig[i]), float(self._macd_hist[i])
        if any(np.isnan(v) for v in (mv, sv, hv)):
            macd_r = _neutral("MACD")
        else:
            mp = float(self._macd[ip])     if ip >= 0 else np.nan
            sp = float(self._macd_sig[ip]) if ip >= 0 else np.nan
            hp = float(self._macd_hist[ip])if ip >= 0 else np.nan
            bc = (not np.isnan(mp)) and mp <= sp and mv > sv
            brc= (not np.isnan(mp)) and mp >= sp and mv < sv
            hr = (not np.isnan(hp)) and hv > hp
            hf = (not np.isnan(hp)) and hv < hp
            _mv = {"macd": mv, "signal": sv, "histogram": hv,
                   "bullish_cross": int(bc), "bearish_cross": int(brc)}
            if bc:
                macd_r = IndicatorResult("MACD", BULLISH, mv, _mv)
            elif brc:
                macd_r = IndicatorResult("MACD", BEARISH, mv, _mv)
            elif mv > sv and hr:
                macd_r = IndicatorResult("MACD", BULLISH, mv, _mv)
            elif mv < sv and hf:
                macd_r = IndicatorResult("MACD", BEARISH, mv, _mv)
            else:
                macd_r = IndicatorResult("MACD", NEUTRAL, mv, _mv)

        # ── Williams %R ──────────────────────────────────────────────────────
        wr_v = float(self._wr[i])
        if np.isnan(wr_v):
            wr_r = _neutral("Williams %R")
        else:
            wr_p = float(self._wr[ip]) if ip >= 0 else np.nan
            _wrv = {"williams_r": wr_v, "williams_r_prev": wr_p}
            if wr_v > -20:
                wr_r = IndicatorResult("Williams %R", BEARISH, wr_v, _wrv)
            elif wr_v < -80:
                wr_r = IndicatorResult("Williams %R", BULLISH, wr_v, _wrv)
            else:
                wr_r = IndicatorResult("Williams %R", NEUTRAL, wr_v, _wrv)

        # ── CCI ──────────────────────────────────────────────────────────────
        cci_v = float(self._cci[i])
        if np.isnan(cci_v):
            cci_r = _neutral("CCI")
        else:
            cci_p = float(self._cci[ip]) if ip >= 0 else np.nan
            _ccv  = {"cci": cci_v, "cci_prev": cci_p}
            if cci_v < Config.CCI_OVERSOLD:
                cci_r = IndicatorResult("CCI", BULLISH, cci_v, _ccv)
            elif cci_v > Config.CCI_OVERBOUGHT:
                cci_r = IndicatorResult("CCI", BEARISH, cci_v, _ccv)
            else:
                cci_r = IndicatorResult("CCI", NEUTRAL, cci_v, _ccv)

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bbu, bbl = float(self._bb_upper[i]), float(self._bb_lower[i])
        bbw, bbwm = float(self._bb_width[i]), float(self._bb_wmin[i])
        rsi_cb = float(self._rsi_for_bb[i])
        if any(np.isnan(v) for v in (bbu, bbl, cl)):
            bb_r = _neutral("Bollinger Bands")
        else:
            sq_b = (not np.isnan(bbwm) and bbwm > 0 and
                    not np.isnan(bbw) and bbw <= bbwm * 1.1)
            nl, nu = cl <= bbl * 1.001, cl >= bbu * 0.999
            _bbv = {"price": cl, "upper": bbu, "lower": bbl,
                    "bandwidth": bbw, "squeeze": int(sq_b), "rsi_confirm": rsi_cb}
            if nl and (np.isnan(rsi_cb) or rsi_cb < 55):
                bb_r = IndicatorResult("Bollinger Bands", BULLISH, cl, _bbv)
            elif nu and (np.isnan(rsi_cb) or rsi_cb > 45):
                bb_r = IndicatorResult("Bollinger Bands", BEARISH, cl, _bbv)
            else:
                bb_r = IndicatorResult("Bollinger Bands", NEUTRAL, cl, _bbv)

        # ── Supertrend ───────────────────────────────────────────────────────
        if self._st_dir is not None:
            std_v = float(self._st_dir[i])
            stv_v = float(self._st_val[i])
            if np.isnan(std_v) or np.isnan(stv_v):
                st_r = _neutral("Supertrend")
            else:
                _stv = {"supertrend": stv_v, "direction": std_v, "price": cl}
                st_r = (IndicatorResult("Supertrend", BULLISH, stv_v, _stv)
                        if std_v > 0 else
                        IndicatorResult("Supertrend", BEARISH, stv_v, _stv))
        else:
            st_r = _neutral("Supertrend", "pandas_ta unavailable")

        # ── Connors RSI ───────────────────────────────────────────────────────
        crsi_v = float(self._crsi[i])
        if np.isnan(crsi_v):
            crsi_r = _neutral("Connors RSI")
        else:
            crsi_p = float(self._crsi[ip]) if ip >= 0 else np.nan
            _crv   = {"connors_rsi": crsi_v, "connors_rsi_prev": crsi_p}
            if crsi_v > 70:
                crsi_r = IndicatorResult("Connors RSI", BEARISH, crsi_v, _crv)
            elif crsi_v < 30:
                crsi_r = IndicatorResult("Connors RSI", BULLISH, crsi_v, _crv)
            else:
                crsi_r = IndicatorResult("Connors RSI", NEUTRAL, crsi_v, _crv)

        # ── Keltner Channels ─────────────────────────────────────────────────
        kcu, kcl, kcm = float(self._kc_upper[i]), float(self._kc_lower[i]), float(self._kc_mid[i])
        if any(np.isnan(v) for v in (kcu, kcl, bbu, bbl)):
            kelt_r = _neutral("Keltner Channels")
        else:
            sq_k = (bbu < kcu) and (bbl > kcl)
            _kcv = {"kc_upper": kcu, "kc_lower": kcl, "kc_mid": kcm,
                    "bb_upper": bbu, "bb_lower": bbl, "squeeze": int(sq_k), "price": cl}
            if sq_k:
                kelt_r = IndicatorResult("Keltner Channels", NEUTRAL, cl, _kcv)
            elif cl > kcu:
                kelt_r = IndicatorResult("Keltner Channels", BULLISH, cl, _kcv)
            elif cl < kcl:
                kelt_r = IndicatorResult("Keltner Channels", BEARISH, cl, _kcv)
            else:
                kelt_r = IndicatorResult("Keltner Channels", NEUTRAL, cl, _kcv)

        # ── ATR ──────────────────────────────────────────────────────────────
        atr_v = float(self._atr[i])
        if np.isnan(atr_v):
            atr_r = _neutral("ATR")
        else:
            atr_pct = (atr_v / cl * 100) if cl > 0 else np.nan
            _atv = {"atr": atr_v, "atr_pct": atr_pct,
                    "sl_distance": atr_v * Config.ATR_SL_MULTIPLIER, "price": cl}
            atr_r = IndicatorResult("ATR", NEUTRAL, atr_v, _atv)

        # ── Volume ───────────────────────────────────────────────────────────
        vl, va = float(self._vol[i]), float(self._vol_avg[i])
        if np.isnan(va) or va == 0:
            vol_r = _neutral("Volume")
        else:
            ratio_v = vl / va
            surge_v = ratio_v >= Config.VOLUME_SURGE_FACTOR
            _vv = {"volume": vl, "avg_volume": va, "ratio": ratio_v, "surge": int(surge_v)}
            # Provisional direction from the strongest directional indicators
            _bull = sum(1 for r in (ema_r, adx_r, rsi_r, macd_r, st_r, crsi_r)
                        if r.signal == BULLISH)
            _bear = sum(1 for r in (ema_r, adx_r, rsi_r, macd_r, st_r, crsi_r)
                        if r.signal == BEARISH)
            _prov = BULLISH if _bull > _bear else (BEARISH if _bear > _bull else None)
            above = ratio_v > 1.0
            if above and _prov == BULLISH:
                vol_r = IndicatorResult("Volume", BULLISH, ratio_v, _vv)
            elif above and _prov == BEARISH:
                vol_r = IndicatorResult("Volume", BEARISH, ratio_v, _vv)
            else:
                vol_r = IndicatorResult("Volume", NEUTRAL, ratio_v, _vv)

        return AllIndicators(
            ema=ema_r, adx=adx_r, ichimoku=ich_r,
            rsi=rsi_r, macd=macd_r, williams_r=wr_r,
            cci=cci_r, bbands=bb_r, supertrend=st_r,
            connors_rsi=crsi_r, keltner=kelt_r,
            atr=atr_r, volume=vol_r,
            latest_close=cl,
        )
