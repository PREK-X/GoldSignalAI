"""
GoldSignalAI — analysis/indicators.py
======================================
Calculates all 10 technical indicators used in the signal scoring engine.

Each indicator returns an IndicatorResult dataclass containing:
  - signal   : "bullish" | "bearish" | "neutral"
  - value    : primary numeric value (e.g. RSI = 62.4)
  - values   : dict of all computed sub-values for display & ML features
  - reason   : human-readable explanation of the signal decision

The top-level function `calculate_all()` returns an AllIndicators
object holding all 10 results and is the only thing other modules need
to import.

Indicator → Signal mapping:
  1.  EMA Alignment       — price vs EMA20/50/200 stack
  2.  ADX                 — trend strength gate
  3.  Ichimoku Cloud      — price vs cloud + kumo twist
  4.  RSI 14              — oversold / overbought + divergence
  5.  MACD                — crossover + histogram momentum
  6.  Stochastic (14,3,3) — %K/%D crossover in oversold/overbought zones
  7.  CCI 20              — below -100 / above +100
  8.  Bollinger Bands     — band touch + squeeze detection
  9.  ATR                 — volatility context (neutral signal, value used for SL)
  10. Volume              — current vs 20-period average

Note: ATR (#9) is always "neutral" in the scoring engine because it is
a volatility measure, not a directional signal. Its value is used
exclusively for SL/TP calculation. Volume (#10) modifies confidence
rather than adding a directional vote — but it still scores +1/−1 when
it confirms or contradicts the overall direction.
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
    Holds all 10 indicator results plus convenience properties.
    Passed to scoring.py, formatter.py, and the ML feature builder.
    """
    ema:        IndicatorResult
    adx:        IndicatorResult
    ichimoku:   IndicatorResult
    rsi:        IndicatorResult
    macd:       IndicatorResult
    stochastic: IndicatorResult
    cci:        IndicatorResult
    bbands:     IndicatorResult
    atr:        IndicatorResult    # directional-neutral; value used for SL
    volume:     IndicatorResult

    # Cached latest close (set by calculate_all)
    latest_close: float = 0.0

    def as_list(self) -> list[IndicatorResult]:
        """Return the 9 voted indicators (BBands excluded — negative predictive accuracy).
        ATR is directionally neutral and also excluded from voting; bbands is kept
        on the dataclass for ML feature use only."""
        return [
            self.ema, self.adx, self.ichimoku, self.rsi,
            self.macd, self.stochastic, self.cci,
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
        return f"{b}/10 Bullish, {br}/10 Bearish"


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

def calc_ema(df: pd.DataFrame) -> IndicatorResult:
    """
    EMA Alignment: price vs EMA20 vs EMA50 vs EMA200 stack.

    BUY  signal: price > EMA20 > EMA50 > EMA200  (perfect bull stack)
    SELL signal: price < EMA20 < EMA50 < EMA200  (perfect bear stack)
    NEUTRAL: mixed / EMA200 not yet calculable

    A partial stack (e.g. price > EMA20 but EMA20 < EMA50) scores neutral
    because the trend is not clean enough for a high-confidence entry.
    """
    name = "EMA Alignment"
    close = df["close"]

    if len(df) < Config.EMA_SLOW:
        return _neutral(name, f"Need {Config.EMA_SLOW}+ candles for EMA200")

    try:
        ema20  = _last(close.ewm(span=Config.EMA_FAST,  adjust=False).mean())
        ema50  = _last(close.ewm(span=Config.EMA_MID,   adjust=False).mean())
        ema200 = _last(close.ewm(span=Config.EMA_SLOW,  adjust=False).mean())
        price  = _last(close)

        if any(np.isnan(v) for v in [price, ema20, ema50, ema200]):
            return _neutral(name, "NaN in EMA calculation")

        values = {
            "price": price, "ema20": ema20,
            "ema50": ema50, "ema200": ema200,
        }

        if price > ema20 > ema50 > ema200:
            return IndicatorResult(
                name=name, signal=BULLISH, value=ema20, values=values,
                reason=f"Price({price:.2f}) > EMA20({ema20:.2f}) > EMA50({ema50:.2f}) > EMA200({ema200:.2f})"
            )
        if price < ema20 < ema50 < ema200:
            return IndicatorResult(
                name=name, signal=BEARISH, value=ema20, values=values,
                reason=f"Price({price:.2f}) < EMA20({ema20:.2f}) < EMA50({ema50:.2f}) < EMA200({ema200:.2f})"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=ema20, values=values,
            reason="EMA stack not aligned — mixed trend"
        )

    except Exception as exc:
        logger.exception("EMA calculation error: %s", exc)
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
# INDICATOR 6 — STOCHASTIC OSCILLATOR
# ─────────────────────────────────────────────────────────────────────────────

def calc_stochastic(df: pd.DataFrame) -> IndicatorResult:
    """
    Stochastic Oscillator (14, 3, 3).

    Signal requires BOTH zone condition AND %K/%D crossover:
      BULLISH: %K and %D are both < 20 (oversold) AND %K crosses above %D
      BEARISH: %K and %D are both > 80 (overbought) AND %K crosses below %D
      NEUTRAL: Not in extreme zone or no crossover

    Requiring both zone + crossover reduces false signals in trending markets.
    """
    name   = "Stochastic"
    k_p    = Config.STOCH_K
    d_p    = Config.STOCH_D
    smooth = Config.STOCH_SMOOTH

    if len(df) < k_p + d_p + smooth + 5:
        return _neutral(name, f"Need {k_p+d_p+smooth+5}+ candles for Stochastic")

    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        lowest_low   = low.rolling(k_p).min()
        highest_high = high.rolling(k_p).max()
        denom        = (highest_high - lowest_low).replace(0, np.nan)

        raw_k = 100 * (close - lowest_low) / denom
        k_line = raw_k.rolling(smooth).mean()      # smoothed %K
        d_line = k_line.rolling(d_p).mean()        # %D

        k_now  = _last(k_line)
        d_now  = _last(d_line)
        k_prev = _prev(k_line)
        d_prev = _prev(d_line)

        if any(np.isnan(v) for v in [k_now, d_now]):
            return _neutral(name, "Stochastic returned NaN")

        bullish_cross = (not np.isnan(k_prev)) and (k_prev <= d_prev) and (k_now > d_now)
        bearish_cross = (not np.isnan(k_prev)) and (k_prev >= d_prev) and (k_now < d_now)

        in_oversold   = k_now < Config.STOCH_OVERSOLD  and d_now < Config.STOCH_OVERSOLD
        in_overbought = k_now > Config.STOCH_OVERBOUGHT and d_now > Config.STOCH_OVERBOUGHT

        values = {
            "k": k_now, "d": d_now,
            "bullish_cross": int(bullish_cross), "bearish_cross": int(bearish_cross),
        }

        if in_oversold and bullish_cross:
            return IndicatorResult(
                name=name, signal=BULLISH, value=k_now, values=values,
                reason=f"%K({k_now:.1f}) crossed above %D({d_now:.1f}) in oversold zone"
            )
        if in_overbought and bearish_cross:
            return IndicatorResult(
                name=name, signal=BEARISH, value=k_now, values=values,
                reason=f"%K({k_now:.1f}) crossed below %D({d_now:.1f}) in overbought zone"
            )

        # Secondary: already deeply in zone with no cross yet
        if in_oversold:
            return IndicatorResult(
                name=name, signal=NEUTRAL, value=k_now, values=values,
                reason=f"Oversold (%K={k_now:.1f}) — awaiting %K/%D crossover"
            )
        if in_overbought:
            return IndicatorResult(
                name=name, signal=NEUTRAL, value=k_now, values=values,
                reason=f"Overbought (%K={k_now:.1f}) — awaiting %K/%D crossover"
            )

        return IndicatorResult(
            name=name, signal=NEUTRAL, value=k_now, values=values,
            reason=f"%K={k_now:.1f}, %D={d_now:.1f} — mid-range"
        )

    except Exception as exc:
        logger.exception("Stochastic calculation error: %s", exc)
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
# INDICATOR 9 — ATR (volatility context, not directional)
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
# INDICATOR 10 — VOLUME
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
    Calculate all 10 indicators on the supplied DataFrame and return
    an AllIndicators object with every result populated.

    Indicators are computed in dependency order:
      - EMA, ADX, Ichimoku, RSI, MACD, Stoch, CCI, BBands, ATR first
      - Volume last (needs direction hint from the other 9)

    Args:
        df: Clean, processed OHLCV DataFrame (output of processor.py)

    Returns:
        AllIndicators with all 10 results. No result will be None —
        indicators that can't compute return a NEUTRAL result with an
        explanatory reason string.
    """
    logger.debug("Calculating all indicators on %d candles…", len(df))

    ema_r   = calc_ema(df)
    adx_r   = calc_adx(df)
    ich_r   = calc_ichimoku(df)
    rsi_r   = calc_rsi(df)
    macd_r  = calc_macd(df)
    stoch_r = calc_stochastic(df)
    cci_r   = calc_cci(df)
    bb_r    = calc_bbands(df)
    atr_r   = calc_atr(df)

    # Determine provisional direction from first 9 for volume confirmation
    first_nine = [ema_r, adx_r, ich_r, rsi_r, macd_r, stoch_r, cci_r, bb_r, atr_r]
    bull_count = sum(1 for r in first_nine if r.signal == BULLISH)
    bear_count = sum(1 for r in first_nine if r.signal == BEARISH)
    if   bull_count > bear_count: provisional = BULLISH
    elif bear_count > bull_count: provisional = BEARISH
    else:                          provisional = None

    vol_r = calc_volume(df, overall_direction=provisional)

    latest_close = _last(df["close"])

    result = AllIndicators(
        ema=ema_r, adx=adx_r, ichimoku=ich_r,
        rsi=rsi_r, macd=macd_r, stochastic=stoch_r,
        cci=cci_r, bbands=bb_r, atr=atr_r, volume=vol_r,
        latest_close=latest_close,
    )

    logger.info(
        "Indicators: %s | Price=%.2f",
        result.summary_line(), latest_close
    )
    return result
