"""
GoldSignalAI — alerts/chart_generator.py
==========================================
Generates candlestick chart images for trading signals.

Each chart includes:
  - OHLCV candlestick chart (last 50 candles)
  - Entry price line (blue)
  - Stop Loss line (red)
  - Take Profit 1 & 2 lines (green)
  - EMA 20/50/200 overlays
  - Volume bars
  - Signal direction annotation

Output: PNG image saved to a temp file path (for Telegram attachment).

Uses plotly for chart generation and kaleido for static export.
Falls back to matplotlib if plotly/kaleido not available.
"""

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# Try plotly first, fall back to matplotlib
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_chart(
    df:          pd.DataFrame,
    direction:   str,
    entry_price: float,
    sl_price:    Optional[float] = None,
    tp1_price:   Optional[float] = None,
    tp2_price:   Optional[float] = None,
    n_candles:   int = 50,
    title:       Optional[str] = None,
) -> Optional[str]:
    """
    Generate a candlestick chart image with signal levels.

    Args:
        df:          OHLCV DataFrame with columns [open, high, low, close, volume]
        direction:   "BUY" or "SELL" or "WAIT"
        entry_price: Entry price level
        sl_price:    Stop loss price (optional)
        tp1_price:   Take profit 1 price (optional)
        tp2_price:   Take profit 2 price (optional)
        n_candles:   Number of candles to display
        title:       Chart title (default: auto-generated)

    Returns:
        Path to the generated PNG file, or None if generation fails.
    """
    if df is None or df.empty:
        logger.warning("Cannot generate chart: empty DataFrame")
        return None

    # Use last N candles
    chart_df = df.tail(n_candles).copy()

    if title is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        icon = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        title = f"GoldSignalAI — {Config.SYMBOL_DISPLAY} {direction} {icon} | {ts}"

    if PLOTLY_AVAILABLE:
        return _generate_plotly(chart_df, direction, entry_price,
                                sl_price, tp1_price, tp2_price, title)
    elif MPL_AVAILABLE:
        return _generate_matplotlib(chart_df, direction, entry_price,
                                    sl_price, tp1_price, tp2_price, title)
    else:
        logger.error("Neither plotly nor matplotlib available for chart generation")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART
# ─────────────────────────────────────────────────────────────────────────────

def _generate_plotly(
    df, direction, entry, sl, tp1, tp2, title
) -> Optional[str]:
    """Generate chart using plotly + kaleido."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # EMAs (if enough data)
        for period, color, name in [
            (Config.EMA_FAST, "#2196F3", f"EMA {Config.EMA_FAST}"),
            (Config.EMA_MID,  "#FF9800", f"EMA {Config.EMA_MID}"),
        ]:
            if len(df) >= period:
                ema = df["close"].ewm(span=period, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=df.index, y=ema,
                    mode="lines", name=name,
                    line=dict(color=color, width=1),
                ), row=1, col=1)

        # Signal levels
        _add_hline(fig, entry, "#2196F3", f"Entry {entry:.2f}", "dash")
        if sl:
            _add_hline(fig, sl, "#F44336", f"SL {sl:.2f}", "dot")
        if tp1:
            _add_hline(fig, tp1, "#4CAF50", f"TP1 {tp1:.2f}", "dot")
        if tp2:
            _add_hline(fig, tp2, "#4CAF50", f"TP2 {tp2:.2f}", "dashdot")

        # Volume bars
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=colors, name="Volume",
            opacity=0.5,
        ), row=2, col=1)

        # Layout
        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=600, width=1000,
            margin=dict(l=60, r=20, t=50, b=30),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="goldsignal_")
        fig.write_image(tmp.name, format="png")
        logger.info("Chart generated (plotly): %s", tmp.name)
        return tmp.name

    except Exception as exc:
        logger.warning("Plotly chart failed: %s — trying matplotlib", exc)
        if MPL_AVAILABLE:
            return _generate_matplotlib(df, direction, entry, sl, tp1, tp2, title)
        return None


def _add_hline(fig, price, color, label, dash):
    """Add a horizontal line annotation to a plotly figure."""
    fig.add_hline(
        y=price, line_color=color, line_dash=dash,
        annotation_text=label, annotation_position="top left",
        row=1, col=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _generate_matplotlib(
    df, direction, entry, sl, tp1, tp2, title
) -> Optional[str]:
    """Generate chart using matplotlib (simpler, always available)."""
    try:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.patch.set_facecolor("#1a1a2e")

        # Candlestick approximation using bar chart
        x = range(len(df))
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]

        # Draw wicks
        for i, (idx, row) in enumerate(df.iterrows()):
            color = colors[i]
            ax1.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
            body_bottom = min(row["open"], row["close"])
            body_height = abs(row["close"] - row["open"])
            ax1.bar(i, body_height, bottom=body_bottom, width=0.6,
                    color=color, edgecolor=color)

        ax1.set_facecolor("#16213e")
        ax1.set_title(title, color="white", fontsize=10)
        ax1.tick_params(colors="white")

        # Signal levels
        ax1.axhline(y=entry, color="#2196F3", linestyle="--", linewidth=1,
                     label=f"Entry {entry:.2f}")
        if sl:
            ax1.axhline(y=sl, color="#F44336", linestyle=":", linewidth=1,
                         label=f"SL {sl:.2f}")
        if tp1:
            ax1.axhline(y=tp1, color="#4CAF50", linestyle=":", linewidth=1,
                         label=f"TP1 {tp1:.2f}")
        if tp2:
            ax1.axhline(y=tp2, color="#4CAF50", linestyle="-.", linewidth=1,
                         label=f"TP2 {tp2:.2f}")

        ax1.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e",
                   edgecolor="white", labelcolor="white")

        # Volume
        ax2.bar(x, df["volume"], color=colors, alpha=0.5)
        ax2.set_facecolor("#16213e")
        ax2.tick_params(colors="white")
        ax2.set_ylabel("Volume", color="white", fontsize=8)

        plt.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="goldsignal_")
        fig.savefig(tmp.name, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info("Chart generated (matplotlib): %s", tmp.name)
        return tmp.name

    except Exception as exc:
        logger.exception("Matplotlib chart failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Generate from TradingSignal
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal_chart(sig, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """
    Generate a chart for a TradingSignal.

    Args:
        sig: TradingSignal from signals/generator.py
        df:  OHLCV DataFrame (if None, uses sig.mtf_result.m15.df)

    Returns:
        Path to PNG file, or None.
    """
    if df is None:
        try:
            df = sig.mtf_result.m15.df
        except AttributeError:
            logger.warning("No DataFrame available for chart generation")
            return None

    if df is None or df.empty:
        return None

    sl  = sig.risk.stop_loss if sig.risk else None
    tp1 = sig.risk.tp1_price if sig.risk else None
    tp2 = sig.risk.tp2_price if sig.risk else None

    return generate_chart(
        df=df,
        direction=sig.direction,
        entry_price=sig.entry_price,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=tp2,
    )
