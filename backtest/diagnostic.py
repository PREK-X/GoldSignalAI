"""
GoldSignalAI — backtest/diagnostic.py
======================================
Signal diagnostic: measures individual indicator accuracy and session win rates
on the 60-day M15 gold data.

Captures ALL signals (before any filters) and checks what price actually did
3 candles later. Outputs per-indicator hit rates and session breakdowns.
"""

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from config import Config
from data.processor import process
from analysis.indicators import calculate_all, BULLISH, BEARISH
from analysis.sr_levels import detect_sr_levels
from analysis.fibonacci import calculate_fibonacci
from analysis.candlestick import detect_patterns
from analysis.scoring import score_signal

# Suppress noisy logs — we only want our diagnostic output
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH (reuse backtest engine's dual-timeframe approach)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_data():
    """Fetch M15 + H1 data the same way the backtest does."""
    from backtest.engine import _yf_fetch, BacktestConfig

    cfg = BacktestConfig()

    print("Fetching M15 data (60 days)...")
    m15_raw = _yf_fetch("GC=F", "15m", 59, "M15")

    print("Fetching H1 data (1 year)...")
    h1_raw = _yf_fetch("GC=F", "1h", 365, "H1")

    if m15_raw is None or h1_raw is None:
        raise RuntimeError("Failed to fetch data")

    m15 = process(m15_raw, label="DIAG_M15")
    h1 = process(h1_raw, label="DIAG_H1")

    print(f"M15: {len(m15)} bars | H1: {len(h1)} bars\n")
    return m15, h1


# ─────────────────────────────────────────────────────────────────────────────
# SESSION CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _classify_session(dt: datetime) -> str:
    """Classify a UTC timestamp into trading session."""
    h = dt.hour
    # London: 07:00–16:00 UTC
    # NY:     12:00–21:00 UTC
    # Overlap: 12:00–16:00 UTC
    # Asia:   23:00–07:00 UTC
    if 12 <= h < 16:
        return "Overlap"
    elif 7 <= h < 12:
        return "London"
    elif 16 <= h < 21:
        return "NewYork"
    else:
        return "Outside"  # Asia + off-hours


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    bar_idx: int
    bar_time: datetime
    direction: str              # BUY or SELL
    confidence: float
    session: str
    price_at_signal: float
    price_3bars_later: float
    actual_moved_up: bool       # True if close[+3] > close[0]
    signal_correct: bool        # direction matches actual move
    indicator_signals: dict     # {name: "bullish"/"bearish"/"neutral"}
    m15_bullish: int
    m15_bearish: int
    m15_neutral: int


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic():
    m15, h1 = _fetch_data()

    lookback = Config.LOOKBACK_CANDLES
    min_bars = Config.MIN_CANDLES_FOR_SIGNAL
    total_bars = len(m15)
    start_idx = max(lookback, min_bars)

    # We need at least 3 bars after the signal to check outcome
    end_idx = total_bars - 3

    records: list[SignalRecord] = []
    signals_checked = 0
    wait_count = 0

    print(f"Scanning {end_idx - start_idx} bars for signals...\n")

    for i in range(start_idx, end_idx):
        bar = m15.iloc[i]
        bar_time = m15.index[i]

        # ── Build slices ─────────────────────────────────────────────
        m15_slice = m15.iloc[max(0, i - lookback):i + 1]
        h1_mask = h1.index <= bar_time
        h1_slice = h1[h1_mask].tail(lookback)

        if len(m15_slice) < min_bars or len(h1_slice) < 50:
            continue

        signals_checked += 1

        # ── Run analysis (both timeframes) ───────────────────────────
        try:
            m15_ind = calculate_all(m15_slice)
            m15_sr = detect_sr_levels(m15_slice)
            m15_fib = calculate_fibonacci(m15_slice)
            m15_cand = detect_patterns(m15_slice)
            m15_score = score_signal(m15_ind, m15_sr, m15_fib, m15_cand, bar_time=bar_time)

            h1_ind = calculate_all(h1_slice)
            h1_sr = detect_sr_levels(h1_slice)
            h1_fib = calculate_fibonacci(h1_slice)
            h1_cand = detect_patterns(h1_slice)
            h1_score = score_signal(h1_ind, h1_sr, h1_fib, h1_cand)
        except Exception:
            continue

        # ── Multi-timeframe agreement ────────────────────────────────
        m15_dir = m15_score.direction
        h1_dir = h1_score.direction
        agree = (m15_dir == h1_dir) and m15_dir in ("BUY", "SELL")

        if not agree:
            wait_count += 1
            continue

        direction = m15_dir
        confidence = min(m15_score.confidence_pct, h1_score.confidence_pct)

        # ── What actually happened 3 candles later ───────────────────
        price_now = m15.iloc[i]["close"]
        price_3 = m15.iloc[i + 3]["close"]
        moved_up = price_3 > price_now

        if direction == "BUY":
            correct = moved_up
        else:  # SELL
            correct = not moved_up

        # ── Per-indicator signals ────────────────────────────────────
        ind_signals = {}
        for r in m15_ind.as_list():
            ind_signals[r.name] = r.signal

        rec = SignalRecord(
            bar_idx=i,
            bar_time=bar_time,
            direction=direction,
            confidence=confidence,
            session=_classify_session(bar_time),
            price_at_signal=price_now,
            price_3bars_later=price_3,
            actual_moved_up=moved_up,
            signal_correct=correct,
            indicator_signals=ind_signals,
            m15_bullish=m15_score.bullish_count,
            m15_bearish=m15_score.bearish_count,
            m15_neutral=m15_score.neutral_count,
        )
        records.append(rec)

    print(f"Bars scanned: {signals_checked}")
    print(f"WAIT (no agreement): {wait_count}")
    print(f"Actionable signals: {len(records)}")
    print()

    if not records:
        print("No signals found — nothing to analyze.")
        return

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 1: Overall signal accuracy
    # ─────────────────────────────────────────────────────────────────

    total = len(records)
    correct = sum(1 for r in records if r.signal_correct)
    buy_recs = [r for r in records if r.direction == "BUY"]
    sell_recs = [r for r in records if r.direction == "SELL"]
    buy_correct = sum(1 for r in buy_recs if r.signal_correct)
    sell_correct = sum(1 for r in sell_recs if r.signal_correct)

    print("=" * 65)
    print(" OVERALL SIGNAL ACCURACY (3 candles = 45 min forward)")
    print("=" * 65)
    print(f"  Total signals:  {total}")
    print(f"  Correct:        {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  BUY signals:    {len(buy_recs):>4d} | correct: {buy_correct}/{len(buy_recs)} = "
          f"{buy_correct/len(buy_recs)*100:.1f}%" if buy_recs else "  BUY signals: 0")
    print(f"  SELL signals:   {len(sell_recs):>4d} | correct: {sell_correct}/{len(sell_recs)} = "
          f"{sell_correct/len(sell_recs)*100:.1f}%" if sell_recs else "  SELL signals: 0")
    print()

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 2: Per-indicator accuracy
    # ─────────────────────────────────────────────────────────────────

    # For each indicator, when it says "bullish", how often does price go up?
    # When it says "bearish", how often does price go down?
    indicator_names = list(records[0].indicator_signals.keys())

    print("=" * 65)
    print(" PER-INDICATOR PREDICTIVE ACCURACY")
    print("=" * 65)
    print(f"  {'Indicator':<15s} {'Bullish→Up':>12s} {'Bearish→Down':>14s} "
          f"{'Combined':>10s} {'Bull#':>6s} {'Bear#':>6s} {'Neut#':>6s}")
    print("  " + "─" * 63)

    indicator_stats = {}

    for name in indicator_names:
        bull_total = 0
        bull_correct = 0
        bear_total = 0
        bear_correct = 0
        neut_total = 0

        for r in records:
            sig = r.indicator_signals.get(name)
            if sig == BULLISH:
                bull_total += 1
                if r.actual_moved_up:
                    bull_correct += 1
            elif sig == BEARISH:
                bear_total += 1
                if not r.actual_moved_up:
                    bear_correct += 1
            else:
                neut_total += 1

        bull_pct = (bull_correct / bull_total * 100) if bull_total > 0 else 0
        bear_pct = (bear_correct / bear_total * 100) if bear_total > 0 else 0

        active_total = bull_total + bear_total
        active_correct = bull_correct + bear_correct
        combined_pct = (active_correct / active_total * 100) if active_total > 0 else 0

        indicator_stats[name] = {
            "bull_total": bull_total, "bull_correct": bull_correct, "bull_pct": bull_pct,
            "bear_total": bear_total, "bear_correct": bear_correct, "bear_pct": bear_pct,
            "neut_total": neut_total, "combined_pct": combined_pct,
            "active_total": active_total, "active_correct": active_correct,
        }

        bull_str = f"{bull_correct}/{bull_total}={bull_pct:.0f}%" if bull_total > 0 else "—"
        bear_str = f"{bear_correct}/{bear_total}={bear_pct:.0f}%" if bear_total > 0 else "—"
        comb_str = f"{combined_pct:.1f}%"

        # Highlight indicators above 50%
        marker = " ✓" if combined_pct > 50 and active_total >= 10 else ""

        print(f"  {name:<15s} {bull_str:>12s} {bear_str:>14s} "
              f"{comb_str:>10s} {bull_total:>6d} {bear_total:>6d} {neut_total:>6d}{marker}")

    print()

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 3: Indicators with >50% accuracy (statistically meaningful)
    # ─────────────────────────────────────────────────────────────────

    print("=" * 65)
    print(" INDICATORS WITH >50% ACCURACY (min 10 active signals)")
    print("=" * 65)

    good_indicators = []
    for name, s in indicator_stats.items():
        if s["active_total"] >= 10 and s["combined_pct"] > 50:
            good_indicators.append((name, s["combined_pct"], s["active_total"]))

    if good_indicators:
        good_indicators.sort(key=lambda x: x[1], reverse=True)
        for name, pct, n in good_indicators:
            print(f"  {name:<15s} {pct:.1f}% accuracy ({n} signals)")
    else:
        print("  NONE — no indicator has >50% directional accuracy on this data")
    print()

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 4: Session-based win rates
    # ─────────────────────────────────────────────────────────────────

    print("=" * 65)
    print(" WIN RATE BY SESSION (UTC times)")
    print("=" * 65)

    sessions = ["London", "Overlap", "NewYork", "Outside"]
    for sess in sessions:
        sess_recs = [r for r in records if r.session == sess]
        if sess_recs:
            sess_correct = sum(1 for r in sess_recs if r.signal_correct)
            pct = sess_correct / len(sess_recs) * 100
            marker = " ✓" if pct > 50 else ""
            print(f"  {sess:<12s} {sess_correct}/{len(sess_recs):>3d} = {pct:.1f}%{marker}")
        else:
            print(f"  {sess:<12s} — no signals")
    print()

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 5: Direction bias check
    # ─────────────────────────────────────────────────────────────────

    print("=" * 65)
    print(" DIRECTION BIAS CHECK")
    print("=" * 65)

    up_bars = sum(1 for r in records if r.actual_moved_up)
    down_bars = total - up_bars
    print(f"  Actual market movement (at signal times):")
    print(f"    Up:   {up_bars}/{total} = {up_bars/total*100:.1f}%")
    print(f"    Down: {down_bars}/{total} = {down_bars/total*100:.1f}%")
    print()
    print(f"  Signal direction distribution:")
    print(f"    BUY:  {len(buy_recs)}/{total} = {len(buy_recs)/total*100:.1f}%")
    print(f"    SELL: {len(sell_recs)}/{total} = {len(sell_recs)/total*100:.1f}%")
    print()

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 6: Confidence vs accuracy
    # ─────────────────────────────────────────────────────────────────

    print("=" * 65)
    print(" CONFIDENCE vs ACCURACY")
    print("=" * 65)

    # Group by confidence brackets
    brackets = [(65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 100)]
    for lo, hi in brackets:
        bracket_recs = [r for r in records if lo <= r.confidence < hi]
        if bracket_recs:
            bc = sum(1 for r in bracket_recs if r.signal_correct)
            pct = bc / len(bracket_recs) * 100
            marker = " ✓" if pct > 50 else ""
            print(f"  {lo}–{hi}%:  {bc}/{len(bracket_recs):>3d} = {pct:.1f}%{marker}")
    print()

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 7: Price move magnitude
    # ─────────────────────────────────────────────────────────────────

    print("=" * 65)
    print(" PRICE MOVE ANALYSIS (3-bar forward)")
    print("=" * 65)

    pip_size = Config.PIP_SIZE  # 0.10 for gold

    moves_pips = []
    for r in records:
        if r.direction == "BUY":
            move = (r.price_3bars_later - r.price_at_signal) / pip_size
        else:
            move = (r.price_at_signal - r.price_3bars_later) / pip_size
        moves_pips.append(move)

    moves = np.array(moves_pips)
    print(f"  Mean move in signal direction: {moves.mean():+.1f} pips")
    print(f"  Median move:                   {np.median(moves):+.1f} pips")
    print(f"  Std dev:                        {moves.std():.1f} pips")
    print(f"  Winners (move > 0):             {(moves > 0).sum()}/{len(moves)} = {(moves > 0).mean()*100:.1f}%")
    print(f"  Winners > 10 pips:              {(moves > 10).sum()}/{len(moves)} = {(moves > 10).mean()*100:.1f}%")
    print(f"  Winners > 30 pips:              {(moves > 30).sum()}/{len(moves)} = {(moves > 30).mean()*100:.1f}%")
    print(f"  Losers < -30 pips:              {(moves < -30).sum()}/{len(moves)} = {(moves < -30).mean()*100:.1f}%")
    print()

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS 8: Indicator agreement depth vs accuracy
    # ─────────────────────────────────────────────────────────────────

    print("=" * 65)
    print(" AGREEMENT DEPTH vs ACCURACY")
    print("=" * 65)
    print("  (how many bullish/bearish indicators when signal fires)")

    for depth in range(3, 9):
        # Dominant count = max(bullish, bearish)
        depth_recs = [r for r in records
                      if max(r.m15_bullish, r.m15_bearish) == depth]
        if depth_recs:
            dc = sum(1 for r in depth_recs if r.signal_correct)
            pct = dc / len(depth_recs) * 100
            marker = " ✓" if pct > 50 else ""
            print(f"  {depth} dominant:  {dc}/{len(depth_recs):>3d} = {pct:.1f}%{marker}")
    print()


if __name__ == "__main__":
    run_diagnostic()
