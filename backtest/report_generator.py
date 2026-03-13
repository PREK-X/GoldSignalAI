"""
GoldSignalAI — backtest/report_generator.py
=============================================
Generates professional PDF backtest reports using fpdf2.

Report sections:
  1. Cover page — title, period, headline stats
  2. Performance Summary — win rate, PnL, profit factor, Sharpe
  3. Drawdown & Risk — max drawdown, streaks, equity curve chart
  4. Monthly Breakdown — table with month-by-month performance
  5. Prop Firm Simulations — pass/fail for each firm preset
  6. Trade Log — last 50 trades in tabular form

Usage:
    from backtest.report_generator import generate_pdf_report
    from backtest.engine import run_backtest

    result = run_backtest()
    path = generate_pdf_report(result)
    print(f"Report saved to: {path}")
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from fpdf import FPDF, XPos, YPos

from config import Config
from backtest.engine import BacktestResult, BacktestTrade, PropFirmSimulation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette (RGB)
DARK_BG      = (18, 18, 18)
PANEL_BG     = (30, 30, 30)
GOLD         = (212, 175, 55)
GREEN        = (0, 200, 100)
RED          = (220, 50, 50)
GREY         = (120, 120, 120)
WHITE        = (240, 240, 240)
LIGHT_GREY   = (200, 200, 200)
ACCENT_BLUE  = (70, 130, 200)

PAGE_W = 210  # A4 width mm
PAGE_H = 297  # A4 height mm
MARGIN = 15   # page margin mm


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY CURVE CHART (matplotlib → temp PNG)
# ─────────────────────────────────────────────────────────────────────────────

def _build_equity_chart(result: BacktestResult, out_path: str) -> bool:
    """
    Render the equity curve as a PNG file for embedding in the PDF.

    Returns True on success, False on failure.
    """
    if len(result.equity_curve) < 2:
        return False

    try:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        equity = result.equity_curve
        dates = result.equity_dates

        x = range(len(equity))

        # Fill under curve — green above starting balance, red below
        start = equity[0]
        above = [e if e >= start else start for e in equity]
        below = [e if e < start else start for e in equity]

        ax.plot(x, equity, color="#d4af37", linewidth=1.5, zorder=3)
        ax.fill_between(x, start, above, alpha=0.25, color="#00c864", zorder=2)
        ax.fill_between(x, start, below, alpha=0.25, color="#dc3232", zorder=2)
        ax.axhline(y=start, color="#666666", linewidth=0.8, linestyle="--")

        ax.set_xlabel("Trade #", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Account Balance ($)", color="#aaaaaa", fontsize=8)
        ax.set_title("Equity Curve", color="#d4af37", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.grid(True, color="#333333", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig)
        return True

    except Exception as exc:
        logger.warning("Could not render equity chart: %s", exc)
        return False


def _build_monthly_chart(result: BacktestResult, out_path: str) -> bool:
    """Render monthly PnL as a bar chart PNG."""
    if not result.monthly:
        return False

    try:
        months = [m.month for m in result.monthly]
        pnl_pips = [m.pnl_pips for m in result.monthly]
        colors = ["#00c864" if p >= 0 else "#dc3232" for p in pnl_pips]

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        x = range(len(months))
        bars = ax.bar(x, pnl_pips, color=colors, width=0.6, edgecolor="#333333")

        ax.set_xticks(list(x))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=6, color="#aaaaaa")
        ax.set_ylabel("PnL (pips)", color="#aaaaaa", fontsize=8)
        ax.set_title("Monthly Performance (Pips)", color="#d4af37", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.axhline(y=0, color="#666666", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(True, axis="y", color="#333333", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig)
        return True

    except Exception as exc:
        logger.warning("Could not render monthly chart: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM FPDF CLASS
# ─────────────────────────────────────────────────────────────────────────────

class GoldSignalPDF(FPDF):
    """
    Custom FPDF subclass with GoldSignalAI branding.
    Dark theme with gold accents.
    """

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.set_margins(MARGIN, MARGIN, MARGIN)

    # ── Header / Footer ───────────────────────────────────────────────────────

    def header(self):
        """Thin gold bar at top of every page (except cover)."""
        if self.page_no() == 1:
            return
        self.set_fill_color(*GOLD)
        self.rect(0, 0, PAGE_W, 3, style="F")
        self.set_y(6)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*GREY)
        self.cell(0, 4, "GoldSignalAI - Backtest Report", align="L")
        self.cell(0, 4, f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", align="R")
        self.ln(4)

    def footer(self):
        """Page number at bottom."""
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GREY)
        self.cell(0, 4, f"Page {self.page_no()}", align="C")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def set_dark_bg(self):
        """Fill the entire page with dark background."""
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, PAGE_W, PAGE_H, style="F")

    def section_title(self, text: str):
        """Bold gold section heading with underline."""
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*GOLD)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*GOLD)
        self.set_line_width(0.4)
        x = self.get_x()
        y = self.get_y()
        self.line(MARGIN, y, PAGE_W - MARGIN, y)
        self.ln(3)

    def stat_box(self, label: str, value: str, x: float, y: float, w: float = 42, h: float = 16, value_color=None):
        """Draw a small stat card at position (x, y)."""
        if value_color is None:
            value_color = WHITE
        self.set_fill_color(*PANEL_BG)
        self.set_draw_color(*GREY)
        self.set_line_width(0.3)
        self.rect(x, y, w, h, style="FD")

        self.set_font("Helvetica", "", 6.5)
        self.set_text_color(*GREY)
        self.set_xy(x + 2, y + 2)
        self.cell(w - 4, 4, label.upper())

        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*value_color)
        self.set_xy(x + 2, y + 7)
        self.cell(w - 4, 7, value)

    def data_row(self, cols: list[str], widths: list[float], is_header: bool = False, is_alt: bool = False):
        """Render one row of a data table."""
        row_h = 6

        if is_header:
            self.set_fill_color(*PANEL_BG)
            self.set_font("Helvetica", "B", 7)
            self.set_text_color(*GOLD)
        elif is_alt:
            self.set_fill_color(25, 25, 25)
            self.set_font("Helvetica", "", 6.5)
            self.set_text_color(*LIGHT_GREY)
        else:
            self.set_fill_color(*DARK_BG)
            self.set_font("Helvetica", "", 6.5)
            self.set_text_color(*LIGHT_GREY)

        for col, w in zip(cols, widths):
            self.cell(w, row_h, str(col), border=0, fill=True)
        self.ln(row_h)


# ─────────────────────────────────────────────────────────────────────────────
# COVER PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _cover_page(pdf: GoldSignalPDF, result: BacktestResult):
    """Build the report cover page."""
    pdf.add_page()
    pdf.set_dark_bg()

    # Gold top bar
    pdf.set_fill_color(*GOLD)
    pdf.rect(0, 0, PAGE_W, 8, style="F")

    # Title block
    pdf.set_y(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*GOLD)
    pdf.cell(0, 14, "GoldSignalAI", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*LIGHT_GREY)
    pdf.cell(0, 8, "Backtest Performance Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GREY)

    period = ""
    if result.start_date and result.end_date:
        period = f"{result.start_date.strftime('%d %b %Y')}  to  {result.end_date.strftime('%d %b %Y')}"
    pdf.cell(0, 6, period, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, f"Account: ${result.config.account_balance:,.0f}   |   "
                   f"Spread: {result.config.spread_pips} pips   |   "
                   f"Risk/trade: {result.config.risk_per_trade_pct}%",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Divider
    pdf.ln(10)
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.5)
    pdf.line(MARGIN + 30, pdf.get_y(), PAGE_W - MARGIN - 30, pdf.get_y())
    pdf.ln(10)

    # Headline stats grid  (4 per row × 2 rows)
    cfg = result.config
    win_pct = result.win_rate_pct
    win_color = GREEN if win_pct >= 60 else (RED if win_pct < 50 else GOLD)
    pnl_color = GREEN if result.total_pnl_usd >= 0 else RED
    dd_color = GREEN if result.max_drawdown_pct < 5 else (RED if result.max_drawdown_pct > 10 else GOLD)
    pf_color = GREEN if result.profit_factor >= 1.5 else (RED if result.profit_factor < 1 else GOLD)

    stats = [
        ("Win Rate",       f"{win_pct:.1f}%",          win_color),
        ("Total Trades",   str(result.total_trades),    WHITE),
        ("Total PnL",      f"${result.total_pnl_usd:+,.2f}", pnl_color),
        ("Final Balance",  f"${result.final_balance:,.2f}", pnl_color),
        ("Profit Factor",  f"{result.profit_factor:.2f}",   pf_color),
        ("Sharpe Ratio",   f"{result.sharpe_ratio:.2f}",    WHITE),
        ("Max Drawdown",   f"{result.max_drawdown_pct:.2f}%", dd_color),
        ("Trading Days",   str(result.trading_days),         WHITE),
    ]

    box_w = 42
    box_h = 18
    gap = 4
    cols = 4
    row_start_x = MARGIN + (PAGE_W - 2 * MARGIN - cols * box_w - (cols - 1) * gap) / 2

    for i, (label, value, color) in enumerate(stats):
        col = i % cols
        row = i // cols
        x = row_start_x + col * (box_w + gap)
        y = pdf.get_y() + row * (box_h + gap)
        if col == 0 and row > 0 and i == cols:
            pdf.ln(box_h + gap)
        pdf.stat_box(label, value, x, y, w=box_w, h=box_h, value_color=color)

    pdf.ln(box_h * 2 + gap * 2 + 10)

    # Footer note
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5,
             "Past performance does not guarantee future results. "
             "For educational and research purposes only.",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Gold bottom bar
    pdf.set_fill_color(*GOLD)
    pdf.rect(0, PAGE_H - 8, PAGE_W, 8, style="F")


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _performance_page(pdf: GoldSignalPDF, result: BacktestResult, equity_chart: Optional[str], monthly_chart: Optional[str]):
    """Build the performance summary + charts page."""
    pdf.add_page()
    pdf.set_dark_bg()
    pdf.set_y(MARGIN + 8)

    pdf.section_title("Performance Summary")

    # ── Stat boxes row 1 ──────────────────────────────────────────────
    box_w = 38
    box_h = 15
    gap = 3
    cols = 4
    x0 = MARGIN

    row1 = [
        ("Winners",     f"{result.winners}",               GREEN),
        ("Losers",      f"{result.losers}",                RED),
        ("Avg Win",     f"{result.avg_win_pips:+.1f} pips", GREEN),
        ("Avg Loss",    f"{result.avg_loss_pips:.1f} pips", RED),
    ]
    row2 = [
        ("Best Trade",  f"{result.best_trade_pips:+.1f} pips",  GREEN),
        ("Worst Trade", f"{result.worst_trade_pips:.1f} pips",   RED),
        ("Win Streak",  f"{result.best_streak}",                 GREEN),
        ("Loss Streak", f"{result.worst_streak}",                RED),
    ]

    y0 = pdf.get_y()
    for i, (label, value, color) in enumerate(row1):
        pdf.stat_box(label, value, x0 + i * (box_w + gap), y0, w=box_w, h=box_h, value_color=color)
    pdf.ln(box_h + gap + 2)

    y0 = pdf.get_y()
    for i, (label, value, color) in enumerate(row2):
        pdf.stat_box(label, value, x0 + i * (box_w + gap), y0, w=box_w, h=box_h, value_color=color)
    pdf.ln(box_h + gap + 6)

    # ── Equity curve ───────────────────────────────────────────────────
    if equity_chart and os.path.exists(equity_chart):
        pdf.section_title("Equity Curve")
        chart_w = PAGE_W - 2 * MARGIN
        chart_h = chart_w * 0.35
        pdf.image(equity_chart, x=MARGIN, y=pdf.get_y(), w=chart_w, h=chart_h)
        pdf.ln(chart_h + 8)

    # ── Monthly chart ──────────────────────────────────────────────────
    if monthly_chart and os.path.exists(monthly_chart):
        pdf.section_title("Monthly PnL")
        chart_w = PAGE_W - 2 * MARGIN
        chart_h = chart_w * 0.30
        pdf.image(monthly_chart, x=MARGIN, y=pdf.get_y(), w=chart_w, h=chart_h)
        pdf.ln(chart_h + 8)


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY BREAKDOWN TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _monthly_table(pdf: GoldSignalPDF, result: BacktestResult):
    """Render the monthly breakdown table."""
    pdf.add_page()
    pdf.set_dark_bg()
    pdf.set_y(MARGIN + 8)
    pdf.section_title("Monthly Breakdown")

    if not result.monthly:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 8, "No monthly data available.")
        return

    headers  = ["Month",     "Trades", "Wins", "Losses", "Win Rate", "PnL (Pips)", "PnL (USD)"]
    col_w    = [30,          20,       15,     15,       20,         25,            25]

    pdf.data_row(headers, col_w, is_header=True)
    pdf.set_draw_color(*GREY)
    pdf.set_line_width(0.2)
    y = pdf.get_y()
    pdf.line(MARGIN, y, PAGE_W - MARGIN, y)

    for idx, m in enumerate(result.monthly):
        win_rate_str = f"{m.win_rate:.1f}%"
        pnl_pips_str = f"{m.pnl_pips:+.1f}"
        pnl_usd_str  = f"${m.pnl_usd:+.2f}"

        cols = [m.month, str(m.trades), str(m.wins), str(m.losses), win_rate_str, pnl_pips_str, pnl_usd_str]
        pdf.data_row(cols, col_w, is_alt=(idx % 2 == 1))

        # Check page break
        if pdf.get_y() > PAGE_H - 30:
            pdf.add_page()
            pdf.set_dark_bg()
            pdf.set_y(MARGIN + 8)
            pdf.data_row(headers, col_w, is_header=True)

    # Totals row
    pdf.ln(1)
    total_wins = sum(m.wins for m in result.monthly)
    total_losses = sum(m.losses for m in result.monthly)
    total_trades = sum(m.trades for m in result.monthly)
    total_pips = sum(m.pnl_pips for m in result.monthly)
    total_usd = sum(m.pnl_usd for m in result.monthly)
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    totals = ["TOTAL", str(total_trades), str(total_wins), str(total_losses),
              f"{overall_wr:.1f}%", f"{total_pips:+.1f}", f"${total_usd:+.2f}"]

    pdf.set_fill_color(*PANEL_BG)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*GOLD)
    for col, w in zip(totals, col_w):
        pdf.cell(w, 7, col, border=0, fill=True)
    pdf.ln(7)


# ─────────────────────────────────────────────────────────────────────────────
# PROP FIRM SIMULATIONS PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _prop_firm_page(pdf: GoldSignalPDF, result: BacktestResult):
    """Render prop firm challenge simulation results."""
    pdf.add_page()
    pdf.set_dark_bg()
    pdf.set_y(MARGIN + 8)
    pdf.section_title("Prop Firm Challenge Simulations")

    if not result.prop_firm_sims:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 8, "No prop firm simulations run.")
        return

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5,
             f"Based on {result.total_trades} trades | "
             f"Account: ${result.config.account_balance:,.0f} | "
             f"Risk/trade: {result.config.risk_per_trade_pct}%",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    headers = ["Firm", "Result", "PnL %", "Max DD %", "Max Daily Loss", "Days", "Notes"]
    col_w   = [38,     18,       18,      18,         25,               15,     39]

    pdf.data_row(headers, col_w, is_header=True)
    pdf.set_draw_color(*GREY)
    pdf.set_line_width(0.2)
    y = pdf.get_y()
    pdf.line(MARGIN, y, PAGE_W - MARGIN, y)

    for idx, sim in enumerate(result.prop_firm_sims):
        status = "PASSED" if sim.passed else "FAILED"
        pnl_str = f"{sim.final_pnl_pct:+.2f}%"
        dd_str  = f"{sim.max_drawdown_pct:.2f}%"
        dl_str  = f"{sim.max_daily_loss_pct:.2f}%"
        days_str = str(sim.days_to_complete) if sim.passed and sim.days_to_complete else str(sim.days_traded)
        notes = sim.breach_reason[:35] if sim.breach_reason else ("Challenge passed!" if sim.passed else "")

        pdf.set_fill_color(25, 25, 25) if idx % 2 == 1 else pdf.set_fill_color(*DARK_BG)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*WHITE)

        cells = [sim.firm_name, status, pnl_str, dd_str, dl_str, days_str, notes]
        colors = [WHITE, GREEN if sim.passed else RED, GREEN if sim.final_pnl_pct >= 0 else RED,
                  GREEN if sim.max_drawdown_pct < 5 else RED,
                  GREEN if sim.max_daily_loss_pct < 2 else RED, WHITE, GREY]

        fill = (25, 25, 25) if idx % 2 == 1 else DARK_BG
        pdf.set_fill_color(*fill)
        for col, w, color in zip(cells, col_w, colors):
            pdf.set_text_color(*color)
            pdf.set_font("Helvetica", "B" if col in ("PASSED", "FAILED") else "", 6.5)
            pdf.cell(w, 7, col, border=0, fill=True)
        pdf.ln(7)

    # Summary note below table
    passed_count = sum(1 for s in result.prop_firm_sims if s.passed)
    total_sims = len(result.prop_firm_sims)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5,
             f"Strategy passed {passed_count}/{total_sims} firm challenges simulated.",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Per-firm detail blocks for the top 3 passed/notable firms
    notable = sorted(result.prop_firm_sims, key=lambda s: (not s.passed, -s.final_pnl_pct))[:4]
    if notable:
        pdf.ln(8)
        pdf.section_title("Challenge Detail Cards")
        box_w = (PAGE_W - 2 * MARGIN - 9) / 4
        y_start = pdf.get_y()
        for i, sim in enumerate(notable):
            x = MARGIN + i * (box_w + 3)
            pdf.set_fill_color(*PANEL_BG)
            pdf.set_draw_color(GREEN[0], GREEN[1], GREEN[2]) if sim.passed else pdf.set_draw_color(RED[0], RED[1], RED[2])
            pdf.set_line_width(0.6)
            pdf.rect(x, y_start, box_w, 55, style="FD")

            pdf.set_xy(x + 2, y_start + 2)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*GOLD)
            pdf.cell(box_w - 4, 5, sim.firm_name[:18])

            lines_data = [
                ("Status",     "PASSED" if sim.passed else "FAILED",  GREEN if sim.passed else RED),
                ("Final PnL",  f"{sim.final_pnl_pct:+.2f}%",              GREEN if sim.final_pnl_pct >= 0 else RED),
                ("Max DD",     f"{sim.max_drawdown_pct:.2f}%",             GREEN if sim.max_drawdown_pct < 5 else RED),
                ("Daily Loss", f"{sim.max_daily_loss_pct:.2f}%",           GREEN if sim.max_daily_loss_pct < 2 else RED),
                ("Days",       str(sim.days_traded),                       WHITE),
            ]
            for j, (label, value, color) in enumerate(lines_data):
                pdf.set_xy(x + 2, y_start + 9 + j * 8)
                pdf.set_font("Helvetica", "", 6)
                pdf.set_text_color(*GREY)
                pdf.cell(box_w - 4, 4, label)
                pdf.set_xy(x + 2, y_start + 13 + j * 8)
                pdf.set_font("Helvetica", "B", 7.5)
                pdf.set_text_color(*color)
                pdf.cell(box_w - 4, 4, value)

        pdf.ln(60)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOG PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _trade_log_page(pdf: GoldSignalPDF, result: BacktestResult):
    """Render the last 50 trades table."""
    pdf.add_page()
    pdf.set_dark_bg()
    pdf.set_y(MARGIN + 8)

    max_trades = 50
    trades = result.trades[-max_trades:] if len(result.trades) > max_trades else result.trades
    pdf.section_title(f"Trade Log (last {len(trades)} of {len(result.trades)} trades)")

    headers = ["#", "Date", "Dir", "Entry", "Exit", "SL Pips", "Exit Reason", "PnL Pips", "PnL USD"]
    col_w   = [8,   26,     10,    20,      20,     14,        20,            16,          16]

    pdf.data_row(headers, col_w, is_header=True)
    y = pdf.get_y()
    pdf.set_draw_color(*GREY)
    pdf.set_line_width(0.2)
    pdf.line(MARGIN, y, PAGE_W - MARGIN, y)

    for idx, trade in enumerate(trades, start=1):
        entry_dt = trade.entry_time.strftime("%m-%d %H:%M") if trade.entry_time else ""
        entry_px = f"{trade.entry_price:.2f}"
        exit_px  = f"{trade.exit_price:.2f}" if trade.exit_price else "Open"
        sl_pips  = f"{trade.sl_pips:.1f}"
        pnl_pips = f"{trade.pnl_pips:+.1f}"
        pnl_usd  = f"{trade.pnl_usd:+.2f}"

        dir_color = GREEN if trade.direction == "BUY" else RED
        pnl_color = GREEN if trade.pnl_pips >= 0 else RED

        fill = (25, 25, 25) if idx % 2 == 1 else DARK_BG
        pdf.set_fill_color(*fill)

        cells  = [str(idx), entry_dt, trade.direction, entry_px, exit_px, sl_pips, trade.exit_reason, pnl_pips, pnl_usd]
        colors = [GREY, LIGHT_GREY, dir_color, WHITE, WHITE, GREY, GREY, pnl_color, pnl_color]

        for col, w, color in zip(cells, col_w, colors):
            pdf.set_text_color(*color)
            pdf.set_font("Helvetica", "", 6)
            pdf.cell(w, 5.5, col, border=0, fill=True)
        pdf.ln(5.5)

        if pdf.get_y() > PAGE_H - 25:
            pdf.add_page()
            pdf.set_dark_bg()
            pdf.set_y(MARGIN + 8)
            pdf.data_row(headers, col_w, is_header=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    result: BacktestResult,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a complete PDF backtest report.

    Args:
        result:      BacktestResult from run_backtest()
        output_path: Where to save the PDF. Defaults to reports/ directory.

    Returns:
        Absolute path to the saved PDF file.
    """
    os.makedirs(Config.REPORTS_DIR, exist_ok=True)

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(Config.REPORTS_DIR, f"backtest_report_{ts}.pdf")

    # ── Build chart images ─────────────────────────────────────────────
    tmp_equity  = os.path.join(Config.REPORTS_DIR, "_tmp_equity.png")
    tmp_monthly = os.path.join(Config.REPORTS_DIR, "_tmp_monthly.png")

    equity_ok  = _build_equity_chart(result, tmp_equity)
    monthly_ok = _build_monthly_chart(result, tmp_monthly)

    # ── Build PDF ──────────────────────────────────────────────────────
    logger.info("Building PDF report...")
    pdf = GoldSignalPDF()

    _cover_page(pdf, result)
    _performance_page(
        pdf, result,
        equity_chart=tmp_equity  if equity_ok  else None,
        monthly_chart=tmp_monthly if monthly_ok else None,
    )
    _monthly_table(pdf, result)
    _prop_firm_page(pdf, result)
    _trade_log_page(pdf, result)

    # ── Save ───────────────────────────────────────────────────────────
    pdf.output(output_path)
    logger.info("PDF report saved to: %s", output_path)

    # ── Clean up temp images ───────────────────────────────────────────
    for tmp in (tmp_equity, tmp_monthly):
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    from backtest.engine import (
        BacktestConfig, BacktestResult, BacktestTrade,
        MonthlyBreakdown, PropFirmSimulation, _compute_statistics,
    )
    from datetime import timedelta

    # Build a synthetic result for testing the report renderer
    cfg = BacktestConfig()
    now = datetime.now(timezone.utc)
    trades = []
    balance = cfg.account_balance

    rng = np.random.default_rng(42)
    for i in range(60):
        entry = now - timedelta(days=60 - i, hours=int(rng.integers(0, 8)))
        direction = "BUY" if rng.random() > 0.45 else "SELL"
        sl_pips = float(rng.uniform(15, 28))
        won = rng.random() > 0.4
        if won:
            pnl_pips = float(rng.uniform(sl_pips * 1.5, sl_pips * 3.5))
        else:
            pnl_pips = -sl_pips
        pnl_usd = pnl_pips * Config.GOLD_PIP_VALUE * 0.05
        balance += pnl_usd

        t = BacktestTrade(
            entry_time=entry,
            entry_price=2350.0 + rng.uniform(-50, 50),
            direction=direction,
            confidence_pct=float(rng.uniform(70, 88)),
            lot_size=0.05,
            stop_loss=2340.0,
            tp1_price=2370.0,
            tp2_price=2385.0,
            sl_pips=sl_pips,
            tp1_pips=sl_pips * 2,
            tp2_pips=sl_pips * 3,
            exit_time=entry + timedelta(hours=int(rng.integers(1, 12))),
            exit_price=2365.0 if won else 2340.0,
            exit_reason="TP2" if won else "SL",
            pnl_pips=pnl_pips,
            pnl_usd=pnl_usd,
            is_winner=won,
        )
        trades.append(t)

    result = _compute_statistics(trades, cfg)

    # Synthetic prop firm sims
    from backtest.engine import _simulate_prop_firm, PROP_FIRM_PROFILES
    for key in PROP_FIRM_PROFILES:
        sim = _simulate_prop_firm(trades, key, cfg.account_balance)
        result.prop_firm_sims.append(sim)

    path = generate_pdf_report(result)
    print(f"Test report saved: {path}")
