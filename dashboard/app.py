"""
GoldSignalAI — dashboard/app.py
=================================
Streamlit web dashboard for real-time monitoring.

Features:
  - Live signal display with formatted card
  - Compliance status (daily loss, drawdown, progress)
  - Trade history table
  - Account balance chart
  - Upcoming news events
  - ML model status

Run: streamlit run dashboard/app.py --server.port 8501
"""

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    import pandas as pd
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.info("Streamlit not installed — dashboard disabled")

from config import Config


def run_dashboard():
    """Main dashboard entry point — called by Streamlit."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Install with: pip install streamlit")
        return

    st.set_page_config(
        page_title="GoldSignalAI Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 GoldSignalAI Dashboard")
    st.caption(f"{Config.SYMBOL_DISPLAY} | {Config.PRIMARY_TIMEFRAME} + {Config.CONFIRMATION_TIMEFRAME}")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        st.text(f"Prop Firm: {Config.ACTIVE_PROP_FIRM}")
        st.text(f"Account: ${Config.CHALLENGE_ACCOUNT_SIZE:,.0f}")
        st.text(f"Risk/Trade: {Config.RISK_PER_TRADE_PCT}%")

        st.divider()
        if st.button("🔄 Refresh"):
            st.rerun()

    # ── Main Content ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        _render_compliance_section()
        _render_trade_history()

    with col2:
        _render_latest_signal()
        _render_ml_status()
        _render_news_section()


def _render_compliance_section():
    """Render compliance status and challenge progress."""
    st.subheader("📊 Compliance Status")

    try:
        from propfirm.tracker import ComplianceTracker
        from propfirm.compliance_report import generate_report_data

        tracker = ComplianceTracker()
        data = generate_report_data(tracker)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Balance", f"${data['current_balance']:,.2f}",
                   f"{data['profit_pct']:+.2f}%")
        c2.metric("Win Rate", f"{data['win_rate']:.0f}%",
                   f"{data['winning_trades']}W/{data['losing_trades']}L")
        c3.metric("Daily P/L", f"${data['daily_pnl_usd']:+,.2f}",
                   f"{data['daily_loss']['icon']}")
        c4.metric("Drawdown", f"{data['drawdown_pct']:.1f}%",
                   f"{data['drawdown']['icon']}")

        # Progress bar
        progress = data["progress"]
        st.progress(
            min(1.0, progress["progress_pct"] / 100),
            text=f"Challenge: {progress['progress_pct']:.0f}% "
                 f"({progress['current_pct']:+.2f}% / {progress['target_pct']}% target)"
        )

        # Warnings
        if data["daily_loss"]["breached"]:
            st.error("🔴 DAILY LOSS LIMIT BREACHED — Stop Trading!")
        elif data["daily_loss"]["warning"]:
            st.warning(f"🟡 Daily loss approaching limit ({data['daily_loss']['headroom_pct']:.1f}% headroom)")

        if data["drawdown"]["breached"]:
            st.error("🔴 MAX DRAWDOWN BREACHED — Challenge Failed!")
        elif data["drawdown"]["warning"]:
            st.warning(f"🟡 Drawdown approaching limit ({data['drawdown']['headroom_pct']:.1f}% headroom)")

    except Exception as exc:
        st.info(f"Compliance data not available: {exc}")


def _render_latest_signal():
    """Render the latest signal card."""
    st.subheader("📡 Latest Signal")

    try:
        signal_file = Config.SIGNAL_HISTORY_FILE
        if os.path.isfile(signal_file):
            import json
            with open(signal_file) as f:
                history = json.load(f)
            if history:
                latest = history[-1]
                st.code(latest.get("formatted", "No signal data"), language=None)
            else:
                st.info("No signals generated yet")
        else:
            st.info("No signal history file — waiting for first signal cycle")
    except Exception as exc:
        st.info(f"Signal data not available: {exc}")


def _render_trade_history():
    """Render trade history table."""
    st.subheader("📋 Trade History")

    try:
        from propfirm.tracker import ComplianceTracker
        tracker = ComplianceTracker()

        if tracker.state.trades:
            df = pd.DataFrame(tracker.state.trades)
            cols = ["timestamp", "direction", "entry_price", "exit_price",
                    "pnl_usd", "pnl_pips", "status"]
            display_cols = [c for c in cols if c in df.columns]
            st.dataframe(
                df[display_cols].tail(Config.DASHBOARD_MAX_SIGNALS),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No trades recorded yet")
    except Exception as exc:
        st.info(f"Trade history not available: {exc}")


def _render_ml_status():
    """Render ML model status."""
    st.subheader("🧠 ML Model")

    try:
        from ml.trainer import get_model_status
        status = get_model_status()

        if status["models_ready"]:
            st.success("Models loaded and ready")
            if status.get("last_training"):
                st.text(f"Last trained: {status['last_training']}")
        else:
            st.warning("Models not trained yet")
            st.text("Run: python ml/trainer.py")
    except Exception as exc:
        st.info(f"ML status not available: {exc}")


def _render_news_section():
    """Render upcoming news events."""
    st.subheader("📰 News Events")

    try:
        from data.news_fetcher import get_upcoming_events, check_news_pause

        paused, reason = check_news_pause()
        if paused:
            st.error(f"⚠️ PAUSED: {reason}")

        events = get_upcoming_events(hours_ahead=24)
        if events:
            for e in events[:5]:
                time_str = e.event_time.strftime("%H:%M UTC")
                icon = "🔴" if e.is_high_impact else "🟡"
                st.text(f"{icon} {time_str} — {e.title} ({e.currency})")
        else:
            st.info("No upcoming high-impact events")
    except Exception as exc:
        st.info(f"News data not available: {exc}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_dashboard()
    else:
        print("Install streamlit: pip install streamlit")
