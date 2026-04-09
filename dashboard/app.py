"""
GoldSignalAI — dashboard/app.py
=================================
Stage 14: Dark gold Bloomberg-style trading dashboard.

Tabs: Trade History | ML Status | Regime Detection | Challenge Progress
      Risk Monitor | Signal Heatmap

Run: venv/bin/python -m streamlit run dashboard/app.py
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta, date
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from config import Config

# ─────────────────────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

C_BG        = "#0d1117"
C_BG2       = "#161b22"
C_CARD      = "#1c2128"
C_BORDER    = "#30363d"
C_TEXT      = "#e6edf3"
C_MUTED     = "#8b949e"
C_GOLD      = "#d4a843"
C_GOLD_DIM  = "#a07830"
C_GREEN     = "#3fb950"
C_RED       = "#f85149"
C_YELLOW    = "#e3b341"
C_BLUE      = "#58a6ff"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=C_CARD,
    plot_bgcolor=C_CARD,
    font=dict(color=C_TEXT, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=[C_GOLD, C_GREEN, C_RED, C_BLUE, C_YELLOW],
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Base app overrides (highest priority) ── */
  .stApp {{
    background-color: {C_BG} !important;
    font-family: 'Inter', sans-serif !important;
  }}
  .stApp > header {{
    background-color: {C_BG} !important;
  }}
  .block-container {{
    background-color: {C_BG} !important;
    padding-top: 1rem !important;
  }}
  section[data-testid="stSidebar"] {{
    background-color: {C_BG2} !important;
    border-right: 1px solid {C_BORDER} !important;
  }}

  /* ── Root & app background ── */
  html, body, [data-testid="stAppViewContainer"] {{
    background-color: {C_BG} !important;
    color: {C_TEXT} !important;
    font-family: 'Inter', sans-serif !important;
  }}
  [data-testid="stSidebar"] {{
    background-color: {C_BG2} !important;
    border-right: 1px solid {C_BORDER};
  }}
  [data-testid="stHeader"] {{
    background-color: {C_BG} !important;
    border-bottom: 1px solid {C_BORDER};
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    background-color: {C_BG2} !important;
    border-bottom: 1px solid {C_BORDER};
    gap: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {C_MUTED} !important;
    border-radius: 0;
    padding: 8px 16px;
    font-size: 13px !important;
    font-weight: 500;
    border-bottom: 2px solid transparent;
  }}
  .stTabs [aria-selected="true"] {{
    color: {C_GOLD} !important;
    border-bottom: 2px solid {C_GOLD} !important;
    border-bottom-color: {C_GOLD} !important;
    background-color: transparent !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{
    background-color: {C_BG} !important;
    padding-top: 16px;
  }}

  /* ── Metric cards ── */
  [data-testid="stMetric"],
  div[data-testid="metric-container"] {{
    background-color: {C_CARD} !important;
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 12px 16px;
  }}
  [data-testid="stMetricLabel"] p,
  [data-testid="stMetricLabel"],
  div[data-testid="stMetricLabel"] {{
    color: {C_MUTED} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  [data-testid="stMetricValue"],
  div[data-testid="stMetricValue"] {{
    color: {C_TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
  }}
  [data-testid="stMetricDelta"],
  div[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
  }}

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {{
    background-color: {C_CARD} !important;
    border: 1px solid {C_BORDER};
    border-radius: 6px;
  }}

  /* ── Buttons ── */
  .stButton button {{
    background-color: {C_BG2} !important;
    color: {C_GOLD} !important;
    border: 1px solid {C_GOLD_DIM};
    border-radius: 4px;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500;
  }}
  .stButton button:hover {{
    background-color: {C_GOLD_DIM} !important;
    color: {C_BG} !important;
    border-color: {C_GOLD};
  }}

  /* ── Sidebar text ── */
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label {{
    color: {C_MUTED} !important;
    font-size: 12px !important;
  }}

  /* ── Info / warning / error boxes ── */
  .stAlert {{
    border-radius: 4px;
    border-left: 3px solid;
  }}

  /* ── Status badge helper ── */
  .status-badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', monospace !important;
  }}
  .badge-green  {{ background: #1e3a1e; color: {C_GREEN} !important; border: 1px solid {C_GREEN}; }}
  .badge-yellow {{ background: #2e2810; color: {C_YELLOW} !important; border: 1px solid {C_YELLOW}; }}
  .badge-red    {{ background: #3a0f0f; color: {C_RED} !important; border: 1px solid {C_RED}; }}
  .badge-gold   {{ background: #2e2010; color: {C_GOLD} !important; border: 1px solid {C_GOLD}; }}
  .badge-muted  {{ background: {C_BG2} !important; color: {C_MUTED} !important; border: 1px solid {C_BORDER}; }}

  /* ── Model card ── */
  .model-card {{
    background-color: {C_CARD} !important;
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 16px;
    height: 100%;
  }}
  .model-card h4 {{
    color: {C_GOLD} !important;
    font-size: 14px !important;
    font-weight: 600;
    margin: 0 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .model-stat {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid {C_BORDER};
    font-size: 12px !important;
  }}
  .model-stat:last-child {{ border-bottom: none; }}
  .model-stat .label {{ color: {C_MUTED} !important; }}
  .model-stat .value {{ color: {C_TEXT} !important; font-family: 'JetBrains Mono', monospace !important; }}

  /* ── Section header ── */
  .section-header {{
    color: {C_MUTED} !important;
    font-size: 10px !important;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 0 8px 0;
    border-bottom: 1px solid {C_BORDER};
    margin-bottom: 12px;
  }}

  /* ── Page title ── */
  h1 {{
    color: {C_GOLD} !important;
    font-weight: 700;
    font-size: 20px !important;
    letter-spacing: -0.02em;
  }}
  h2, h3 {{
    color: {C_TEXT} !important;
    font-weight: 600;
  }}
  h4 {{
    color: {C_MUTED} !important;
    font-weight: 500;
    font-size: 13px !important;
  }}

  /* ── Number inputs ── */
  input[type="number"], input[type="text"] {{
    background-color: {C_BG2} !important;
    color: {C_TEXT} !important;
    border: 1px solid {C_BORDER} !important;
    font-family: 'JetBrains Mono', monospace !important;
  }}

  /* ── Selectbox ── */
  [data-baseweb="select"] div {{
    background-color: {C_BG2} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
  }}

  /* ── Plotly chart container ── */
  .js-plotly-plot {{
    border-radius: 6px;
    overflow: hidden;
  }}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(Config.BASE_DIR, "database", "goldsignalai.db")


def _db_query(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT query and return list of dicts. Never crashes."""
    try:
        if not os.path.isfile(DB_PATH):
            return []
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("DB query failed: %s", exc)
        return []


def _ensure_tables() -> None:
    """Create missing tables silently."""
    try:
        if not os.path.isfile(DB_PATH):
            return
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT 'XAUUSD',
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                bullish_count INTEGER,
                bearish_count INTEGER,
                ml_confirms INTEGER,
                reason TEXT,
                is_paused INTEGER DEFAULT 0,
                forward_test INTEGER DEFAULT 0
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL DEFAULT 'XAUUSD',
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL,
                take_profit1 REAL,
                take_profit2 REAL,
                lot_size REAL,
                status TEXT NOT NULL DEFAULT 'open',
                result TEXT,
                pnl_usd REAL,
                pnl_pips REAL,
                closed_at TEXT
            )""")
        conn.commit()
        conn.close()
    except Exception:
        pass


def _load_trades_df(start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
    """Load trades from SQLite as a DataFrame."""
    rows = _db_query("SELECT * FROM trades ORDER BY timestamp ASC")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "closed_at" in df.columns:
        df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    if start_date and "timestamp" in df.columns:
        df = df[df["timestamp"].dt.date >= start_date]
    if end_date and "timestamp" in df.columns:
        df = df[df["timestamp"].dt.date <= end_date]
    return df


def _load_signals_df(start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
    """Load signals from SQLite as a DataFrame."""
    rows = _db_query("SELECT * FROM signals ORDER BY timestamp ASC")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if start_date and "timestamp" in df.columns:
        df = df[df["timestamp"].dt.date >= start_date]
    if end_date and "timestamp" in df.columns:
        df = df[df["timestamp"].dt.date <= end_date]
    return df


def _load_retrain_state() -> dict:
    """Load retrain state JSON."""
    path = Config.RETRAIN_STATE_FILE
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _get_model_info(rel_path: str) -> dict:
    """Return file existence and mtime for a model file."""
    full = os.path.join(Config.BASE_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path
    if os.path.isfile(full):
        mtime = os.path.getmtime(full)
        return {
            "exists": True,
            "path": full,
            "mtime": datetime.fromtimestamp(mtime, tz=timezone.utc),
        }
    return {"exists": False, "path": full, "mtime": None}


def _load_challenge_state() -> dict:
    """Load challenge state from JSON (ChallengeTracker persist file)."""
    path = os.path.join(Config.BASE_DIR, Config.CHALLENGE_STATE_FILE
                        if hasattr(Config, "CHALLENGE_STATE_FILE") else "state/challenge_state.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_state_manager() -> dict:
    """Load StateManager JSON."""
    path = os.path.join(Config.BASE_DIR, "state", "state.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _fmt_pct(val: float, decimals: int = 2) -> str:
    return f"{val:+.{decimals}f}%"


def _fmt_usd(val: float) -> str:
    return f"${val:,.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar and return filter state dict."""
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center; padding: 8px 0 16px 0;'>
          <span style='font-size:28px; color:{C_GOLD}; font-weight:700; letter-spacing:-0.03em;'>
            ⬡ GOLD<span style='color:{C_TEXT}'>SIGNAL</span>AI
          </span>
          <div style='color:{C_MUTED}; font-size:11px; margin-top:2px;'>
            {Config.SYMBOL_DISPLAY} · {Config.PRIMARY_TIMEFRAME}+{Config.CONFIRMATION_TIMEFRAME}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='section-header'>Account</div>", unsafe_allow_html=True)
        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=100.0,
            max_value=1_000_000.0,
            value=float(Config.CHALLENGE_ACCOUNT_SIZE),
            step=100.0,
            format="%.2f",
            label_visibility="collapsed",
        )
        st.markdown(f"<div style='color:{C_MUTED}; font-size:11px; margin-top:-8px;'>Balance: <span style='color:{C_GOLD}; font-family:JetBrains Mono'>${account_balance:,.0f}</span></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='section-header' style='margin-top:16px;'>Date Range</div>", unsafe_allow_html=True)
        default_start = date.today() - timedelta(days=90)
        start_date = st.date_input("From", value=default_start, label_visibility="collapsed")
        end_date   = st.date_input("To",   value=date.today(),   label_visibility="collapsed")

        st.markdown(f"<div class='section-header' style='margin-top:16px;'>Filters</div>", unsafe_allow_html=True)
        direction_filter = st.selectbox("Direction", ["All", "BUY", "SELL"], label_visibility="collapsed")

        st.markdown(f"<div class='section-header' style='margin-top:16px;'>Bot Status</div>", unsafe_allow_html=True)
        # Infer status from challenge state
        cstate = _load_challenge_state()
        if cstate.get("breach_halted"):
            badge = f"<span class='status-badge badge-red'>HALTED</span>"
        elif Config.CHALLENGE_MODE_ENABLED:
            badge = f"<span class='status-badge badge-gold'>CHALLENGE MODE</span>"
        else:
            badge = f"<span class='status-badge badge-green'>RUNNING</span>"
        st.markdown(badge, unsafe_allow_html=True)

        st.markdown(f"<div class='section-header' style='margin-top:16px;'>System</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:11px; color:{C_MUTED};'>
          Prop Firm: <span style='color:{C_TEXT};'>{Config.ACTIVE_PROP_FIRM}</span><br>
          Risk/Trade: <span style='color:{C_TEXT};'>{Config.RISK_PER_TRADE_PCT}%</span><br>
          LGBM Filter: <span style='color:{"#3fb950" if Config.USE_LGBM_FILTER else C_MUTED};'>
            {"ON" if Config.USE_LGBM_FILTER else "OFF"}</span><br>
          Deep Filter: <span style='color:{"#3fb950" if Config.USE_DEEP_FILTER else C_MUTED};'>
            {"ON" if Config.USE_DEEP_FILTER else "OFF"}</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        st.markdown(f"<div style='font-size:10px; color:{C_MUTED};'>Updated: {now_utc}</div>", unsafe_allow_html=True)
        if st.button("↺ Refresh", use_container_width=True):
            st.rerun()

        # Auto-refresh via streamlit-autorefresh if available
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60_000, key="auto_refresh")
        except ImportError:
            pass

    return {
        "account_balance": account_balance,
        "start_date": start_date,
        "end_date": end_date,
        "direction": direction_filter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def tab_trade_history(filters: dict) -> None:
    trades_df = _load_trades_df(filters["start_date"], filters["end_date"])

    # Also pull from ComplianceTracker as fallback
    if trades_df.empty:
        try:
            from propfirm.tracker import ComplianceTracker
            tracker = ComplianceTracker()
            if tracker.state.trades:
                trades_df = pd.DataFrame(tracker.state.trades)
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True, errors="coerce")
        except Exception:
            pass

    if trades_df.empty:
        st.markdown(f"""
        <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
                    padding:40px; text-align:center; color:{C_MUTED};'>
          <div style='font-size:32px; margin-bottom:8px;'>📋</div>
          <div style='font-size:14px;'>No trades recorded yet.</div>
          <div style='font-size:12px; margin-top:4px;'>Trades appear here once the bot executes signals.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Direction filter ──────────────────────────────────────────────────────
    if filters["direction"] != "All" and "direction" in trades_df.columns:
        trades_df = trades_df[trades_df["direction"] == filters["direction"]]

    closed = trades_df[trades_df.get("status", pd.Series(dtype=str)) == "closed"] \
        if "status" in trades_df.columns else trades_df
    if "pnl_usd" in trades_df.columns:
        closed = trades_df.dropna(subset=["pnl_usd"])

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_trades = len(trades_df)
    if not closed.empty and "pnl_usd" in closed.columns:
        wins     = (closed["pnl_usd"] > 0).sum()
        losses   = (closed["pnl_usd"] < 0).sum()
        win_rate = wins / max(1, wins + losses) * 100
        total_pnl = closed["pnl_usd"].sum()
        gross_win  = closed[closed["pnl_usd"] > 0]["pnl_usd"].sum()
        gross_loss = abs(closed[closed["pnl_usd"] < 0]["pnl_usd"].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_win  = closed[closed["pnl_usd"] > 0]["pnl_usd"].mean() if wins > 0 else 0
        avg_loss = closed[closed["pnl_usd"] < 0]["pnl_usd"].mean() if losses > 0 else 0
    else:
        wins = losses = 0
        win_rate = total_pnl = pf = avg_win = avg_loss = 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades", f"{total_trades}")
    c2.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}W / {losses}L")
    c3.metric("Profit Factor", f"{pf:.2f}" if pf != float('inf') else "∞")
    c4.metric("Total PnL", _fmt_usd(total_pnl), delta_color="normal")
    c5.metric("Avg Win / Loss", f"${avg_win:.0f} / ${avg_loss:.0f}")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Equity curve ─────────────────────────────────────────────────────────
    if not closed.empty and "pnl_usd" in closed.columns:
        eq_df = closed.copy().sort_values("timestamp")
        eq_df["cumulative_pnl"] = eq_df["pnl_usd"].cumsum()
        eq_df["balance"] = filters["account_balance"] + eq_df["cumulative_pnl"]
        eq_df["color"] = eq_df["pnl_usd"].apply(lambda x: C_GREEN if x >= 0 else C_RED)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["timestamp"],
            y=eq_df["balance"],
            mode="lines",
            name="Balance",
            line=dict(color=C_GOLD, width=2),
            fill="tozeroy",
            fillcolor=f"rgba(212,168,67,0.07)",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Balance: $%{y:,.2f}<extra></extra>",
        ))
        fig.add_hline(
            y=filters["account_balance"],
            line_dash="dot",
            line_color=C_MUTED,
            annotation_text="Starting Balance",
            annotation_font_color=C_MUTED,
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Equity Curve", font=dict(color=C_GOLD, size=13)),
            height=280,
            xaxis=dict(gridcolor=C_BORDER, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=C_BORDER, tickfont=dict(size=10), tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Trade table ───────────────────────────────────────────────────────────
    display_cols = [c for c in ["timestamp", "direction", "entry_price", "result",
                                 "pnl_usd", "pnl_pips", "status", "closed_at"]
                    if c in trades_df.columns]
    display_df = trades_df[display_cols].copy().sort_values("timestamp", ascending=False)

    if "pnl_usd" in display_df.columns:
        display_df["pnl_usd"] = display_df["pnl_usd"].apply(
            lambda v: f"+${v:.2f}" if pd.notna(v) and v >= 0 else (f"-${abs(v):.2f}" if pd.notna(v) else "—")
        )
    if "timestamp" in display_df.columns:
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(
        display_df.head(Config.DASHBOARD_MAX_SIGNALS),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML STATUS
# ─────────────────────────────────────────────────────────────────────────────

def tab_ml_status() -> None:
    retrain = _load_retrain_state()

    lgbm_info  = _get_model_info("models/lgbm_direction.pkl")
    hmm_info   = _get_model_info("models/hmm_regime.pkl")
    deep_info  = _get_model_info("models/deep_model.keras")

    lgbm_state = retrain.get("lgbm", {})
    deep_state = retrain.get("deep", {})

    def _fmt_date(dt_obj):
        if dt_obj is None:
            return "Never"
        if isinstance(dt_obj, str):
            try:
                dt_obj = datetime.fromisoformat(dt_obj)
            except Exception:
                return dt_obj
        return dt_obj.strftime("%Y-%m-%d %H:%M")

    def _pct_badge(val, gate):
        if val is None:
            return f"<span class='status-badge badge-muted'>N/A</span>"
        pct = val * 100 if val < 1 else val
        color = "badge-green" if pct >= gate * 100 else "badge-yellow"
        return f"<span class='status-badge {color}'>{pct:.1f}%</span>"

    def _bool_badge(val, true_label="ENABLED", false_label="DISABLED"):
        color = "badge-green" if val else "badge-muted"
        label = true_label if val else false_label
        return f"<span class='status-badge {color}'>{label}</span>"

    col1, col2, col3 = st.columns(3)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    with col1:
        last_acc  = lgbm_state.get("last_accuracy")
        gate_met  = last_acc is not None and last_acc >= Config.RETRAIN_LGBM_MIN_ACCURACY
        mtime_str = _fmt_date(lgbm_info["mtime"])
        retrain_cnt = lgbm_state.get("retrain_count", 0)

        st.markdown(f"""
        <div class='model-card'>
          <h4>⚡ LightGBM</h4>
          <div class='model-stat'>
            <span class='label'>Status</span>
            <span class='value'>{_bool_badge(Config.USE_LGBM_FILTER)}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>File</span>
            <span class='value'>{_bool_badge(lgbm_info["exists"], "FOUND", "MISSING")}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>CV Accuracy</span>
            <span class='value'>{_pct_badge(last_acc, Config.RETRAIN_LGBM_MIN_ACCURACY)}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Gate</span>
            <span class='value'>{_bool_badge(gate_met, f"MET ≥{Config.RETRAIN_LGBM_MIN_ACCURACY*100:.0f}%", f"NOT MET &lt;{Config.RETRAIN_LGBM_MIN_ACCURACY*100:.0f}%")}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Last Trained</span>
            <span class='value' style='font-size:11px;'>{mtime_str}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Retrain Count</span>
            <span class='value'>{retrain_cnt}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Role</span>
            <span class='value' style='color:{C_MUTED}; font-size:11px;'>Direction classifier</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Note</span>
            <span class='value' style='color:{C_MUTED}; font-size:10px;'>Meta-decision soft vote always active</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── HMM ──────────────────────────────────────────────────────────────────
    with col2:
        mtime_str = _fmt_date(hmm_info["mtime"])
        st.markdown(f"""
        <div class='model-card'>
          <h4>🌀 HMM Regime</h4>
          <div class='model-stat'>
            <span class='label'>Status</span>
            <span class='value'>{_bool_badge(True, "ACTIVE", "INACTIVE")}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>File</span>
            <span class='value'>{_bool_badge(hmm_info["exists"], "FOUND", "MISSING")}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Architecture</span>
            <span class='value' style='font-size:11px;'>GaussianHMM 3-state</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Timeframe</span>
            <span class='value'>H1</span>
          </div>
          <div class='model-stat'>
            <span class='label'>States</span>
            <span class='value' style='font-size:11px;'>TRENDING / RANGING / CRISIS</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Last Trained</span>
            <span class='value' style='font-size:11px;'>{mtime_str}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Role</span>
            <span class='value' style='color:{C_MUTED}; font-size:11px;'>Hard gate (Rule 1)</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Note</span>
            <span class='value' style='color:{C_MUTED}; font-size:10px;'>CRISIS blocks all; RANGING halves size</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── CNN-BiLSTM ────────────────────────────────────────────────────────────
    with col3:
        last_acc_deep = deep_state.get("last_accuracy")
        gate_met_deep = last_acc_deep is not None and last_acc_deep >= Config.DEEP_ACCURACY_GATE
        mtime_str = _fmt_date(deep_info["mtime"])
        retrain_cnt_deep = deep_state.get("retrain_count", 0)
        outcomes = deep_state.get("trade_outcomes_available", 0)
        min_outcomes = Config.RETRAIN_DEEP_MIN_TRADES if hasattr(Config, "RETRAIN_DEEP_MIN_TRADES") else 150

        st.markdown(f"""
        <div class='model-card'>
          <h4>🧠 CNN-BiLSTM</h4>
          <div class='model-stat'>
            <span class='label'>Status</span>
            <span class='value'>{_bool_badge(Config.USE_DEEP_FILTER)}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>File</span>
            <span class='value'>{_bool_badge(deep_info["exists"], "FOUND", "MISSING")}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Val Accuracy</span>
            <span class='value'>{_pct_badge(last_acc_deep, Config.DEEP_ACCURACY_GATE)}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Gate</span>
            <span class='value'>{_bool_badge(gate_met_deep, f"MET ≥{Config.DEEP_ACCURACY_GATE*100:.0f}%", f"NOT MET &lt;{Config.DEEP_ACCURACY_GATE*100:.0f}%")}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Last Trained</span>
            <span class='value' style='font-size:11px;'>{mtime_str}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Retrain Count</span>
            <span class='value'>{retrain_cnt_deep}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Live Outcomes</span>
            <span class='value'>{outcomes} / {min_outcomes}</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Note</span>
            <span class='value' style='color:{C_MUTED}; font-size:10px;'>UP bias noted; retrain after {min_outcomes} trades</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Retrain schedule info ─────────────────────────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-header'>Auto-Retraining Schedule</div>", unsafe_allow_html=True)

    last_lgbm_retrain = lgbm_state.get("last_retrain", "Never")
    if last_lgbm_retrain and last_lgbm_retrain != "Never":
        try:
            last_dt = datetime.fromisoformat(last_lgbm_retrain)
            next_dt = last_dt + timedelta(days=Config.RETRAIN_LGBM_INTERVAL_DAYS)
            next_str = next_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            next_str = "Unknown"
    else:
        next_str = "Next Sunday 02:00 UTC"

    r1, r2, r3 = st.columns(3)
    r1.metric("LGBM Schedule", f"Every {Config.RETRAIN_LGBM_INTERVAL_DAYS}d", "Sunday 02:00 UTC")
    r2.metric("Next LGBM Retrain", next_str)
    r3.metric("CNN-BiLSTM Trigger", f"{outcomes}/{min_outcomes} live outcomes",
              "Retrain unlocked" if outcomes >= min_outcomes else f"Need {min_outcomes - outcomes} more")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def tab_regime_detection() -> None:
    hmm_info = _get_model_info("models/hmm_regime.pkl")

    # Try to get current regime from signals reason field
    signals = _db_query(
        "SELECT timestamp, direction, reason FROM signals ORDER BY timestamp DESC LIMIT 200"
    )

    regime_from_signals = []
    for sig in signals:
        reason = (sig.get("reason") or "").upper()
        ts = sig.get("timestamp", "")
        if "TRENDING" in reason:
            regime_from_signals.append({"ts": ts, "regime": "TRENDING", "raw": reason})
        elif "RANGING" in reason:
            regime_from_signals.append({"ts": ts, "regime": "RANGING", "raw": reason})
        elif "CRISIS" in reason:
            regime_from_signals.append({"ts": ts, "regime": "CRISIS", "raw": reason})

    current_regime = regime_from_signals[0]["regime"] if regime_from_signals else None

    # ── Current regime badge ──────────────────────────────────────────────────
    regime_color = {
        "TRENDING": C_GREEN,
        "RANGING":  C_YELLOW,
        "CRISIS":   C_RED,
    }
    regime_desc = {
        "TRENDING": "Strong directional trend — full position sizing active",
        "RANGING":  "Sideways / low conviction — position size halved",
        "CRISIS":   "High volatility / crisis mode — all signals blocked",
    }

    c1, c2 = st.columns([1, 2])
    with c1:
        if current_regime:
            color = regime_color.get(current_regime, C_MUTED)
            desc  = regime_desc.get(current_regime, "")
            st.markdown(f"""
            <div style='background:{C_CARD}; border:1px solid {color}; border-radius:8px;
                        padding:24px; text-align:center;'>
              <div style='font-size:11px; color:{C_MUTED}; text-transform:uppercase;
                          letter-spacing:0.1em; margin-bottom:8px;'>Current Regime</div>
              <div style='font-size:36px; font-weight:700; color:{color};
                          font-family:JetBrains Mono; letter-spacing:0.02em;'>
                {current_regime}
              </div>
              <div style='font-size:11px; color:{C_MUTED}; margin-top:8px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
                        padding:24px; text-align:center;'>
              <div style='font-size:11px; color:{C_MUTED}; text-transform:uppercase;
                          letter-spacing:0.1em; margin-bottom:8px;'>Current Regime</div>
              <div style='font-size:24px; font-weight:700; color:{C_MUTED};'>UNKNOWN</div>
              <div style='font-size:11px; color:{C_MUTED}; margin-top:8px;'>
                No signal data yet — regime determined at each signal cycle
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        model_color = C_GREEN if hmm_info["exists"] else C_RED
        mtime = hmm_info["mtime"].strftime("%Y-%m-%d") if hmm_info["mtime"] else "—"
        st.markdown(f"""
        <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; padding:16px;'>
          <div class='section-header'>HMM Model</div>
          <div class='model-stat'>
            <span class='label'>File</span>
            <span style='color:{model_color}; font-family:JetBrains Mono; font-size:12px;'>
              {"LOADED" if hmm_info["exists"] else "MISSING"}
            </span>
          </div>
          <div class='model-stat'>
            <span class='label'>Type</span>
            <span class='value'>GaussianHMM 3-state</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Features</span>
            <span class='value'>Log returns + realized vol</span>
          </div>
          <div class='model-stat'>
            <span class='label'>Last trained</span>
            <span class='value'>{mtime}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # State distribution pie from signal history
        if regime_from_signals:
            regime_counts = {}
            for r in regime_from_signals:
                regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1

            labels = list(regime_counts.keys())
            values = list(regime_counts.values())
            colors = [regime_color.get(lbl, C_MUTED) for lbl in labels]

            fig_pie = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color=C_BG, width=2)),
                textinfo="label+percent",
                textfont=dict(color=C_TEXT, size=12, family="Inter"),
                hovertemplate="<b>%{label}</b><br>%{value} signals (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"State Distribution (last {len(regime_from_signals)} signals)",
                           font=dict(color=C_GOLD, size=13)),
                height=300,
                showlegend=True,
                legend=dict(font=dict(color=C_TEXT, size=11)),
                annotations=[dict(
                    text=f"<b>{current_regime or '?'}</b>",
                    x=0.5, y=0.5,
                    font=dict(size=16, color=regime_color.get(current_regime, C_MUTED)),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(f"""
            <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
                        padding:40px; text-align:center; color:{C_MUTED}; height:300px;
                        display:flex; align-items:center; justify-content:center;'>
              <div>
                <div style='font-size:28px; margin-bottom:8px;'>🌀</div>
                <div style='font-size:13px;'>No regime history available.</div>
                <div style='font-size:11px; margin-top:4px;'>Run main.py to populate signal history.</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 7-day regime timeline ─────────────────────────────────────────────────
    if regime_from_signals:
        st.markdown(f"<div class='section-header' style='margin-top:16px;'>Recent Regime History</div>", unsafe_allow_html=True)
        timeline_df = pd.DataFrame(regime_from_signals[:50])
        timeline_df["ts"] = pd.to_datetime(timeline_df["ts"], utc=True, errors="coerce")
        timeline_df = timeline_df.dropna(subset=["ts"]).sort_values("ts")

        fig_time = go.Figure()
        for regime, color in regime_color.items():
            subset = timeline_df[timeline_df["regime"] == regime]
            if not subset.empty:
                fig_time.add_trace(go.Scatter(
                    x=subset["ts"],
                    y=[regime] * len(subset),
                    mode="markers",
                    name=regime,
                    marker=dict(color=color, size=10, symbol="square"),
                    hovertemplate=f"<b>{regime}</b><br>%{{x|%Y-%m-%d %H:%M}}<extra></extra>",
                ))
        fig_time.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Regime Timeline", font=dict(color=C_GOLD, size=13)),
            height=200,
            yaxis=dict(
                categoryorder="array",
                categoryarray=["CRISIS", "RANGING", "TRENDING"],
                tickfont=dict(size=11),
                gridcolor=C_BORDER,
            ),
            xaxis=dict(gridcolor=C_BORDER, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CHALLENGE PROGRESS
# ─────────────────────────────────────────────────────────────────────────────

def tab_challenge_progress(account_balance: float) -> None:
    # Try ChallengeTracker first, then ComplianceTracker as fallback
    status: Optional[dict] = None

    try:
        from propfirm.tracker import ChallengeTracker
        ct = ChallengeTracker(Config.ACTIVE_PROP_FIRM, account_balance)
        state_path = os.path.join(
            Config.BASE_DIR,
            getattr(Config, "CHALLENGE_STATE_FILE", "state/challenge_state.json"),
        )
        ct.load(state_path)
        ct.update_balance(ct.current_balance, datetime.now(timezone.utc))
        status = ct.get_status()
    except Exception:
        pass

    if status is None:
        try:
            from propfirm.tracker import ComplianceTracker
            from propfirm.compliance_report import generate_report_data
            comp = ComplianceTracker()
            data = generate_report_data(comp)
            # Map to unified format
            status = {
                "current_balance":    data["current_balance"],
                "initial_balance":    data["starting_balance"],
                "profit_pct":         data["profit_pct"],
                "daily_loss_pct":     data["daily_loss"]["current_pct"],
                "total_dd_pct":       data["drawdown_pct"],
                "daily_limit_pct":    data["daily_loss"]["limit_pct"],
                "total_dd_limit_pct": data["drawdown"]["limit_pct"],
                "daily_warning_pct":  data["daily_loss"]["limit_pct"] * 0.83,
                "total_dd_warning_pct": data["drawdown"]["limit_pct"] * 0.83,
                "profit_target_pct":  data["progress"]["target_pct"],
                "profit_progress_pct": data["progress"]["progress_pct"],
                "target_amount":      data["starting_balance"] * (1 + data["progress"]["target_pct"] / 100),
                "target_met":         data["progress"]["target_met"],
                "compliance_status":  "OK",
                "pause_reason":       None,
                "daily_loss_dollars": data["daily_pnl_usd"],
                "daily_remaining_dollars": 0,
                "total_dd_dollars":   data["drawdown_pct"] * data["starting_balance"] / 100,
                "total_dd_remaining_dollars": 0,
            }
        except Exception:
            pass

    if status is None:
        # Fallback: use defaults with zero values
        from propfirm.profiles import get_profile
        try:
            prof = get_profile(Config.ACTIVE_PROP_FIRM)
        except Exception:
            prof = None

        status = {
            "current_balance":    account_balance,
            "initial_balance":    account_balance,
            "profit_pct":         0.0,
            "daily_loss_pct":     0.0,
            "total_dd_pct":       0.0,
            "daily_limit_pct":    3.0 if prof is None else prof.daily_loss_limit,
            "total_dd_limit_pct": 6.0 if prof is None else prof.max_total_drawdown,
            "daily_warning_pct":  2.5 if prof is None else prof.daily_loss_warning,
            "total_dd_warning_pct": 5.0 if prof is None else prof.total_drawdown_warning,
            "profit_target_pct":  10.0 if prof is None else prof.profit_target,
            "profit_progress_pct": 0.0,
            "target_amount":      account_balance * 1.10,
            "target_met":         False,
            "compliance_status":  "OK",
            "pause_reason":       None,
            "daily_loss_dollars": 0,
            "daily_remaining_dollars": account_balance * 0.03,
            "total_dd_dollars":   0,
            "total_dd_remaining_dollars": account_balance * 0.06,
        }

    # ── Compliance status banner ──────────────────────────────────────────────
    cs = status.get("compliance_status", "OK")
    if cs == "BREACHED":
        st.error(f"🔴 TRADING HALTED — {status.get('pause_reason', 'Limit breached')}")
    elif cs == "PAUSED":
        st.warning(f"⚠️ APPROACHING LIMIT — {status.get('pause_reason', '')}")
    else:
        st.success("✅ Challenge status: ON TRACK")

    # ── Top-level balance metrics ─────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Current Balance",
        _fmt_usd(status["current_balance"]),
        _fmt_pct(status["profit_pct"]),
    )
    m2.metric(
        "Profit Target",
        _fmt_usd(status["target_amount"]),
        f"{status['profit_target_pct']:.0f}% required",
    )
    m3.metric(
        "Today's Loss",
        _fmt_usd(status.get("daily_loss_dollars", 0)),
        f"Limit: {status['daily_limit_pct']:.1f}%",
    )
    m4.metric(
        "Total Drawdown",
        f"{status['total_dd_pct']:.2f}%",
        f"Limit: {status['total_dd_limit_pct']:.1f}%",
    )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Four gauges ───────────────────────────────────────────────────────────
    def _gauge_color(current, limit):
        ratio = current / limit if limit > 0 else 0
        if ratio < 0.5:
            return C_GREEN
        if ratio < 0.8:
            return C_YELLOW
        return C_RED

    def make_gauge(title, current, limit, unit="%", prefix="", warning=None):
        bar_color = _gauge_color(current, limit)
        steps = [
            {"range": [0, limit * 0.5],  "color": "#1a2a1a"},
            {"range": [limit * 0.5, limit * 0.8], "color": "#2a2510"},
            {"range": [limit * 0.8, limit],       "color": "#2a1010"},
        ]
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current,
            number=dict(
                suffix=unit,
                prefix=prefix,
                font=dict(color=C_TEXT, family="JetBrains Mono, monospace", size=22),
            ),
            delta=dict(
                reference=warning if warning else limit * 0.8,
                increasing=dict(color=C_RED),
                decreasing=dict(color=C_GREEN),
                font=dict(size=11),
            ),
            domain={"x": [0, 1], "y": [0, 1]},
            title=dict(text=title, font=dict(color=C_MUTED, size=12, family="Inter")),
            gauge=dict(
                axis=dict(
                    range=[0, limit],
                    tickcolor=C_MUTED,
                    tickfont=dict(size=9, color=C_MUTED),
                    nticks=5,
                ),
                bar=dict(color=bar_color, thickness=0.6),
                bgcolor=C_CARD,
                borderwidth=1,
                bordercolor=C_BORDER,
                steps=steps,
                threshold=dict(
                    line=dict(color=C_RED, width=2),
                    thickness=0.75,
                    value=limit,
                ),
            ),
        ))
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=60, b=10))
        return fig

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        fig = make_gauge(
            "PnL Progress",
            max(0, status["profit_pct"]),
            status["profit_target_pct"],
            unit="%",
            warning=status["profit_target_pct"] * 0.8,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div style='text-align:center; font-size:11px; color:{C_MUTED};'>"
            f"{status['profit_progress_pct']:.1f}% of target reached</div>",
            unsafe_allow_html=True,
        )

    with g2:
        fig = make_gauge(
            "Daily Loss",
            status["daily_loss_pct"],
            status["daily_limit_pct"],
            unit="%",
            warning=status["daily_warning_pct"],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div style='text-align:center; font-size:11px; color:{C_MUTED};'>"
            f"Remaining: ${status.get('daily_remaining_dollars', 0):,.0f}</div>",
            unsafe_allow_html=True,
        )

    with g3:
        fig = make_gauge(
            "Total Drawdown",
            status["total_dd_pct"],
            status["total_dd_limit_pct"],
            unit="%",
            warning=status["total_dd_warning_pct"],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div style='text-align:center; font-size:11px; color:{C_MUTED};'>"
            f"Buffer: ${status.get('total_dd_remaining_dollars', 0):,.0f}</div>",
            unsafe_allow_html=True,
        )

    with g4:
        challenge_days = 0
        try:
            from propfirm.tracker import ComplianceTracker
            comp2 = ComplianceTracker()
            challenge_days = comp2.state.trading_days
        except Exception:
            pass

        from propfirm.profiles import get_profile
        try:
            prof = get_profile(Config.ACTIVE_PROP_FIRM)
            min_days = prof.min_trading_days
        except Exception:
            min_days = 0

        max_days = max(30, min_days, challenge_days + 5)
        fig = make_gauge(
            "Trading Days",
            challenge_days,
            max_days,
            unit="d",
            warning=min_days if min_days > 0 else max_days * 0.8,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        days_label = f"Min required: {min_days}d" if min_days > 0 else "No minimum days"
        st.markdown(
            f"<div style='text-align:center; font-size:11px; color:{C_MUTED};'>{days_label}</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — RISK MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def tab_risk_monitor() -> None:
    state_data = _load_state_manager()
    session_losses = state_data.get("session_consecutive_losses", 0)
    session_date   = state_data.get("session_date", "—")

    # Today's trades from DB
    today_str = date.today().isoformat()
    today_trades = _db_query(
        "SELECT * FROM trades WHERE timestamp LIKE ? ORDER BY timestamp DESC",
        (f"{today_str}%",),
    )
    today_count = len(today_trades)
    today_wins  = sum(1 for t in today_trades if (t.get("pnl_usd") or 0) > 0)
    today_losses = sum(1 for t in today_trades if (t.get("pnl_usd") or 0) < 0)

    # Circuit breaker level from state
    max_sess_loss = getattr(Config, "META_MAX_SESSION_LOSS", 2)
    cb_level = 0 if session_losses == 0 else (
        2 if session_losses >= max_sess_loss else 1
    )
    cb_labels = {0: ("NONE", C_GREEN), 1: ("WARNING", C_YELLOW), 2: ("HALTED", C_RED)}
    cb_text, _ = cb_labels[cb_level]

    # Last signal info from DB
    last_sigs = _db_query(
        "SELECT * FROM signals WHERE direction != 'WAIT' ORDER BY timestamp DESC LIMIT 1"
    )
    last_sig = last_sigs[0] if last_sigs else None

    # ── Row 1: Key risk metrics ───────────────────────────────────────────────
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        badge_html = f"<span class='status-badge badge-{'green' if cb_level==0 else 'yellow' if cb_level==1 else 'red'}'>{cb_text}</span>"
        st.markdown(f"""
        <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; padding:16px;'>
          <div style='font-size:10px; color:{C_MUTED}; text-transform:uppercase;
                      letter-spacing:0.1em; margin-bottom:8px;'>Circuit Breaker</div>
          {badge_html}
          <div style='font-size:11px; color:{C_MUTED}; margin-top:8px;'>
            Level {cb_level} of 2
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        loss_color = C_GREEN if session_losses == 0 else (C_YELLOW if session_losses < max_sess_loss else C_RED)
        st.markdown(f"""
        <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; padding:16px;'>
          <div style='font-size:10px; color:{C_MUTED}; text-transform:uppercase;
                      letter-spacing:0.1em; margin-bottom:8px;'>Session Losses</div>
          <div style='font-size:32px; font-weight:700; color:{loss_color};
                      font-family:JetBrains Mono;'>
            {session_losses} <span style='font-size:14px; color:{C_MUTED};'>/ {max_sess_loss}</span>
          </div>
          <div style='font-size:11px; color:{C_MUTED}; margin-top:4px;'>{session_date}</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; padding:16px;'>
          <div style='font-size:10px; color:{C_MUTED}; text-transform:uppercase;
                      letter-spacing:0.1em; margin-bottom:8px;'>Today's Trades</div>
          <div style='font-size:32px; font-weight:700; color:{C_TEXT};
                      font-family:JetBrains Mono;'>{today_count}</div>
          <div style='font-size:11px; margin-top:4px;'>
            <span style='color:{C_GREEN};'>{today_wins}W</span>
            <span style='color:{C_MUTED};'> / </span>
            <span style='color:{C_RED};'>{today_losses}L</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        if last_sig:
            sig_dir    = last_sig.get("direction", "—")
            sig_conf   = last_sig.get("confidence", 0)
            sig_ts_raw = last_sig.get("timestamp", "—")
            try:
                sig_ts = datetime.fromisoformat(sig_ts_raw).strftime("%Y-%m-%d %H:%M")
            except Exception:
                sig_ts = sig_ts_raw
            dir_color = C_GREEN if sig_dir == "BUY" else (C_RED if sig_dir == "SELL" else C_MUTED)
            st.markdown(f"""
            <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; padding:16px;'>
              <div style='font-size:10px; color:{C_MUTED}; text-transform:uppercase;
                          letter-spacing:0.1em; margin-bottom:8px;'>Last Signal</div>
              <div style='font-size:18px; font-weight:700; color:{dir_color};
                          font-family:JetBrains Mono;'>{sig_dir}</div>
              <div style='font-size:11px; color:{C_MUTED}; margin-top:4px;'>
                Conf: <span style='color:{C_TEXT};'>{sig_conf:.1f}%</span><br>
                {sig_ts}
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; padding:16px;'>
              <div style='font-size:10px; color:{C_MUTED}; text-transform:uppercase;
                          letter-spacing:0.1em; margin-bottom:8px;'>Last Signal</div>
              <div style='font-size:14px; color:{C_MUTED};'>No signals yet</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── News filter status ────────────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>News & Volatility Filter</div>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)

    news_enabled = getattr(Config, "NEWS_FILTER_ENABLED", True)
    n1.metric("News Filter", "ENABLED" if news_enabled else "DISABLED")
    n2.metric("Pre-Event Block", f"{getattr(Config, 'NEWS_HIGH_IMPACT_PRE_MIN', 30)} min")
    n3.metric("ATR Spike Block", f"{getattr(Config, 'NEWS_ATR_SPIKE_BLOCK', 2.0)}×")

    try:
        from data.news_fetcher import get_upcoming_events, check_news_pause
        paused, reason = check_news_pause()
        if paused:
            st.warning(f"⚠️ NEWS PAUSE ACTIVE: {reason}")

        events = get_upcoming_events(hours_ahead=24)
        if events:
            st.markdown(f"<div class='section-header' style='margin-top:16px;'>Upcoming Events (24h)</div>",
                        unsafe_allow_html=True)
            for e in events[:8]:
                time_str = e.event_time.strftime("%H:%M UTC") if hasattr(e, "event_time") else "—"
                impact_icon = "🔴" if e.is_high_impact else "🟡"
                title = getattr(e, "title", str(e))
                currency = getattr(e, "currency", "")
                st.markdown(
                    f"<div style='font-size:12px; padding:4px 0; border-bottom:1px solid {C_BORDER};'>"
                    f"{impact_icon} <span style='color:{C_MUTED};'>{time_str}</span> — "
                    f"<span style='color:{C_TEXT};'>{title}</span> "
                    f"<span style='color:{C_MUTED};'>({currency})</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No upcoming high-impact events in next 24h")
    except Exception as exc:
        st.info(f"News data unavailable: {exc}")

    # ── Recent signals table ──────────────────────────────────────────────────
    recent_sigs = _db_query(
        "SELECT timestamp, direction, confidence, entry_price, reason FROM signals "
        "ORDER BY timestamp DESC LIMIT 20"
    )
    if recent_sigs:
        st.markdown(f"<div class='section-header' style='margin-top:20px;'>Recent Signals</div>",
                    unsafe_allow_html=True)
        sig_df = pd.DataFrame(recent_sigs)
        if "timestamp" in sig_df.columns:
            sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"], utc=True, errors="coerce")
            sig_df["timestamp"] = sig_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        if "confidence" in sig_df.columns:
            sig_df["confidence"] = sig_df["confidence"].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        st.dataframe(sig_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — SIGNAL HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def tab_signal_heatmap(filters: dict) -> None:
    trades_df = _load_trades_df(filters["start_date"], filters["end_date"])

    # Fallback to ComplianceTracker
    if trades_df.empty:
        try:
            from propfirm.tracker import ComplianceTracker
            tracker = ComplianceTracker()
            if tracker.state.trades:
                trades_df = pd.DataFrame(tracker.state.trades)
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True, errors="coerce")
        except Exception:
            pass

    if trades_df.empty or "pnl_usd" not in trades_df.columns:
        st.markdown(f"""
        <div style='background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px;
                    padding:40px; text-align:center; color:{C_MUTED};'>
          <div style='font-size:32px; margin-bottom:8px;'>📊</div>
          <div style='font-size:14px;'>No trade data available for heatmap analysis.</div>
          <div style='font-size:12px; margin-top:4px;'>Trade history populates after first live trades.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = trades_df.copy()
    df = df.dropna(subset=["pnl_usd", "timestamp"])
    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["win"]     = (df["pnl_usd"] > 0).astype(int)

    # ── Win rate by hour ──────────────────────────────────────────────────────
    hour_stats = (
        df.groupby("hour")
        .agg(win_rate=("win", "mean"), trades=("win", "count"), pnl=("pnl_usd", "sum"))
        .reset_index()
    )
    hour_stats["win_rate_pct"] = hour_stats["win_rate"] * 100

    # Color: highlight NY session (13-21 UTC)
    hour_stats["is_ny"] = hour_stats["hour"].between(13, 21)
    hour_stats["bar_color"] = hour_stats.apply(
        lambda r: C_GOLD if r["is_ny"] else C_MUTED, axis=1
    )

    # Overlay win rate color intensity
    def _wr_color(wr):
        if wr >= 60:
            return C_GREEN
        if wr >= 40:
            return C_YELLOW
        return C_RED

    hour_stats["color"] = hour_stats["win_rate_pct"].apply(_wr_color)

    col1, col2 = st.columns(2)
    with col1:
        fig_hour = go.Figure(go.Bar(
            x=hour_stats["hour"],
            y=hour_stats["win_rate_pct"],
            marker_color=hour_stats["color"],
            text=hour_stats["trades"].apply(lambda n: f"{n}t"),
            textposition="outside",
            textfont=dict(size=9, color=C_MUTED),
            hovertemplate=(
                "<b>Hour %{x}:00 UTC</b><br>"
                "Win Rate: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
        # NY session shading
        fig_hour.add_vrect(x0=12.5, x1=21.5, fillcolor=C_GOLD, opacity=0.05,
                           line_width=0, annotation_text="NY Session",
                           annotation_position="top left",
                           annotation_font_color=C_GOLD_DIM,
                           annotation_font_size=9)
        fig_hour.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Win Rate by Hour (UTC)", font=dict(color=C_GOLD, size=13)),
            height=300,
            xaxis=dict(
                tickvals=list(range(0, 24)),
                ticktext=[f"{h:02d}" for h in range(24)],
                tickfont=dict(size=9),
                gridcolor=C_BORDER,
            ),
            yaxis=dict(
                range=[0, 100],
                ticksuffix="%",
                gridcolor=C_BORDER,
                tickfont=dict(size=10),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})

    # ── Win rate by weekday ───────────────────────────────────────────────────
    with col2:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_stats = (
            df.groupby("weekday")
            .agg(win_rate=("win", "mean"), trades=("win", "count"), pnl=("pnl_usd", "sum"))
            .reindex(day_order)
            .dropna()
            .reset_index()
        )
        day_stats["win_rate_pct"] = day_stats["win_rate"] * 100
        day_stats["color"] = day_stats["win_rate_pct"].apply(_wr_color)

        fig_day = go.Figure(go.Bar(
            x=day_stats["weekday"].str[:3],
            y=day_stats["win_rate_pct"],
            marker_color=day_stats["color"],
            text=day_stats["trades"].apply(lambda n: f"{n}t"),
            textposition="outside",
            textfont=dict(size=9, color=C_MUTED),
            hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
        ))
        fig_day.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Win Rate by Day of Week", font=dict(color=C_GOLD, size=13)),
            height=300,
            yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor=C_BORDER, tickfont=dict(size=10)),
            xaxis=dict(gridcolor=C_BORDER, tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig_day, use_container_width=True, config={"displayModeBar": False})

    # ── Best/worst hour+day combos ────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>Best & Worst Time Combinations</div>", unsafe_allow_html=True)
    combo = (
        df.groupby(["hour", "weekday"])
        .agg(win_rate=("win", "mean"), trades=("win", "count"), total_pnl=("pnl_usd", "sum"))
        .reset_index()
    )
    combo["win_rate_pct"] = (combo["win_rate"] * 100).round(1)
    combo = combo[combo["trades"] >= 2]  # filter noise

    if not combo.empty:
        c1, c2 = st.columns(2)
        with c1:
            best = combo.nlargest(5, "total_pnl")[["hour", "weekday", "win_rate_pct", "trades", "total_pnl"]]
            best.columns = ["Hour (UTC)", "Day", "Win Rate", "Trades", "PnL ($)"]
            best["Hour (UTC)"] = best["Hour (UTC)"].apply(lambda h: f"{h:02d}:00")
            best["Win Rate"]   = best["Win Rate"].apply(lambda v: f"{v:.1f}%")
            best["PnL ($)"]   = best["PnL ($)"].apply(lambda v: f"+${v:.0f}" if v >= 0 else f"-${abs(v):.0f}")
            st.markdown(f"<div style='color:{C_GREEN}; font-size:12px; font-weight:600; margin-bottom:4px;'>Best 5 Combinations</div>", unsafe_allow_html=True)
            st.dataframe(best, use_container_width=True, hide_index=True)
        with c2:
            worst = combo.nsmallest(5, "total_pnl")[["hour", "weekday", "win_rate_pct", "trades", "total_pnl"]]
            worst.columns = ["Hour (UTC)", "Day", "Win Rate", "Trades", "PnL ($)"]
            worst["Hour (UTC)"] = worst["Hour (UTC)"].apply(lambda h: f"{h:02d}:00")
            worst["Win Rate"]   = worst["Win Rate"].apply(lambda v: f"{v:.1f}%")
            worst["PnL ($)"]   = worst["PnL ($)"].apply(lambda v: f"+${v:.0f}" if v >= 0 else f"-${abs(v):.0f}")
            st.markdown(f"<div style='color:{C_RED}; font-size:12px; font-weight:600; margin-bottom:4px;'>Worst 5 Combinations</div>", unsafe_allow_html=True)
            st.dataframe(worst, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough trade combinations for analysis (need ≥2 trades per time slot).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_dashboard() -> None:
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Install with: venv/bin/python -m pip install streamlit")
        return

    st.set_page_config(
        page_title="GoldSignalAI",
        page_icon="⬡",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "GoldSignalAI — XAU/USD Signal Bot"},
    )

    # Inject CSS + fonts
    st.markdown(_CSS, unsafe_allow_html=True)

    # Ensure DB tables exist
    _ensure_tables()

    # Sidebar (returns filter state)
    filters = render_sidebar()

    # Title bar
    st.markdown(
        f"<h1>⬡ GoldSignalAI — Trading Dashboard</h1>"
        f"<div style='color:{C_MUTED}; font-size:12px; margin-top:-12px; margin-bottom:16px;'>"
        f"{Config.SYMBOL_DISPLAY} · {Config.PRIMARY_TIMEFRAME}+{Config.CONFIRMATION_TIMEFRAME} · "
        f"Stage 14</div>",
        unsafe_allow_html=True,
    )

    # Tabs
    tabs = st.tabs([
        "📋 Trade History",
        "🧠 ML Status",
        "🌀 Regime Detection",
        "🏆 Challenge Progress",
        "⚡ Risk Monitor",
        "📊 Signal Heatmap",
    ])

    with tabs[0]:
        tab_trade_history(filters)

    with tabs[1]:
        tab_ml_status()

    with tabs[2]:
        tab_regime_detection()

    with tabs[3]:
        tab_challenge_progress(filters["account_balance"])

    with tabs[4]:
        tab_risk_monitor()

    with tabs[5]:
        tab_signal_heatmap(filters)


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_dashboard()
    else:
        print("Install streamlit: venv/bin/python -m pip install streamlit")
