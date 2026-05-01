"""
GoldSignalAI — database/db.py
==============================
Persistent SQLite storage for signals and trades.

Works alongside the existing JSON/CSV logs — this adds
crash-safe, queryable storage without replacing anything.

Tables:
  signals — every signal generated (BUY/SELL/WAIT)
  trades  — every trade opened, with outcome tracking

Usage:
    from database.db import initialize_database, save_signal, save_trade
    initialize_database()
    save_signal(signal_dict)
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(Config.BASE_DIR, "database", "goldsignalai.db")

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL DEFAULT 'XAUUSD',
    direction       TEXT NOT NULL,
    confidence      REAL NOT NULL,
    entry_price     REAL,
    bullish_count   INTEGER,
    bearish_count   INTEGER,
    ml_confirms     INTEGER,
    reason          TEXT,
    is_paused       INTEGER DEFAULT 0,
    forward_test    INTEGER DEFAULT 0,
    order_id        INTEGER DEFAULT NULL
)
"""

_MIGRATE_SIGNALS_FORWARD_TEST = """
ALTER TABLE signals ADD COLUMN forward_test INTEGER DEFAULT 0
"""

_MIGRATE_SIGNALS_ORDER_ID = """
ALTER TABLE signals ADD COLUMN order_id INTEGER DEFAULT NULL
"""

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL DEFAULT 'XAUUSD',
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    stop_loss       REAL,
    take_profit1    REAL,
    take_profit2    REAL,
    lot_size        REAL,
    status          TEXT NOT NULL DEFAULT 'open',
    result          TEXT,
    pnl_usd         REAL,
    pnl_pips        REAL,
    closed_at       TEXT
)
"""

_CREATE_PAPER_TRADES = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER,
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    sl_price        REAL NOT NULL,
    tp_price        REAL NOT NULL,
    tp2_price       REAL,
    lot_size        REAL NOT NULL DEFAULT 0.01,
    entry_time      TEXT NOT NULL,
    exit_price      REAL,
    exit_time       TEXT,
    exit_reason     TEXT,
    pnl_pct         REAL,
    pnl_dollar      REAL,
    outcome         TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_exit ON paper_trades(exit_time);
CREATE INDEX IF NOT EXISTS idx_paper_trades_outcome ON paper_trades(outcome);
"""


def _get_conn() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def initialize_database() -> bool:
    """
    Create tables if they don't exist.
    Returns True on success, False on failure.
    """
    try:
        conn = _get_conn()
        conn.execute(_CREATE_SIGNALS)
        conn.execute(_CREATE_TRADES)
        conn.execute(_CREATE_PAPER_TRADES)
        conn.executescript(_CREATE_INDEX)
        # Idempotent migrations for existing DBs.
        existing_cols = {row["name"] for row in
                         conn.execute("PRAGMA table_info(signals)").fetchall()}
        if "forward_test" not in existing_cols:
            try:
                conn.execute(_MIGRATE_SIGNALS_FORWARD_TEST)
            except Exception:
                pass
        if "order_id" not in existing_cols:
            try:
                conn.execute(_MIGRATE_SIGNALS_ORDER_ID)
            except Exception:
                pass
        conn.commit()
        conn.close()
        logger.info("Database initialized at %s", DB_PATH)
        return True
    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)
        return False


def save_signal(signal_data: dict) -> Optional[int]:
    """
    Insert a signal record. Returns the row ID, or None on failure.

    Expected keys: timestamp, direction, confidence_pct, entry_price,
    bullish_count, bearish_count, ml_confirms, reason, is_paused, forward_test,
    order_id (Stage 3: MT5 ticket from place_order on success, else None)
    """
    try:
        conn = _get_conn()
        cur = conn.execute(
            """INSERT INTO signals
               (timestamp, symbol, direction, confidence, entry_price,
                bullish_count, bearish_count, ml_confirms, reason, is_paused,
                forward_test, order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                signal_data.get("symbol", Config.SYMBOL),
                signal_data.get("direction", "WAIT"),
                signal_data.get("confidence_pct", 0.0),
                signal_data.get("entry_price"),
                signal_data.get("bullish_count", 0),
                signal_data.get("bearish_count", 0),
                1 if signal_data.get("ml_confirms") else 0,
                signal_data.get("reason", ""),
                1 if signal_data.get("is_paused") else 0,
                1 if signal_data.get("forward_test") else 0,
                signal_data.get("order_id"),
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id
    except Exception as exc:
        logger.error("Failed to save signal: %s", exc)
        return None


def count_forward_test_trades() -> int:
    """Return number of actionable signals logged with forward_test=1."""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM signals WHERE forward_test = 1 AND direction != 'WAIT'"
        ).fetchone()
        conn.close()
        return row["cnt"] if row else 0
    except Exception as exc:
        logger.warning("count_forward_test_trades failed: %s", exc)
        return 0


def save_trade(trade_data: dict) -> Optional[int]:
    """
    Insert a trade record. Returns the row ID, or None on failure.

    Expected keys: timestamp, direction, entry_price, stop_loss,
    take_profit1, take_profit2, lot_size, status
    """
    try:
        conn = _get_conn()
        cur = conn.execute(
            """INSERT INTO trades
               (timestamp, symbol, direction, entry_price, stop_loss,
                take_profit1, take_profit2, lot_size, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                trade_data.get("symbol", Config.SYMBOL),
                trade_data.get("direction"),
                trade_data.get("entry_price"),
                trade_data.get("stop_loss"),
                trade_data.get("take_profit1"),
                trade_data.get("take_profit2"),
                trade_data.get("lot_size"),
                trade_data.get("status", "open"),
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id
    except Exception as exc:
        logger.error("Failed to save trade: %s", exc)
        return None


def update_trade_result(trade_id: int, result: str, pnl_usd: float = 0.0, pnl_pips: float = 0.0) -> bool:
    """Update a trade's outcome. Returns True if a row was actually updated."""
    try:
        conn = _get_conn()
        cur = conn.execute(
            """UPDATE trades
               SET result = ?, status = 'closed', pnl_usd = ?, pnl_pips = ?,
                   closed_at = ?
               WHERE id = ?""",
            (result, pnl_usd, pnl_pips, datetime.now(timezone.utc).isoformat(), trade_id),
        )
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()
        if not updated:
            logger.warning("update_trade_result: trade_id=%d not found (0 rows updated)", trade_id)
        return updated
    except Exception as exc:
        logger.error("Failed to update trade %d: %s", trade_id, exc)
        return False


def has_recent_signal(direction: str, hours: float = 4.0) -> bool:
    """
    Check if a similar signal already exists within the last N hours.
    Used for duplicate prevention.

    Args:
        direction: "BUY" or "SELL"
        hours: lookback window (default 4 hours)

    Returns:
        True if a duplicate exists (should skip), False if clear to send.
    """
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        conn = _get_conn()
        row = conn.execute(
            """SELECT COUNT(*) as cnt FROM signals
               WHERE direction = ? AND timestamp > ? AND direction != 'WAIT'""",
            (direction, cutoff),
        ).fetchone()
        conn.close()
        return row["cnt"] > 0
    except Exception as exc:
        logger.warning("Duplicate check failed (allowing signal): %s", exc)
        return False


def get_recent_signals(limit: int = 50) -> list[dict]:
    """Return the most recent signals for dashboard display."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to fetch recent signals: %s", exc)
        return []


def get_open_trades() -> list[dict]:
    """Return all trades with status 'open'."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY id DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to fetch open trades: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADING (Stage 4)
# ─────────────────────────────────────────────────────────────────────────────
# Paper trades exit on TP1, SL, or 48-bar TIME exit. tp2_price is recorded
# observationally only — exit logic never reads it.

def open_paper_trade(
    signal_id: Optional[int],
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    entry_time: str,
    lot_size: float,
    tp2_price: Optional[float] = None,
) -> Optional[int]:
    """Insert a new open paper trade. Returns row id."""
    try:
        conn = _get_conn()
        cur = conn.execute(
            """INSERT INTO paper_trades
               (signal_id, direction, entry_price, sl_price, tp_price, tp2_price,
                lot_size, entry_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal_id, direction, entry_price, sl_price, tp_price, tp2_price,
             lot_size, entry_time),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id
    except Exception as exc:
        logger.error("Failed to open paper trade: %s", exc)
        return None


def close_paper_trade(
    trade_id: int,
    exit_price: float,
    exit_time: str,
    exit_reason: str,
    pnl_pct: float,
    pnl_dollar: float,
    outcome: str,
) -> bool:
    """Update an open paper trade with exit details. Returns True on success."""
    try:
        conn = _get_conn()
        cur = conn.execute(
            """UPDATE paper_trades
               SET exit_price = ?, exit_time = ?, exit_reason = ?,
                   pnl_pct = ?, pnl_dollar = ?, outcome = ?
               WHERE id = ? AND exit_time IS NULL""",
            (exit_price, exit_time, exit_reason, pnl_pct, pnl_dollar, outcome,
             trade_id),
        )
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()
        if not updated:
            logger.warning("close_paper_trade: id=%d already closed or missing",
                           trade_id)
        return updated
    except Exception as exc:
        logger.error("Failed to close paper trade %d: %s", trade_id, exc)
        return False


def get_open_paper_trades() -> list[dict]:
    """Return all paper trades that have not been closed yet."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE exit_time IS NULL ORDER BY id ASC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to fetch open paper trades: %s", exc)
        return []


def get_paper_trade_stats() -> dict:
    """
    Aggregate stats over closed paper trades.
    Returns: trades, wins, losses, win_rate_pct, profit_factor, avg_pnl_dollar,
             total_pnl_dollar, max_drawdown_dollar.
    """
    empty = {
        "trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
        "profit_factor": 0.0, "avg_pnl_dollar": 0.0, "total_pnl_dollar": 0.0,
        "max_drawdown_dollar": 0.0,
    }
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT pnl_dollar, outcome FROM paper_trades
               WHERE exit_time IS NOT NULL ORDER BY exit_time ASC"""
        ).fetchall()
        conn.close()
        if not rows:
            return empty
        wins = sum(1 for r in rows if r["outcome"] == "WIN")
        losses = sum(1 for r in rows if r["outcome"] == "LOSS")
        n = len(rows)
        gross_win = sum(r["pnl_dollar"] for r in rows
                        if r["pnl_dollar"] is not None and r["pnl_dollar"] > 0)
        gross_loss = sum(-r["pnl_dollar"] for r in rows
                         if r["pnl_dollar"] is not None and r["pnl_dollar"] < 0)
        total = sum(r["pnl_dollar"] or 0.0 for r in rows)
        # Cumulative-equity drawdown (dollar units).
        peak = 0.0
        cum = 0.0
        max_dd = 0.0
        for r in rows:
            cum += (r["pnl_dollar"] or 0.0)
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        return {
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": (wins / n * 100.0) if n else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (
                float("inf") if gross_win > 0 else 0.0),
            "avg_pnl_dollar": total / n if n else 0.0,
            "total_pnl_dollar": total,
            "max_drawdown_dollar": max_dd,
        }
    except Exception as exc:
        logger.error("get_paper_trade_stats failed: %s", exc)
        return empty
