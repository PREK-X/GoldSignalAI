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
    is_paused       INTEGER DEFAULT 0
)
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

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
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
        conn.executescript(_CREATE_INDEX)
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
    bullish_count, bearish_count, ml_confirms, reason, is_paused
    """
    try:
        conn = _get_conn()
        cur = conn.execute(
            """INSERT INTO signals
               (timestamp, symbol, direction, confidence, entry_price,
                bullish_count, bearish_count, ml_confirms, reason, is_paused)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id
    except Exception as exc:
        logger.error("Failed to save signal: %s", exc)
        return None


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
    """Update a trade's outcome. Returns True on success."""
    try:
        conn = _get_conn()
        conn.execute(
            """UPDATE trades
               SET result = ?, status = 'closed', pnl_usd = ?, pnl_pips = ?,
                   closed_at = ?
               WHERE id = ?""",
            (result, pnl_usd, pnl_pips, datetime.now(timezone.utc).isoformat(), trade_id),
        )
        conn.commit()
        conn.close()
        return True
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
