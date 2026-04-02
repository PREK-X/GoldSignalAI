"""
GoldSignalAI -- state/state_manager.py
=======================================
Persistent runtime state for the live signal generator.

Tracks session-level statistics (consecutive losses, current session date)
that survive bot restarts via JSON serialisation.

The backtest engine manages these counters internally (local variables in
run_backtest). This module is for the *live* generator + main loop only.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(Config.BASE_DIR, "state", "state.json")


class StateManager:
    """
    JSON-backed runtime state.

    Persisted fields:
        session_consecutive_losses  -- reset at each new NY session day
        session_date                -- "YYYY-MM-DD" of the current session
        last_signal_direction       -- most recent emitted signal direction
        last_signal_time            -- ISO timestamp of most recent signal
    """

    def __init__(self, state_file: str = STATE_FILE):
        self._path = state_file
        self.session_consecutive_losses: int = 0
        self.session_date: str = ""
        self.last_signal_direction: str = ""
        self.last_signal_time: str = ""
        self._load()

    # ── Session loss tracking ─────────────────────────────────────────────

    def get_session_losses(self) -> int:
        return self.session_consecutive_losses

    def increment_session_loss(self, signal_time: datetime) -> None:
        """Increment consecutive loss counter, resetting if new session day."""
        day_str = signal_time.strftime("%Y-%m-%d")
        if day_str != self.session_date:
            self.session_consecutive_losses = 0
            self.session_date = day_str
        self.session_consecutive_losses += 1
        self._save()

    def reset_session_losses(self) -> None:
        self.session_consecutive_losses = 0
        self._save()

    def register_trade_outcome(
        self,
        trade_id: str,
        outcome: str,
        signal_time: datetime,
    ) -> None:
        """
        Record the result of a completed trade.

        Args:
            trade_id:    Unique trade identifier (ticket or timestamp).
            outcome:     "WIN" or "LOSS".
            signal_time: UTC datetime of the original signal.
        """
        if outcome == "LOSS":
            self.increment_session_loss(signal_time)
        elif outcome == "WIN":
            # A win resets the consecutive loss streak
            day_str = signal_time.strftime("%Y-%m-%d")
            if day_str != self.session_date:
                self.session_date = day_str
            self.session_consecutive_losses = 0
            self._save()

    # ── Signal tracking ───────────────────────────────────────────────────

    def record_signal(self, direction: str, signal_time: datetime) -> None:
        """Record that a signal was emitted (for state continuity)."""
        self.last_signal_direction = direction
        self.last_signal_time = signal_time.isoformat()
        self._save()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self.session_consecutive_losses = data.get("session_consecutive_losses", 0)
            self.session_date = data.get("session_date", "")
            self.last_signal_direction = data.get("last_signal_direction", "")
            self.last_signal_time = data.get("last_signal_time", "")
        except Exception as exc:
            logger.warning("Failed to load state from %s: %s", self._path, exc)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            data = {
                "session_consecutive_losses": self.session_consecutive_losses,
                "session_date": self.session_date,
                "last_signal_direction": self.last_signal_direction,
                "last_signal_time": self.last_signal_time,
            }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save state to %s: %s", self._path, exc)
