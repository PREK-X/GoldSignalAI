"""
GoldSignalAI — infrastructure/circuit_breaker.py
==================================================
Multi-level circuit breaker for drawdown protection.

State machine with 4 daily levels (checked in order):
  Level 0 (NORMAL):     daily_loss < 2%  -> size_multiplier=1.0, all signals
  Level 1 (CAUTION):    daily_loss >= 2% -> size_multiplier=0.5, all signals
  Level 2 (RESTRICTED): daily_loss >= 3% -> size_multiplier=0.5, confidence > 80% only
  Level 3 (HALTED):     daily_loss >= 4% -> size_multiplier=0.0, no new trades

Total DD override (on top of daily):
  total_dd >= 8% -> multiply by 0.25 (until total_dd recovers below 5%)
"""

# Daily loss thresholds (% of account)
_CAUTION_PCT = 2.0
_RESTRICTED_PCT = 3.0
_HALTED_PCT = 4.0

# Restricted level minimum confidence
_RESTRICTED_MIN_CONFIDENCE = 80.0

# Total drawdown thresholds
_TOTAL_DD_TRIGGER_PCT = 8.0
_TOTAL_DD_RECOVER_PCT = 5.0
_TOTAL_DD_MULTIPLIER = 0.25

# State labels
NORMAL = "NORMAL"
CAUTION = "CAUTION"
RESTRICTED = "RESTRICTED"
HALTED = "HALTED"


class CircuitBreaker:
    """
    Stateful circuit breaker that tracks the total-DD override latch.

    The total-DD override activates at >= 8% and stays active until
    total DD recovers below 5% (hysteresis to prevent flapping).
    """

    def __init__(self):
        self._total_dd_override_active = False

    def reset(self):
        """Reset total-DD override (e.g. for a new backtest run)."""
        self._total_dd_override_active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_circuit_state(self, daily_pnl_pct: float, total_dd_pct: float) -> str:
        """
        Return the current circuit breaker state label.

        Args:
            daily_pnl_pct: Today's PnL as % of account (negative = loss).
            total_dd_pct:  Current drawdown from peak as % (positive number).
        """
        daily_loss = abs(min(0.0, daily_pnl_pct))

        if daily_loss >= _HALTED_PCT:
            return HALTED
        if daily_loss >= _RESTRICTED_PCT:
            return RESTRICTED
        if daily_loss >= _CAUTION_PCT:
            return CAUTION
        return NORMAL

    def get_size_multiplier(self, daily_pnl_pct: float, total_dd_pct: float) -> float:
        """
        Return the position-size multiplier (0.0 – 1.0).

        Combines the daily-level multiplier with the total-DD override.
        """
        state = self.get_circuit_state(daily_pnl_pct, total_dd_pct)

        if state == HALTED:
            base = 0.0
        elif state in (CAUTION, RESTRICTED):
            base = 0.5
        else:
            base = 1.0

        # Total DD override with hysteresis
        self._update_total_dd_latch(total_dd_pct)
        if self._total_dd_override_active:
            base *= _TOTAL_DD_MULTIPLIER

        return base

    def is_signal_allowed(
        self,
        daily_pnl_pct: float,
        total_dd_pct: float,
        confidence: float,
    ) -> bool:
        """
        Return True if a new signal is allowed under current conditions.

        Args:
            daily_pnl_pct: Today's PnL as % of account (negative = loss).
            total_dd_pct:  Current drawdown from peak as % (positive number).
            confidence:    Signal confidence in % (0-100).
        """
        state = self.get_circuit_state(daily_pnl_pct, total_dd_pct)

        if state == HALTED:
            return False
        if state == RESTRICTED and confidence < _RESTRICTED_MIN_CONFIDENCE:
            return False
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_total_dd_latch(self, total_dd_pct: float):
        """Update the total-DD override latch with hysteresis."""
        if self._total_dd_override_active:
            if total_dd_pct < _TOTAL_DD_RECOVER_PCT:
                self._total_dd_override_active = False
        else:
            if total_dd_pct >= _TOTAL_DD_TRIGGER_PCT:
                self._total_dd_override_active = True

    @property
    def total_dd_override_active(self) -> bool:
        return self._total_dd_override_active


# Module-level convenience functions (stateless — no total-DD hysteresis)

def get_circuit_state(daily_pnl_pct: float, total_dd_pct: float) -> str:
    """Stateless version — does not track total-DD override latch."""
    daily_loss = abs(min(0.0, daily_pnl_pct))
    if daily_loss >= _HALTED_PCT:
        return HALTED
    if daily_loss >= _RESTRICTED_PCT:
        return RESTRICTED
    if daily_loss >= _CAUTION_PCT:
        return CAUTION
    return NORMAL


def get_size_multiplier(daily_pnl_pct: float, total_dd_pct: float) -> float:
    """Stateless version — applies total-DD override without hysteresis."""
    state = get_circuit_state(daily_pnl_pct, total_dd_pct)
    if state == HALTED:
        base = 0.0
    elif state in (CAUTION, RESTRICTED):
        base = 0.5
    else:
        base = 1.0
    if total_dd_pct >= _TOTAL_DD_TRIGGER_PCT:
        base *= _TOTAL_DD_MULTIPLIER
    return base


def is_signal_allowed(
    daily_pnl_pct: float,
    total_dd_pct: float,
    confidence: float,
) -> bool:
    """Stateless version — no total-DD hysteresis."""
    state = get_circuit_state(daily_pnl_pct, total_dd_pct)
    if state == HALTED:
        return False
    if state == RESTRICTED and confidence < _RESTRICTED_MIN_CONFIDENCE:
        return False
    return True
